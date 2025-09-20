"""
公用工具组件 - 供同步/异步实现共享
"""

import time
import logging
from typing import Optional, List, Any, Dict, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio

try:
    from ..thinkon_core import S, SolveStatus, TokenUsage
except ImportError:
    # 当作为顶层模块运行时的回退导入
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from thinkon_core import S, SolveStatus, TokenUsage


class ExecutionMode(Enum):
    """执行模式"""
    SYNC = "sync"       # 同步模式
    ASYNC = "async"     # 异步模式


@dataclass
class ExecutionBudget:
    """执行预算 - 支持动态调整的全局约束管理"""
    max_depth: int = 10
    max_tokens: int = 10000
    max_time: float = 60.0

    # 当前消耗（由BudgetManager维护）
    spent_depth: int = 0
    spent_tokens: int = 0
    start_time: float = field(default_factory=time.time)

    def elapsed_time(self) -> float:
        """计算已消耗时间"""
        return time.time() - self.start_time

    def is_depth_exceeded(self, current_depth: int) -> bool:
        """检查深度是否超限"""
        return current_depth >= self.max_depth

    def is_tokens_exceeded(self, current_tokens: int) -> bool:
        """检查token是否超限"""
        return current_tokens >= self.max_tokens

    def is_time_exceeded(self) -> bool:
        """检查时间是否超限"""
        return self.elapsed_time() >= self.max_time

    def adjust_limits(self, max_depth: Optional[int] = None,
                     max_tokens: Optional[int] = None,
                     max_time: Optional[float] = None):
        """动态调整预算上限（用于恢复续跑）"""
        if max_depth is not None:
            self.max_depth = max_depth
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if max_time is not None:
            self.max_time = max_time

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'max_depth': self.max_depth,
            'max_tokens': self.max_tokens,
            'max_time': self.max_time,
            'spent_depth': self.spent_depth,
            'spent_tokens': self.spent_tokens,
            'start_time': self.start_time
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionBudget':
        """从字典反序列化"""
        return cls(**data)


class BudgetManager:
    """预算管理器 - 协程安全的约束检查"""

    def __init__(self, budget: ExecutionBudget):
        self.budget = budget
        self._lock = None  # 会在第一次异步调用时初始化

    async def check_constraints_async(self, node: S, tokens: int) -> Optional[str]:
        """异步约束检查"""
        # 懒初始化锁
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            return self._check_constraints_impl(node, tokens)

    def check_constraints_sync(self, node: S, tokens: int) -> Optional[str]:
        """同步约束检查"""
        return self._check_constraints_impl(node, tokens)

    def _check_constraints_impl(self, node: S, tokens: int) -> Optional[str]:
        """约束检查实现"""
        if self.budget.is_depth_exceeded(node.level):
            return f"DepthLimitExceeded: {node.level} >= {self.budget.max_depth}"

        if self.budget.is_tokens_exceeded(tokens):
            return f"ResourceLimitExceeded: {tokens} >= {self.budget.max_tokens}"

        if self.budget.is_time_exceeded():
            elapsed = self.budget.elapsed_time()
            return f"TimeLimitExceeded: {elapsed:.2f}s >= {self.budget.max_time}s"

        return None

    async def consume_budget_async(self, depth_delta: int = 0, tokens_delta: int = 0):
        """异步消费预算"""
        # 懒初始化锁
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            self._consume_budget_impl(depth_delta, tokens_delta)

    def consume_budget_sync(self, depth_delta: int = 0, tokens_delta: int = 0):
        """同步消费预算"""
        self._consume_budget_impl(depth_delta, tokens_delta)

    def _consume_budget_impl(self, depth_delta: int, tokens_delta: int):
        """预算消费实现"""
        self.budget.spent_depth += depth_delta
        self.budget.spent_tokens += tokens_delta


class UnifiedTokenUsage(TokenUsage):
    """统一的Token使用统计 - 扩展原有TokenUsage"""

    def __init__(self):
        super().__init__()
        self._lock = None  # 异步模式下会设置

    def set_async_mode(self):
        """设置为异步模式"""
        self._lock = asyncio.Lock()

    async def add_think_async(self, tokens: int):
        """异步添加Think统计"""
        # 懒初始化锁
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            self.add_think(tokens)

    async def add_eval_async(self, tokens: int):
        """异步添加Eval统计"""
        # 懒初始化锁
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            self.add_eval(tokens)

    async def merge_async(self, other: TokenUsage):
        """异步合并统计"""
        # 懒初始化锁
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            self.merge(other)


class UnifiedLogger:
    """统一日志系统 - 同步/异步兼容"""

    def __init__(self, logger: Optional[logging.Logger] = None,
                 mode: ExecutionMode = ExecutionMode.SYNC):
        self.logger = logger
        self.mode = mode
        self._async_queue = None

        if mode == ExecutionMode.ASYNC and logger:
            # 异步模式下使用队列缓冲日志
            self._async_queue = asyncio.Queue()

    def info(self, msg: str):
        """信息日志"""
        self._log("info", msg)

    def debug(self, msg: str):
        """调试日志"""
        self._log("debug", msg)

    def warning(self, msg: str):
        """警告日志"""
        self._log("warning", msg)

    def error(self, msg: str):
        """错误日志"""
        self._log("error", msg)

    def _log(self, level: str, msg: str):
        """实际日志记录"""
        if not self.logger:
            return

        if self.mode == ExecutionMode.ASYNC and self._async_queue:
            # 异步模式：放入队列，避免阻塞
            try:
                self._async_queue.put_nowait((level, msg))
            except asyncio.QueueFull:
                pass  # 队列满时丢弃日志
        else:
            # 同步模式：直接记录
            log_method = getattr(self.logger, level, None)
            if log_method:
                log_method(msg)

    async def flush_async_logs(self):
        """刷新异步日志队列"""
        if self.mode != ExecutionMode.ASYNC or not self._async_queue:
            return

        while not self._async_queue.empty():
            try:
                level, msg = self._async_queue.get_nowait()
                log_method = getattr(self.logger, level, None)
                if log_method:
                    log_method(msg)
            except asyncio.QueueEmpty:
                break


def build_failure_path(node: S) -> List[str]:
    """构建失败路径 - 从公用工具中提取"""
    path = []
    current = node
    while current is not None:
        path.append(current.goal)
        current = current.parent
    return list(reversed(path))


def format_constraint_error(constraint_type: str, node: S, details: str) -> str:
    """格式化约束错误信息"""
    return f"""约束触发：{constraint_type}
失败路径：{' -> '.join(build_failure_path(node))}
失败层级：Level {node.level}
节点目标：{node.goal}
详细信息：{details}"""


# 兼容性适配器
def convert_legacy_constraints(constraints) -> ExecutionBudget:
    """将旧的Constraints转换为ExecutionBudget"""
    if hasattr(constraints, 'max_depth'):
        return ExecutionBudget(
            max_depth=constraints.max_depth,
            max_tokens=constraints.max_tokens,
            max_time=constraints.max_time
        )
    return ExecutionBudget()