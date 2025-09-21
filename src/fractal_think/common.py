"""
公用工具组件 - 预算管理、Token统计和日志工具
"""

import time
import logging
from typing import Optional, List, Any, Dict, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from .types import (
    S,
    TokenUsage,
    MaxDepthExceeded,
    ResourceExhausted,
    ExecutionTimeout,
    ConstraintViolationError,
)


class ExecutionMode(Enum):
    """执行模式"""
    SYNC = "sync"       # 同步模式
    ASYNC = "async"     # 异步模式


@dataclass
class ExecutionBudget:
    """执行预算 - 全局约束管理"""
    max_depth: int = 10           # 最大递归深度
    max_tokens: int = 10000       # 最大Token消耗
    max_time: float = 60.0        # 最大执行时间（秒）

    def __post_init__(self):
        """验证预算参数"""
        if self.max_depth <= 0:
            raise ValueError("max_depth 必须大于0")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens 必须大于0")
        if self.max_time <= 0:
            raise ValueError("max_time 必须大于0")


class UnifiedTokenUsage:
    """统一的Token使用统计器"""

    def __init__(self):
        self.think_calls = 0
        self.eval_calls = 0
        self.think_tokens = 0
        self.eval_tokens = 0
        self._lock = None

    async def _ensure_lock(self):
        """确保异步锁已初始化"""
        if self._lock is None:
            self._lock = asyncio.Lock()

    async def add_think_async(self, tokens: int):
        """异步添加Think Token"""
        await self._ensure_lock()
        async with self._lock:
            self.think_calls += 1
            self.think_tokens += tokens

    async def add_eval_async(self, tokens: int):
        """异步添加Eval Token"""
        await self._ensure_lock()
        async with self._lock:
            self.eval_calls += 1
            self.eval_tokens += tokens

    def add_think_sync(self, tokens: int):
        """同步添加Think Token"""
        self.think_calls += 1
        self.think_tokens += tokens

    def add_eval_sync(self, tokens: int):
        """同步添加Eval Token"""
        self.eval_calls += 1
        self.eval_tokens += tokens

    @property
    def total(self) -> int:
        """总Token消耗"""
        return self.think_tokens + self.eval_tokens

    def to_token_usage(self) -> TokenUsage:
        """转换为TokenUsage对象"""
        return TokenUsage(
            total=self.total,
            think_calls=self.think_calls,
            eval_calls=self.eval_calls,
            think_tokens=self.think_tokens,
            eval_tokens=self.eval_tokens
        )


class BudgetManager:
    """预算管理器"""

    def __init__(self, budget: ExecutionBudget):
        self.budget = budget
        self.start_time = time.time()
        self.consumed_tokens = 0
        self._lock = None

    async def _ensure_lock(self):
        """确保异步锁已初始化"""
        if self._lock is None:
            self._lock = asyncio.Lock()

    async def check_constraints_async(self, node: S, tokens_delta: int = 0) -> None:
        """异步检查约束"""
        await self._ensure_lock()
        async with self._lock:
            self._check_constraints_impl(node, tokens_delta)

    def check_constraints_sync(self, node: S, tokens_delta: int = 0) -> None:
        """同步检查约束"""
        self._check_constraints_impl(node, tokens_delta)

    def _check_constraints_impl(self, node: S, tokens_delta: int = 0) -> None:
        """约束检查实现"""
        # 检查深度限制
        if node.level >= self.budget.max_depth:
            raise MaxDepthExceeded(
                f"MaxDepthExceeded: 当前递归深度 {node.level} >= 最大深度 {self.budget.max_depth}",
                node=node,
            )

        # 检查Token限制
        projected_tokens = self.consumed_tokens + tokens_delta
        if projected_tokens >= self.budget.max_tokens:
            raise ResourceExhausted(
                f"ResourceExhausted: 已消耗token数量 {projected_tokens} >= 预算 {self.budget.max_tokens}",
                node=node,
            )

        # 检查时间限制
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.budget.max_time:
            raise ExecutionTimeout(
                f"ExecutionTimeout: 执行时长 {elapsed_time:.2f}s >= 限制 {self.budget.max_time}s",
                node=node,
            )

    async def consume_budget_async(self, tokens_delta: int = 0):
        """异步消耗预算"""
        await self._ensure_lock()
        async with self._lock:
            self.consumed_tokens += tokens_delta

    def consume_budget_sync(self, tokens_delta: int = 0):
        """同步消耗预算"""
        self.consumed_tokens += tokens_delta


class UnifiedLogger:
    """统一日志器"""

    def __init__(self, logger: Optional[logging.Logger] = None, mode: ExecutionMode = ExecutionMode.ASYNC):
        self.logger = logger or self._create_default_logger()
        self.mode = mode

    def _create_default_logger(self) -> logging.Logger:
        """创建默认日志器"""
        logger = logging.getLogger("fractal_think")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def debug(self, message: str, **kwargs):
        """调试信息"""
        self.logger.debug(f"[{self.mode.value.upper()}] {message}", **kwargs)

    def info(self, message: str, **kwargs):
        """普通信息"""
        self.logger.info(f"[{self.mode.value.upper()}] {message}", **kwargs)

    def warning(self, message: str, **kwargs):
        """警告信息"""
        self.logger.warning(f"[{self.mode.value.upper()}] {message}", **kwargs)

    def error(self, message: str, **kwargs):
        """错误信息"""
        self.logger.error(f"[{self.mode.value.upper()}] {message}", **kwargs)


def convert_legacy_constraints(constraints: Any) -> ExecutionBudget:
    """
    转换遗留约束对象为ExecutionBudget

    支持从旧的Constraints对象或字典转换
    """
    if isinstance(constraints, ExecutionBudget):
        return constraints

    # 处理字典格式
    if isinstance(constraints, dict):
        return ExecutionBudget(
            max_depth=constraints.get("max_depth", 10),
            max_tokens=constraints.get("max_tokens", 10000),
            max_time=constraints.get("max_time", 60.0)
        )

    # 处理对象格式（假设有相应属性）
    try:
        return ExecutionBudget(
            max_depth=getattr(constraints, "max_depth", 10),
            max_tokens=getattr(constraints, "max_tokens", 10000),
            max_time=getattr(constraints, "max_time", 60.0)
        )
    except Exception:
        # 回退到默认值
        return ExecutionBudget()
