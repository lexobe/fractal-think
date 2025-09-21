"""
核心类型定义

包含分形思考框架的基础数据结构和枚举类型。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import json


class SolveStatus(Enum):
    """求解状态枚举"""
    COMPLETED = "completed"       # 正常完成
    FAILED = "failed"             # 异常失败


@dataclass
class TokenUsage:
    """Token使用统计"""
    total: int = 0                # 总消耗
    think_calls: int = 0          # Think调用次数
    eval_calls: int = 0           # Eval调用次数
    think_tokens: int = 0         # Think消耗
    eval_tokens: int = 0          # Eval消耗


@dataclass
class SolveResult:
    """统一的求解结果对象"""
    status: SolveStatus           # 执行状态
    result: str                   # 最终结果描述
    token_usage: TokenUsage       # Token消耗统计
    execution_time: float         # 执行时间（秒）
    max_depth_reached: int        # 达到的最大递归深度
    constraint_triggered: Optional[str] = None  # 触发的约束类型
    partial_results: List[str] = field(default_factory=list)  # 兼容旧API
    failure_path: List[str] = field(default_factory=list)     # 兼容旧API
    failure_level: Optional[int] = None                      # 兼容旧API
    failure_node_goal: Optional[str] = None                  # 兼容旧API
    failure_node_done: List[str] = field(default_factory=list)


class ConstraintViolationError(RuntimeError):
    """约束违反基类"""

    def __init__(self, message: str, *, node: Optional['S'] = None):
        super().__init__(message)
        self.node = node


class MaxDepthExceeded(ConstraintViolationError):
    """深度约束违反"""


class ResourceExhausted(ConstraintViolationError):
    """资源约束违反"""


class ExecutionTimeout(ConstraintViolationError):
    """时间约束违反"""


# 兼容性别名，保持向后兼容
DepthLimitExceeded = MaxDepthExceeded
ResourceLimitExceeded = ResourceExhausted
TimeLimitExceeded = ExecutionTimeout


class S:
    """
    状态结构 S = {goal, parent, todo, done}

    按照 thinkon.md 规范实现：
    - goal: 当前层目的，定义本层的求解目标
    - parent: 父节点引用（根节点为None），形成递归调用栈
    - todo: 纯自然语言计划文本，支持活性计划描述
    - done: 完成历史列表，每个条目存储子层返回的结果字符串
    """

    def __init__(self, goal: str, parent: Optional['S'] = None, todo: str = "", done: Optional[List[str]] = None):
        self.goal = goal
        self.parent = parent
        self.todo = todo
        self.done = done if done is not None else []

        # 计算派生字段
        self.level = 0 if parent is None else parent.level + 1
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，支持JSON序列化"""
        return {
            "goal": self.goal,
            "todo": self.todo,
            "done": self.done.copy(),
            "level": self.level,
            "timestamp": self.timestamp,
            "parent_goal": self.parent.goal if self.parent else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], parent: Optional['S'] = None) -> 'S':
        """从字典恢复S对象"""
        node = cls(
            goal=data["goal"],
            parent=parent,
            todo=data.get("todo", ""),
            done=data.get("done", [])
        )
        return node

    def __repr__(self) -> str:
        return f"S(goal='{self.goal}', level={self.level}, done={len(self.done)})"
