"""
Fractal Think - 分形思考异步执行框架

高效的异步分形思考执行引擎，支持复杂问题的递归分解和并行处理。

主要功能:
- 异步Think/Eval算子执行
- 智能计划解析和子任务管理
- 预算约束和Token统计
- 同步适配器向后兼容
"""

# 主要API导出
from .engine import solve_async
from .types import (
    S,
    SolveResult,
    SolveStatus,
    TokenUsage,
    ConstraintViolationError,
    DepthLimitExceeded,
    ResourceLimitExceeded,
    TimeLimitExceeded,
)
from .common import ExecutionBudget, ExecutionMode
from .interfaces import AsyncThinkLLM, AsyncEvalLLM, ThinkLLM, EvalLLM
from .frame import ExecutionFrame, FrameState
from .frame_stack import (
    FrameStackEntry,
    FrameStackProtocolError,
    frame_stack_to_json,
)
from .memory import Memory
from . import examples

# 同步适配器（可选）
try:
    from .sync_adapter import solve_with_async_engine
    _SYNC_AVAILABLE = True
except ImportError:
    _SYNC_AVAILABLE = False

__version__ = "1.0.0"
__author__ = "Fractal Think Team"

# 主要接口
__all__ = [
    # 核心函数
    "solve_async",

    # 数据类型
    "S",
    "SolveResult",
    "SolveStatus",
    "TokenUsage",
    "ExecutionBudget",
    "ExecutionFrame",
    "FrameStackEntry",
    "FrameStackProtocolError",
    "Memory",

    # 协议接口
    "AsyncThinkLLM",
    "AsyncEvalLLM",
    "ThinkLLM",
    "EvalLLM",

    # 枚举
    "ExecutionMode",
    "FrameState",

    # 示例模块
    "examples",

    # 约束异常
    "ConstraintViolationError",
    "DepthLimitExceeded",
    "ResourceLimitExceeded",
    "TimeLimitExceeded",
]

# 条件导出同步适配器
if _SYNC_AVAILABLE:
    __all__.append("solve_with_async_engine")

def get_version() -> str:
    """获取版本信息"""
    return __version__

def is_sync_available() -> bool:
    """检查同步适配器是否可用"""
    return _SYNC_AVAILABLE
