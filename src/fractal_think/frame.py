"""
执行帧数据结构 - 显式栈式状态机的核心组件
"""

import time
from typing import Optional, List, Any, Dict
from dataclasses import dataclass, field
from enum import Enum

from .types import S


class FrameState(Enum):
    """执行帧状态"""
    THINK = "think"                   # Think阶段
    PLANNING = "planning"             # Plan_made阶段
    FIRST_EVAL = "first_eval"         # 首启阶段
    EVAL = "eval"                     # 续步阶段
    RETURNING = "returning"           # Return收束
    FAILED = "failed"                 # 失败状态


@dataclass
class SubTaskInfo:
    """子任务信息"""
    task_id: str
    description: str
    created_at: float = field(default_factory=time.time)
    status: str = "pending"


@dataclass
class ExecutionFrame:
    """
    执行帧 - 异步状态机的基础单元

    每个ExecutionFrame对应一个S节点的完整执行过程，
    包含状态跟踪、子任务管理和Token统计。
    """
    frame_id: str
    node: S
    depth: int
    state: FrameState = FrameState.THINK
    parent_frame_id: Optional[str] = None

    # 执行状态
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    # Think/Eval结果缓存
    think_action: Optional[Dict[str, Any]] = None
    eval_action: Optional[Dict[str, Any]] = None

    # 子任务管理
    subtasks: List[SubTaskInfo] = field(default_factory=list)
    current_subtask_index: int = 0

    # Token统计
    think_tokens: int = 0
    eval_tokens: int = 0

    # 失败处理
    failure_reason: Optional[str] = None

    def add_subtask(self, description: str) -> SubTaskInfo:
        """添加子任务"""
        task_info = SubTaskInfo(
            task_id=f"{self.frame_id}_subtask_{len(self.subtasks)}",
            description=description
        )
        self.subtasks.append(task_info)
        self.last_updated = time.time()
        return task_info

    def get_current_subtask(self) -> Optional[SubTaskInfo]:
        """获取当前子任务"""
        if 0 <= self.current_subtask_index < len(self.subtasks):
            return self.subtasks[self.current_subtask_index]
        return None

    def advance_subtask(self) -> bool:
        """推进到下一个子任务"""
        if self.current_subtask_index < len(self.subtasks) - 1:
            self.current_subtask_index += 1
            self.last_updated = time.time()
            return True
        return False

    def add_frame_tokens(self, think_tokens: int = 0, eval_tokens: int = 0):
        """添加Token消耗"""
        self.think_tokens += think_tokens
        self.eval_tokens += eval_tokens
        self.last_updated = time.time()

    @property
    def total_tokens(self) -> int:
        """总Token消耗"""
        return self.think_tokens + self.eval_tokens

    def set_failed(self, reason: str):
        """设置失败状态"""
        self.state = FrameState.FAILED
        self.failure_reason = reason
        self.last_updated = time.time()

    def is_completed(self) -> bool:
        """检查是否完成"""
        return self.state in (FrameState.RETURNING, FrameState.FAILED)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "frame_id": self.frame_id,
            "node": self.node.to_dict(),
            "depth": self.depth,
            "state": self.state.value,
            "parent_frame_id": self.parent_frame_id,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "subtasks": [
                {
                    "task_id": task.task_id,
                    "description": task.description,
                    "created_at": task.created_at,
                    "status": task.status
                }
                for task in self.subtasks
            ],
            "current_subtask_index": self.current_subtask_index,
            "think_tokens": self.think_tokens,
            "eval_tokens": self.eval_tokens,
            "failure_reason": self.failure_reason
        }

    def __repr__(self) -> str:
        return (f"ExecutionFrame(id={self.frame_id[:8]}, "
                f"goal='{self.node.goal[:30]}...', "
                f"state={self.state.value}, "
                f"depth={self.depth})")