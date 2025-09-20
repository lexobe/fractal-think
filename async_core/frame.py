"""
执行帧数据结构 - 显式栈式状态机的核心组件
"""

import time
from typing import Optional, List, Any, Dict, Union
from dataclasses import dataclass, field
from enum import Enum

try:
    from ..thinkon_core import S, ReturnAction, PlanAction, CallAction
except ImportError:
    # 当作为顶层模块运行时的回退导入
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from thinkon_core import S, ReturnAction, PlanAction, CallAction


class FrameState(Enum):
    """执行帧状态"""
    CREATED = "created"           # 刚创建
    THINKING = "thinking"         # Think阶段
    PLANNING = "planning"         # Plan_made阶段
    FIRST_EVAL = "first_eval"     # 首启阶段
    CALLING = "calling"           # 调用子任务中
    WAITING = "waiting"           # 等待子任务完成
    CONTINUING = "continuing"     # 续步阶段
    RETURNING = "returning"       # Return收束
    COMPLETED = "completed"       # 已完成
    FAILED = "failed"             # 执行失败


@dataclass
class SubTaskInfo:
    """子任务信息"""
    goal: str                     # 子任务目标
    frame_id: Optional[str] = None  # 对应的Frame ID
    result: Optional[str] = None   # 子任务结果
    completed: bool = False        # 是否完成
    failed: bool = False          # 是否失败
    error_msg: Optional[str] = None  # 错误信息


@dataclass
class ExecutionFrame:
    """执行帧 - 保存单个节点的执行状态"""

    # 基本信息
    frame_id: str                 # 唯一标识
    node: S                       # 对应的S节点
    parent_frame_id: Optional[str] = None  # 父帧ID

    # 执行状态
    state: FrameState = FrameState.CREATED

    # Think/Eval结果
    think_action: Optional[Union[ReturnAction, PlanAction]] = None
    current_eval_action: Optional[Union[CallAction, ReturnAction]] = None

    # 子任务管理
    subtasks: List[SubTaskInfo] = field(default_factory=list)
    current_subtask_index: int = 0

    # 资源统计（仅本帧产生的消耗）
    frame_think_tokens: int = 0
    frame_eval_tokens: int = 0
    frame_think_calls: int = 0
    frame_eval_calls: int = 0

    # 时间戳
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    # 错误信息
    error_msg: Optional[str] = None
    constraint_triggered: Optional[str] = None

    @property
    def is_completed(self) -> bool:
        """是否已完成"""
        return self.state in (FrameState.COMPLETED, FrameState.FAILED)

    @property
    def is_failed(self) -> bool:
        """是否失败"""
        return self.state == FrameState.FAILED

    @property
    def has_pending_subtasks(self) -> bool:
        """是否有待处理的子任务"""
        return self.current_subtask_index < len(self.subtasks)

    @property
    def current_subtask(self) -> Optional[SubTaskInfo]:
        """当前处理的子任务"""
        if self.has_pending_subtasks:
            return self.subtasks[self.current_subtask_index]
        return None

    def add_subtask(self, goal: str) -> SubTaskInfo:
        """添加子任务"""
        subtask = SubTaskInfo(goal=goal)
        self.subtasks.append(subtask)
        return subtask

    def complete_current_subtask(self, result: str, frame_id: Optional[str] = None):
        """完成当前子任务"""
        if self.has_pending_subtasks:
            subtask = self.subtasks[self.current_subtask_index]
            subtask.result = result
            subtask.completed = True
            subtask.frame_id = frame_id
            # 强不变式：先入档到node.done
            self.node.done.append(result)
            self.current_subtask_index += 1

    def fail_current_subtask(self, error_msg: str):
        """标记当前子任务失败"""
        if self.has_pending_subtasks:
            subtask = self.subtasks[self.current_subtask_index]
            subtask.failed = True
            subtask.error_msg = error_msg
            # 部分进展保全
            partial_result = f"子任务'{subtask.goal[:30]}...'部分完成（{error_msg}）"
            self.node.done.append(partial_result)
            self.current_subtask_index += 1

    def add_frame_tokens(self, think_tokens: int = 0, eval_tokens: int = 0):
        """记录本帧的token消耗"""
        if think_tokens > 0:
            self.frame_think_tokens += think_tokens
            self.frame_think_calls += 1
        if eval_tokens > 0:
            self.frame_eval_tokens += eval_tokens
            self.frame_eval_calls += 1

    @property
    def frame_total_tokens(self) -> int:
        """本帧总token消耗"""
        return self.frame_think_tokens + self.frame_eval_tokens

    def set_completed(self, result: Optional[str] = None):
        """标记帧完成"""
        self.state = FrameState.COMPLETED
        self.completed_at = time.time()
        if result and hasattr(self.current_eval_action, 'description'):
            # 可能需要更新最终结果
            pass

    def set_failed(self, error_msg: str, constraint_triggered: Optional[str] = None):
        """标记帧失败"""
        self.state = FrameState.FAILED
        self.error_msg = error_msg
        self.constraint_triggered = constraint_triggered
        self.completed_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'frame_id': self.frame_id,
            'parent_frame_id': self.parent_frame_id,
            'state': self.state.value,
            'node': self.node.to_dict(),
            'subtasks': [
                {
                    'goal': st.goal,
                    'frame_id': st.frame_id,
                    'result': st.result,
                    'completed': st.completed,
                    'failed': st.failed,
                    'error_msg': st.error_msg
                } for st in self.subtasks
            ],
            'current_subtask_index': self.current_subtask_index,
            'frame_think_tokens': self.frame_think_tokens,
            'frame_eval_tokens': self.frame_eval_tokens,
            'frame_think_calls': self.frame_think_calls,
            'frame_eval_calls': self.frame_eval_calls,
            'created_at': self.created_at,
            'completed_at': self.completed_at,
            'error_msg': self.error_msg,
            'constraint_triggered': self.constraint_triggered
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionFrame':
        """从字典反序列化"""
        # 重建node
        node = S.from_dict(data['node'])

        # 重建subtasks
        subtasks = []
        for st_data in data['subtasks']:
            subtask = SubTaskInfo(
                goal=st_data['goal'],
                frame_id=st_data['frame_id'],
                result=st_data['result'],
                completed=st_data['completed'],
                failed=st_data['failed'],
                error_msg=st_data['error_msg']
            )
            subtasks.append(subtask)

        frame = cls(
            frame_id=data['frame_id'],
            node=node,
            parent_frame_id=data['parent_frame_id'],
            state=FrameState(data['state']),
            subtasks=subtasks,
            current_subtask_index=data['current_subtask_index'],
            frame_think_tokens=data['frame_think_tokens'],
            frame_eval_tokens=data['frame_eval_tokens'],
            frame_think_calls=data['frame_think_calls'],
            frame_eval_calls=data['frame_eval_calls'],
            created_at=data['created_at'],
            completed_at=data['completed_at'],
            error_msg=data['error_msg'],
            constraint_triggered=data['constraint_triggered']
        )

        return frame

    @classmethod
    def reconstruct_parent_relationships(cls, frames: List['ExecutionFrame']):
        """重建父子关系 - 在反序列化快照后调用"""
        frame_map = {frame.frame_id: frame for frame in frames}

        for frame in frames:
            if frame.parent_frame_id and frame.parent_frame_id in frame_map:
                parent_frame = frame_map[frame.parent_frame_id]
                frame.node.parent = parent_frame.node