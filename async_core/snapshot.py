"""
执行快照 - 支持状态恢复和续跑
"""

import time
import json
import copy
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .frame import ExecutionFrame
from .common import ExecutionBudget, UnifiedTokenUsage


@dataclass
class ExecutionSnapshot:
    """执行快照 - 包含完整的可恢复状态"""

    # 基本信息
    snapshot_id: str              # 快照ID
    created_at: float = field(default_factory=time.time)
    goal: str = ""                # 根任务目标

    # 执行栈状态
    frame_stack: List[ExecutionFrame] = field(default_factory=list)
    active_frame_id: Optional[str] = None  # 当前活跃的帧ID

    # 预算和统计
    budget: Optional[ExecutionBudget] = None
    global_token_usage: Optional[UnifiedTokenUsage] = None

    # 触发信息
    trigger_reason: Optional[str] = None    # 快照触发原因
    trigger_frame_id: Optional[str] = None  # 触发快照的帧ID
    constraint_triggered: Optional[str] = None

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """是否为空快照"""
        return len(self.frame_stack) == 0

    @property
    def stack_depth(self) -> int:
        """栈深度"""
        return len(self.frame_stack)

    @property
    def current_frame(self) -> Optional[ExecutionFrame]:
        """当前活跃帧"""
        if self.active_frame_id:
            for frame in self.frame_stack:
                if frame.frame_id == self.active_frame_id:
                    return frame
        return None

    def get_frame_by_id(self, frame_id: str) -> Optional[ExecutionFrame]:
        """根据ID获取帧"""
        for frame in self.frame_stack:
            if frame.frame_id == frame_id:
                return frame
        return None

    def add_frame(self, frame: ExecutionFrame):
        """添加帧到栈中"""
        self.frame_stack.append(frame)
        if not self.active_frame_id:
            self.active_frame_id = frame.frame_id

    def pop_frame(self) -> Optional[ExecutionFrame]:
        """弹出栈顶帧"""
        if self.frame_stack:
            frame = self.frame_stack.pop()
            # 更新活跃帧ID
            if self.active_frame_id == frame.frame_id:
                self.active_frame_id = self.frame_stack[-1].frame_id if self.frame_stack else None
            return frame
        return None

    def set_trigger_info(self, reason: str, frame_id: Optional[str] = None,
                        constraint: Optional[str] = None):
        """设置触发信息"""
        self.trigger_reason = reason
        self.trigger_frame_id = frame_id
        self.constraint_triggered = constraint

    def calculate_total_tokens(self) -> int:
        """计算所有帧的总token消耗"""
        total = 0
        for frame in self.frame_stack:
            total += frame.frame_total_tokens
        return total

    def get_completion_summary(self) -> Dict[str, int]:
        """获取完成情况摘要"""
        completed_frames = sum(1 for f in self.frame_stack if f.is_completed)
        failed_frames = sum(1 for f in self.frame_stack if f.is_failed)
        total_subtasks = sum(len(f.subtasks) for f in self.frame_stack)
        completed_subtasks = sum(
            len([st for st in f.subtasks if st.completed])
            for f in self.frame_stack
        )

        return {
            'total_frames': len(self.frame_stack),
            'completed_frames': completed_frames,
            'failed_frames': failed_frames,
            'total_subtasks': total_subtasks,
            'completed_subtasks': completed_subtasks
        }

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'snapshot_id': self.snapshot_id,
            'created_at': self.created_at,
            'goal': self.goal,
            'frame_stack': [frame.to_dict() for frame in self.frame_stack],
            'active_frame_id': self.active_frame_id,
            'budget': self.budget.to_dict() if self.budget else None,
            'global_token_usage': {
                'total': self.global_token_usage.total,
                'think_calls': self.global_token_usage.think_calls,
                'eval_calls': self.global_token_usage.eval_calls,
                'think_tokens': self.global_token_usage.think_tokens,
                'eval_tokens': self.global_token_usage.eval_tokens
            } if self.global_token_usage else None,
            'trigger_reason': self.trigger_reason,
            'trigger_frame_id': self.trigger_frame_id,
            'constraint_triggered': self.constraint_triggered,
            'metadata': self.metadata
        }

    def to_json(self) -> str:
        """序列化为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionSnapshot':
        """从字典反序列化"""
        snapshot = cls(
            snapshot_id=data['snapshot_id'],
            created_at=data['created_at'],
            goal=data['goal'],
            active_frame_id=data['active_frame_id'],
            trigger_reason=data['trigger_reason'],
            trigger_frame_id=data['trigger_frame_id'],
            constraint_triggered=data['constraint_triggered'],
            metadata=data.get('metadata', {})
        )

        # 恢复帧栈
        for frame_data in data['frame_stack']:
            frame = ExecutionFrame.from_dict(frame_data)
            snapshot.frame_stack.append(frame)

        # 重建父子关系
        ExecutionFrame.reconstruct_parent_relationships(snapshot.frame_stack)

        # 恢复预算
        if data['budget']:
            snapshot.budget = ExecutionBudget.from_dict(data['budget'])

        # 恢复token使用情况
        if data['global_token_usage']:
            token_data = data['global_token_usage']
            token_usage = UnifiedTokenUsage()
            token_usage.total = token_data['total']
            token_usage.think_calls = token_data['think_calls']
            token_usage.eval_calls = token_data['eval_calls']
            token_usage.think_tokens = token_data['think_tokens']
            token_usage.eval_tokens = token_data['eval_tokens']
            snapshot.global_token_usage = token_usage

        return snapshot

    @classmethod
    def from_json(cls, json_str: str) -> 'ExecutionSnapshot':
        """从JSON字符串反序列化"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def validate_integrity(self) -> bool:
        """验证快照完整性"""
        try:
            # 检查帧栈
            if not self.frame_stack:
                return False

            # 检查活跃帧ID
            if self.active_frame_id and not self.get_frame_by_id(self.active_frame_id):
                return False

            # 检查父子关系
            for frame in self.frame_stack:
                if frame.parent_frame_id:
                    parent = self.get_frame_by_id(frame.parent_frame_id)
                    if not parent:
                        return False

            # 检查token一致性
            if self.global_token_usage:
                calculated_total = self.calculate_total_tokens()
                # 允许一定的误差，因为全局统计可能包含其他开销
                if abs(self.global_token_usage.total - calculated_total) > calculated_total * 0.1:
                    return False

            return True

        except Exception:
            return False

    def create_recovery_plan(self) -> Dict[str, Any]:
        """创建恢复计划"""
        plan = {
            'recovery_type': 'continue',  # 或 'restart', 'partial_retry'
            'target_frame_id': self.active_frame_id,
            'estimated_remaining_work': 0,
            'recommendations': []
        }

        # 分析剩余工作量
        current_frame = self.current_frame
        if current_frame:
            remaining_subtasks = len(current_frame.subtasks) - current_frame.current_subtask_index
            plan['estimated_remaining_work'] = remaining_subtasks

            # 提供建议
            if current_frame.constraint_triggered:
                plan['recommendations'].append(f"约束触发：{current_frame.constraint_triggered}")
                plan['recommendations'].append("建议调高对应的预算限制")

            if remaining_subtasks > 0:
                plan['recommendations'].append(f"还有{remaining_subtasks}个子任务待完成")

        return plan