"""
恢复和续跑管理器 - 提供智能状态恢复和执行续跑

功能包括：
- 快照持久化（文件存储/加载）
- 恢复计划分析和建议
- 智能预算调整和续跑策略
- 多种恢复模式（完全恢复、部分重试、重新开始）
- 恢复完整性验证
"""

import os
import json
import uuid
import time
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio

from .snapshot import ExecutionSnapshot
from .engine import AsyncExecutionEngine, AsyncExecutionContext
from .common import ExecutionBudget, BudgetManager, UnifiedLogger, ExecutionMode, UnifiedTokenUsage
from .interfaces import AsyncThinkLLM, AsyncEvalLLM, create_async_think, create_async_eval
try:
    from ..thinkon_core import SolveResult, SolveStatus, ThinkLLM, EvalLLM
except ImportError:
    # 当作为顶层模块运行时的回退导入
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from thinkon_core import SolveResult, SolveStatus, ThinkLLM, EvalLLM


class RecoveryMode(Enum):
    """恢复模式"""
    CONTINUE = "continue"           # 从中断点继续
    PARTIAL_RETRY = "partial_retry" # 部分重试（重新执行失败部分）
    RESTART = "restart"             # 完全重新开始
    HYBRID = "hybrid"               # 混合模式（智能选择）


class RecoveryStrategy(Enum):
    """恢复策略"""
    CONSERVATIVE = "conservative"   # 保守策略（最小化变更）
    AGGRESSIVE = "aggressive"       # 激进策略（最大化恢复）
    BALANCED = "balanced"           # 平衡策略（智能权衡）


@dataclass
class RecoveryPlan:
    """恢复计划"""
    mode: RecoveryMode
    strategy: RecoveryStrategy

    # 预算调整
    adjusted_budget: Optional[ExecutionBudget] = None
    budget_increase_ratio: float = 1.5  # 预算增加比例

    # 执行建议
    target_frame_id: Optional[str] = None
    restart_from_frame: Optional[str] = None
    skip_frames: List[str] = None

    # 风险评估
    estimated_success_rate: float = 0.0
    estimated_additional_cost: int = 0
    risk_factors: List[str] = None

    # 建议说明
    recommendations: List[str] = None

    def __post_init__(self):
        if self.skip_frames is None:
            self.skip_frames = []
        if self.risk_factors is None:
            self.risk_factors = []
        if self.recommendations is None:
            self.recommendations = []


class SnapshotManager:
    """快照管理器 - 处理快照的持久化和元数据管理"""

    def __init__(self, storage_dir: str = "./snapshots"):
        self.storage_dir = storage_dir
        self._ensure_storage_dir()

    def _ensure_storage_dir(self):
        """确保存储目录存在"""
        os.makedirs(self.storage_dir, exist_ok=True)

    def save_snapshot(self, snapshot: ExecutionSnapshot) -> str:
        """保存快照到文件"""
        filename = f"snapshot_{snapshot.snapshot_id}_{int(time.time())}.json"
        filepath = os.path.join(self.storage_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(snapshot.to_json())
            return filepath
        except Exception as e:
            raise RuntimeError(f"快照保存失败: {e}")

    def load_snapshot(self, filepath: str) -> ExecutionSnapshot:
        """从文件加载快照"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                json_data = f.read()
            return ExecutionSnapshot.from_json(json_data)
        except Exception as e:
            raise RuntimeError(f"快照加载失败: {e}")

    def list_snapshots(self, goal_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出所有快照（带过滤）"""
        snapshots = []

        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json') and filename.startswith('snapshot_'):
                filepath = os.path.join(self.storage_dir, filename)
                try:
                    snapshot = self.load_snapshot(filepath)
                    if goal_filter is None or goal_filter in snapshot.goal:
                        snapshots.append({
                            'filepath': filepath,
                            'snapshot_id': snapshot.snapshot_id,
                            'goal': snapshot.goal,
                            'created_at': snapshot.created_at,
                            'trigger_reason': snapshot.trigger_reason,
                            'stack_depth': snapshot.stack_depth,
                            'total_tokens': snapshot.calculate_total_tokens()
                        })
                except Exception:
                    continue  # 跳过损坏的快照

        # 按创建时间排序
        snapshots.sort(key=lambda x: x['created_at'], reverse=True)
        return snapshots

    def cleanup_old_snapshots(self, max_age_days: int = 7, max_count: int = 100):
        """清理旧快照"""
        snapshots = self.list_snapshots()
        current_time = time.time()

        # 按时间和数量清理
        for i, snapshot_info in enumerate(snapshots):
            age_days = (current_time - snapshot_info['created_at']) / (24 * 3600)

            if age_days > max_age_days or i >= max_count:
                try:
                    os.remove(snapshot_info['filepath'])
                except Exception:
                    pass  # 忽略清理失败


class RecoveryAnalyzer:
    """恢复分析器 - 分析快照并生成恢复计划"""

    def __init__(self):
        pass

    def analyze_snapshot(self, snapshot: ExecutionSnapshot) -> RecoveryPlan:
        """分析快照并生成恢复计划"""
        # 基础分析
        total_frames = len(snapshot.frame_stack)
        completed_frames = sum(1 for f in snapshot.frame_stack if f.is_completed)
        failed_frames = sum(1 for f in snapshot.frame_stack if f.is_failed)
        progress_rate = completed_frames / total_frames if total_frames > 0 else 0

        # 约束分析
        constraint_triggered = snapshot.constraint_triggered
        trigger_reason = snapshot.trigger_reason

        # 选择恢复模式
        mode = self._select_recovery_mode(
            progress_rate, constraint_triggered, trigger_reason, failed_frames
        )

        # 选择恢复策略
        strategy = self._select_recovery_strategy(snapshot, mode)

        # 预算调整
        adjusted_budget = self._adjust_budget(snapshot, mode, constraint_triggered)

        # 生成详细计划
        plan = RecoveryPlan(
            mode=mode,
            strategy=strategy,
            adjusted_budget=adjusted_budget,
            target_frame_id=snapshot.active_frame_id
        )

        # 填充详细分析
        self._fill_recovery_details(plan, snapshot)

        return plan

    def _select_recovery_mode(self, progress_rate: float, constraint_triggered: Optional[str],
                             trigger_reason: Optional[str], failed_frames: int) -> RecoveryMode:
        """选择恢复模式"""
        if trigger_reason == "constraint_violation":
            if progress_rate > 0.7:
                return RecoveryMode.CONTINUE  # 已完成大部分，继续执行
            elif progress_rate > 0.3:
                return RecoveryMode.PARTIAL_RETRY  # 部分重试
            else:
                return RecoveryMode.RESTART  # 重新开始

        elif failed_frames > 0:
            if failed_frames == 1 and progress_rate > 0.5:
                return RecoveryMode.PARTIAL_RETRY
            else:
                return RecoveryMode.RESTART

        else:
            return RecoveryMode.CONTINUE  # 默认继续

    def _select_recovery_strategy(self, snapshot: ExecutionSnapshot,
                                 mode: RecoveryMode) -> RecoveryStrategy:
        """选择恢复策略"""
        if mode == RecoveryMode.RESTART:
            return RecoveryStrategy.AGGRESSIVE
        elif mode == RecoveryMode.CONTINUE:
            return RecoveryStrategy.CONSERVATIVE
        else:
            return RecoveryStrategy.BALANCED

    def _adjust_budget(self, snapshot: ExecutionSnapshot, mode: RecoveryMode,
                      constraint_triggered: Optional[str]) -> Optional[ExecutionBudget]:
        """调整预算"""
        if not snapshot.budget:
            return ExecutionBudget()

        original_budget = snapshot.budget
        new_budget = ExecutionBudget(
            max_depth=original_budget.max_depth,
            max_tokens=original_budget.max_tokens,
            max_time=original_budget.max_time
        )

        # 根据约束类型和模式调整
        if constraint_triggered:
            if "DepthLimitExceeded" in constraint_triggered:
                new_budget.max_depth = int(original_budget.max_depth * 1.5)
            elif "ResourceLimitExceeded" in constraint_triggered:
                new_budget.max_tokens = int(original_budget.max_tokens * 2.0)
            elif "TimeLimitExceeded" in constraint_triggered:
                new_budget.max_time = original_budget.max_time * 1.5

        # 根据模式微调
        if mode == RecoveryMode.RESTART:
            # 重新开始时保持原预算
            pass
        elif mode == RecoveryMode.CONTINUE:
            # 继续执行时适度增加
            new_budget.max_tokens = int(new_budget.max_tokens * 1.2)
            new_budget.max_time *= 1.1
        elif mode == RecoveryMode.PARTIAL_RETRY:
            # 部分重试时大幅增加
            new_budget.max_tokens = int(new_budget.max_tokens * 1.8)
            new_budget.max_time *= 1.3

        return new_budget

    def _fill_recovery_details(self, plan: RecoveryPlan, snapshot: ExecutionSnapshot):
        """填充恢复计划详细信息"""
        # 估算成功率
        if plan.mode == RecoveryMode.CONTINUE:
            plan.estimated_success_rate = 0.8
        elif plan.mode == RecoveryMode.PARTIAL_RETRY:
            plan.estimated_success_rate = 0.65
        elif plan.mode == RecoveryMode.RESTART:
            plan.estimated_success_rate = 0.9

        # 估算额外成本
        current_tokens = snapshot.calculate_total_tokens()
        if plan.mode == RecoveryMode.CONTINUE:
            plan.estimated_additional_cost = int(current_tokens * 0.3)
        elif plan.mode == RecoveryMode.PARTIAL_RETRY:
            plan.estimated_additional_cost = int(current_tokens * 0.6)
        elif plan.mode == RecoveryMode.RESTART:
            plan.estimated_additional_cost = current_tokens  # 重新开始

        # 生成建议
        self._generate_recommendations(plan, snapshot)

    def _generate_recommendations(self, plan: RecoveryPlan, snapshot: ExecutionSnapshot):
        """生成恢复建议"""
        if plan.mode == RecoveryMode.CONTINUE:
            plan.recommendations.append("建议从中断点继续执行")
            if plan.adjusted_budget:
                plan.recommendations.append(f"已调整预算限制以避免重复约束触发")

        elif plan.mode == RecoveryMode.PARTIAL_RETRY:
            plan.recommendations.append("建议重新执行失败的部分")
            plan.recommendations.append("保留已完成的进展")

        elif plan.mode == RecoveryMode.RESTART:
            plan.recommendations.append("建议完全重新开始")
            plan.recommendations.append("当前进展较少，重新开始成本较低")

        # 添加约束相关建议
        if snapshot.constraint_triggered:
            if "Depth" in snapshot.constraint_triggered:
                plan.recommendations.append("检测到深度限制，建议适当增加max_depth")
            elif "Resource" in snapshot.constraint_triggered:
                plan.recommendations.append("检测到token限制，建议增加max_tokens或简化任务")
            elif "Time" in snapshot.constraint_triggered:
                plan.recommendations.append("检测到时间限制，建议增加max_time或使用异步并发")


class RecoveryManager:
    """恢复管理器 - 统一的快照和恢复管理接口"""

    def __init__(self, storage_dir: str = "./snapshots"):
        self.snapshot_manager = SnapshotManager(storage_dir)
        self.analyzer = RecoveryAnalyzer()

    def save_execution_snapshot(self, snapshot: ExecutionSnapshot) -> str:
        """保存执行快照"""
        return self.snapshot_manager.save_snapshot(snapshot)

    def load_execution_snapshot(self, filepath: str) -> ExecutionSnapshot:
        """加载执行快照"""
        return self.snapshot_manager.load_snapshot(filepath)

    def create_recovery_plan(self, snapshot: ExecutionSnapshot) -> RecoveryPlan:
        """创建恢复计划"""
        if not snapshot.validate_integrity():
            raise ValueError("快照完整性验证失败")

        return self.analyzer.analyze_snapshot(snapshot)

    async def execute_recovery(self,
                              snapshot: ExecutionSnapshot,
                              think_llm: Union[ThinkLLM, AsyncThinkLLM],
                              eval_llm: Union[EvalLLM, AsyncEvalLLM],
                              plan: Optional[RecoveryPlan] = None,
                              logger: Optional[UnifiedLogger] = None) -> SolveResult:
        """执行恢复"""

        # 生成恢复计划
        if plan is None:
            plan = self.create_recovery_plan(snapshot)

        if logger:
            logger.info(f"开始执行恢复: {plan.mode.value} 模式")

        # 根据恢复模式执行
        if plan.mode == RecoveryMode.CONTINUE:
            return await self._execute_continue_recovery(snapshot, think_llm, eval_llm, plan, logger)
        elif plan.mode == RecoveryMode.PARTIAL_RETRY:
            return await self._execute_partial_retry(snapshot, think_llm, eval_llm, plan, logger)
        elif plan.mode == RecoveryMode.RESTART:
            return await self._execute_restart_recovery(snapshot, think_llm, eval_llm, plan, logger)
        else:
            raise ValueError(f"不支持的恢复模式: {plan.mode}")

    async def _execute_continue_recovery(self, snapshot: ExecutionSnapshot,
                                        think_llm: Union[ThinkLLM, AsyncThinkLLM],
                                        eval_llm: Union[EvalLLM, AsyncEvalLLM],
                                        plan: RecoveryPlan,
                                        logger: Optional[UnifiedLogger]) -> SolveResult:
        """执行继续恢复"""
        # 创建异步算子
        async_think = create_async_think(think_llm)
        async_eval = create_async_eval(eval_llm)

        # 创建执行上下文
        if logger is None:
            logger = UnifiedLogger(mode=ExecutionMode.ASYNC)

        token_usage = snapshot.global_token_usage or UnifiedTokenUsage()
        token_usage.set_async_mode()

        context = AsyncExecutionContext(
            think_llm=async_think,
            eval_llm=async_eval,
            budget_manager=BudgetManager(plan.adjusted_budget or ExecutionBudget()),
            token_usage=token_usage,
            logger=logger,
            frame_stack=[]
        )

        # 创建引擎并恢复执行
        engine = AsyncExecutionEngine(context)
        return await engine.resume_from_snapshot(snapshot)

    async def _execute_partial_retry(self, snapshot: ExecutionSnapshot,
                                    think_llm: Union[ThinkLLM, AsyncThinkLLM],
                                    eval_llm: Union[EvalLLM, AsyncEvalLLM],
                                    plan: RecoveryPlan,
                                    logger: Optional[UnifiedLogger]) -> SolveResult:
        """执行部分重试恢复"""
        # TODO: 实现更复杂的部分重试逻辑
        # 目前简化为继续恢复
        return await self._execute_continue_recovery(snapshot, think_llm, eval_llm, plan, logger)

    async def _execute_restart_recovery(self, snapshot: ExecutionSnapshot,
                                       think_llm: Union[ThinkLLM, AsyncThinkLLM],
                                       eval_llm: Union[EvalLLM, AsyncEvalLLM],
                                       plan: RecoveryPlan,
                                       logger: Optional[UnifiedLogger]) -> SolveResult:
        """执行重新开始恢复"""
        from .engine import solve_async

        # 重新开始，使用调整后的预算
        if logger is None:
            logger = UnifiedLogger(mode=ExecutionMode.ASYNC)

        return await solve_async(
            goal=snapshot.goal,
            think_llm=think_llm,
            eval_llm=eval_llm,
            budget=plan.adjusted_budget,
            logger=logger
        )

    def list_recoverable_snapshots(self, goal_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出可恢复的快照"""
        return self.snapshot_manager.list_snapshots(goal_filter)

    def cleanup_snapshots(self, max_age_days: int = 7, max_count: int = 100):
        """清理旧快照"""
        self.snapshot_manager.cleanup_old_snapshots(max_age_days, max_count)


# 便捷函数
async def recover_from_file(filepath: str,
                           think_llm: Union[ThinkLLM, AsyncThinkLLM],
                           eval_llm: Union[EvalLLM, AsyncEvalLLM],
                           recovery_mode: Optional[RecoveryMode] = None,
                           logger: Optional[UnifiedLogger] = None) -> SolveResult:
    """从快照文件恢复执行的便捷函数"""
    manager = RecoveryManager()

    # 加载快照
    snapshot = manager.load_execution_snapshot(filepath)

    # 创建恢复计划
    plan = manager.create_recovery_plan(snapshot)
    if recovery_mode:
        plan.mode = recovery_mode

    # 执行恢复
    return await manager.execute_recovery(snapshot, think_llm, eval_llm, plan, logger)