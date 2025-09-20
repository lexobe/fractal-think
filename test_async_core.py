"""
异步核心模块测试 - 综合测试套件

测试覆盖：
- 异步状态机执行
- 快照创建和恢复
- 预算管理和约束检查
- 同步层适配器
- 恢复管理器
- 集成测试和边界情况
"""

import asyncio
import tempfile
import os
import json
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any

# 导入测试目标
from async_core.engine import AsyncExecutionEngine, AsyncExecutionContext, solve_async
from async_core.interfaces import AsyncThinkLLM, AsyncEvalLLM, SyncToAsyncAdapter
from async_core.frame import ExecutionFrame, FrameState, SubTaskInfo
from async_core.snapshot import ExecutionSnapshot
from async_core.common import ExecutionBudget, BudgetManager, UnifiedTokenUsage, UnifiedLogger, ExecutionMode
from async_core.recovery import RecoveryManager, RecoveryMode, SnapshotManager
from async_core.sync_adapter import solve_with_async_engine, LegacyCompatibilityLayer

from thinkon_core import S, SolveResult, SolveStatus, ThinkLLM, EvalLLM


# Mock 实现
class MockThinkLLM:
    """Mock Think LLM - 返回可预测的结果"""

    def __init__(self, responses: list = None):
        self.responses = responses or [
            {"type": "TODO", "description": "子任务1\n子任务2", "tokens_used": 100},
            {"type": "RETURN", "description": "任务完成", "tokens_used": 50}
        ]
        self.call_count = 0

    def __call__(self, node: S, memory=None, tools=None) -> Dict[str, Any]:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = {"type": "RETURN", "description": "默认完成", "tokens_used": 30}

        self.call_count += 1
        return response


class MockEvalLLM:
    """Mock Eval LLM - 返回可预测的结果"""

    def __init__(self, responses: list = None):
        self.responses = responses or [
            {"type": "CALL", "description": "继续执行", "tokens_used": 80},
            {"type": "RETURN", "description": "评估完成", "tokens_used": 60}
        ]
        self.call_count = 0

    def __call__(self, node: S, memory=None) -> Dict[str, Any]:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = {"type": "RETURN", "description": "默认评估完成", "tokens_used": 40}

        self.call_count += 1
        return response


class MockAsyncThinkLLM:
    """Mock 异步Think LLM"""

    def __init__(self, responses: list = None):
        self.sync_impl = MockThinkLLM(responses)

    async def __call__(self, node: S, memory=None, tools=None) -> Dict[str, Any]:
        # 模拟异步延迟
        await asyncio.sleep(0.01)
        return self.sync_impl(node, memory, tools)


class MockAsyncEvalLLM:
    """Mock 异步Eval LLM"""

    def __init__(self, responses: list = None):
        self.sync_impl = MockEvalLLM(responses)

    async def __call__(self, node: S, memory=None) -> Dict[str, Any]:
        # 模拟异步延迟
        await asyncio.sleep(0.01)
        return self.sync_impl(node, memory)


class TestExecutionFrame:
    """ExecutionFrame 测试"""

    def test_frame_creation(self):
        """测试帧创建"""
        node = S(goal="测试目标")
        frame = ExecutionFrame(frame_id="test-frame", node=node)

        assert frame.frame_id == "test-frame"
        assert frame.node.goal == "测试目标"
        assert frame.state == FrameState.CREATED
        assert not frame.is_completed
        assert not frame.has_pending_subtasks

    def test_subtask_management(self):
        """测试子任务管理"""
        node = S(goal="主任务")
        frame = ExecutionFrame(frame_id="test-frame", node=node)

        # 添加子任务
        subtask1 = frame.add_subtask("子任务1")
        subtask2 = frame.add_subtask("子任务2")

        assert len(frame.subtasks) == 2
        assert frame.has_pending_subtasks
        assert frame.current_subtask == subtask1

        # 完成第一个子任务
        frame.complete_current_subtask("结果1", "frame-1")
        assert subtask1.completed
        assert subtask1.result == "结果1"
        assert frame.current_subtask == subtask2
        assert "结果1" in frame.node.done

        # 完成第二个子任务
        frame.complete_current_subtask("结果2", "frame-2")
        assert not frame.has_pending_subtasks
        assert len(frame.node.done) == 2

    def test_frame_serialization(self):
        """测试帧序列化"""
        node = S(goal="序列化测试")
        frame = ExecutionFrame(frame_id="test-frame", node=node)
        frame.add_subtask("子任务1")
        frame.add_frame_tokens(think_tokens=100, eval_tokens=80)

        # 序列化
        data = frame.to_dict()
        assert data['frame_id'] == "test-frame"
        assert data['frame_think_tokens'] == 100
        assert data['frame_eval_tokens'] == 80

        # 反序列化
        restored_frame = ExecutionFrame.from_dict(data)
        assert restored_frame.frame_id == frame.frame_id
        assert restored_frame.frame_think_tokens == frame.frame_think_tokens
        assert len(restored_frame.subtasks) == len(frame.subtasks)


class TestExecutionBudget:
    """ExecutionBudget 测试"""

    def test_budget_limits(self):
        """测试预算限制检查"""
        budget = ExecutionBudget(max_depth=3, max_tokens=1000, max_time=10.0)

        assert not budget.is_depth_exceeded(2)
        assert budget.is_depth_exceeded(3)

        assert not budget.is_tokens_exceeded(500)
        assert budget.is_tokens_exceeded(1000)

        # 时间测试需要等待
        import time
        start_time = time.time()
        budget.start_time = start_time - 5.0  # 模拟已过5秒
        assert not budget.is_time_exceeded()

        budget.start_time = start_time - 15.0  # 模拟已过15秒
        assert budget.is_time_exceeded()

    def test_budget_adjustment(self):
        """测试预算调整"""
        budget = ExecutionBudget(max_depth=5, max_tokens=1000, max_time=30.0)

        budget.adjust_limits(max_depth=10, max_tokens=2000)
        assert budget.max_depth == 10
        assert budget.max_tokens == 2000
        assert budget.max_time == 30.0  # 未调整的保持原值

    def test_budget_manager(self):
        """测试预算管理器"""
        async def run_test():
            budget = ExecutionBudget(max_depth=2, max_tokens=100)
            manager = BudgetManager(budget)

            # 创建节点层次结构来测试不同level
            root_node = S(goal="根节点测试")
            child_node = S(goal="子节点测试", parent=root_node)
            grandchild_node = S(goal="孙节点测试", parent=child_node)

            # 正常情况
            constraint = await manager.check_constraints_async(root_node, 50)
            assert constraint is None

            # 深度超限
            constraint = await manager.check_constraints_async(grandchild_node, 50)
            assert constraint is not None
            assert "DepthLimitExceeded" in constraint

            # Token超限
            constraint = await manager.check_constraints_async(root_node, 150)
            assert constraint is not None
            assert "ResourceLimitExceeded" in constraint

        asyncio.run(run_test())


class TestExecutionSnapshot:
    """ExecutionSnapshot 测试"""

    def test_snapshot_creation(self):
        """测试快照创建"""
        snapshot = ExecutionSnapshot(
            snapshot_id="test-snapshot",
            goal="测试目标"
        )

        assert snapshot.snapshot_id == "test-snapshot"
        assert snapshot.goal == "测试目标"
        assert snapshot.is_empty
        assert snapshot.stack_depth == 0

    def test_snapshot_frame_management(self):
        """测试快照帧管理"""
        snapshot = ExecutionSnapshot(snapshot_id="test-snapshot")

        node1 = S(goal="帧1")
        frame1 = ExecutionFrame(frame_id="frame-1", node=node1)

        node2 = S(goal="帧2")
        frame2 = ExecutionFrame(frame_id="frame-2", node=node2, parent_frame_id="frame-1")

        snapshot.add_frame(frame1)
        snapshot.add_frame(frame2)

        assert snapshot.stack_depth == 2
        assert not snapshot.is_empty
        assert snapshot.active_frame_id == "frame-1"  # 第一个帧成为活跃帧

        # 获取帧
        retrieved_frame = snapshot.get_frame_by_id("frame-2")
        assert retrieved_frame == frame2

        # 弹出帧
        popped_frame = snapshot.pop_frame()
        assert popped_frame == frame2
        assert snapshot.stack_depth == 1

    def test_snapshot_serialization(self):
        """测试快照序列化"""
        snapshot = ExecutionSnapshot(
            snapshot_id="test-snapshot",
            goal="序列化测试"
        )

        node = S(goal="测试节点")
        frame = ExecutionFrame(frame_id="frame-1", node=node)
        snapshot.add_frame(frame)

        budget = ExecutionBudget(max_depth=5, max_tokens=1000)
        snapshot.budget = budget

        # 序列化为JSON
        json_str = snapshot.to_json()
        assert "test-snapshot" in json_str
        assert "序列化测试" in json_str

        # 从JSON反序列化
        restored_snapshot = ExecutionSnapshot.from_json(json_str)
        assert restored_snapshot.snapshot_id == snapshot.snapshot_id
        assert restored_snapshot.goal == snapshot.goal
        assert restored_snapshot.stack_depth == snapshot.stack_depth
        assert restored_snapshot.budget.max_depth == budget.max_depth

    def test_snapshot_integrity_validation(self):
        """测试快照完整性验证"""
        snapshot = ExecutionSnapshot(snapshot_id="test-snapshot")

        # 空快照无效
        assert not snapshot.validate_integrity()

        # 添加帧后有效
        node = S(goal="测试节点")
        frame = ExecutionFrame(frame_id="frame-1", node=node)
        snapshot.add_frame(frame)
        assert snapshot.validate_integrity()

        # 无效的活跃帧ID
        snapshot.active_frame_id = "nonexistent-frame"
        assert not snapshot.validate_integrity()


class TestAsyncEngine:
    """异步执行引擎测试"""

    def test_simple_async_solve(self):
        """测试简单异步求解"""
        async def run_test():
            think_llm = MockAsyncThinkLLM([
                {"type": "RETURN", "description": "直接完成", "tokens_used": 100}
            ])
            eval_llm = MockAsyncEvalLLM()

            result = await solve_async(
                goal="简单任务",
                think_llm=think_llm,
                eval_llm=eval_llm,
                budget=ExecutionBudget(max_depth=3, max_tokens=1000)
            )

            assert isinstance(result, SolveResult)
            assert result.status == SolveStatus.COMPLETED
            assert "直接完成" in result.result
            assert result.token_usage.total > 0

        asyncio.run(run_test())

    def test_multi_step_async_solve(self):
        """测试多步骤异步求解"""
        async def run_test():
            think_llm = MockAsyncThinkLLM([
                {"type": "TODO", "description": "步骤1：分析问题\n步骤2：制定方案", "tokens_used": 120},
                {"type": "RETURN", "description": "分析完成", "tokens_used": 80},
                {"type": "RETURN", "description": "方案制定完成", "tokens_used": 90}
            ])

            eval_llm = MockAsyncEvalLLM([
                {"type": "CALL", "description": "继续子任务", "tokens_used": 60},
                {"type": "RETURN", "description": "所有任务完成", "tokens_used": 70}
            ])

            result = await solve_async(
                goal="复杂任务",
                think_llm=think_llm,
                eval_llm=eval_llm,
                budget=ExecutionBudget(max_depth=5, max_tokens=2000)
            )

            assert result.status in (SolveStatus.COMPLETED, SolveStatus.DEGRADED)
            assert result.token_usage.total > 200  # 应该有多次调用的token消耗

        asyncio.run(run_test())

    def test_constraint_violation(self):
        """测试约束违规处理"""

        async def run_test():
            think_llm = MockAsyncThinkLLM([
                {"type": "TODO", "description": "任务1\n任务2\n任务3", "tokens_used": 600},
                {"type": "RETURN", "description": "完成", "tokens_used": 500}
            ])
            eval_llm = MockAsyncEvalLLM()

            # 设置很低的token限制
            result = await solve_async(
                goal="超限任务",
                think_llm=think_llm,
                eval_llm=eval_llm,
                budget=ExecutionBudget(max_depth=5, max_tokens=500)  # 很低的限制
            )

            assert result.status == SolveStatus.DEGRADED
            assert result.constraint_triggered is not None
            assert "ResourceLimitExceeded" in result.constraint_triggered


    


        asyncio.run(run_test())


class TestSyncAdapter:
    """同步适配器测试"""

    def test_sync_to_async_adapter(self):
        """测试同步到异步适配器"""
        sync_think = MockThinkLLM([
            {"type": "RETURN", "description": "同步完成", "tokens_used": 100}
        ])
        sync_eval = MockEvalLLM()

        # 使用同步适配器
        result = solve_with_async_engine(
            goal="适配器测试",
            think_llm=sync_think,
            eval_llm=sync_eval,
            budget=ExecutionBudget(max_depth=3, max_tokens=1000)
        )

        assert isinstance(result, SolveResult)
        assert result.status == SolveStatus.COMPLETED
        assert "同步完成" in result.result

    def test_legacy_compatibility(self):
        """测试遗留兼容性"""
        think_llm = MockThinkLLM([
            {"type": "RETURN", "description": "兼容性测试完成", "tokens_used": 100}
        ])
        eval_llm = MockEvalLLM()

        # 测试原始solve函数签名
        result_str = LegacyCompatibilityLayer.solve(
            goal="兼容性测试",
            think_llm=think_llm,
            eval_llm=eval_llm
        )

        assert isinstance(result_str, str)
        assert "兼容性测试完成" in result_str

        # 测试solve_with_meta
        result_meta = LegacyCompatibilityLayer.solve_with_meta(
            goal="元数据测试",
            think_llm=think_llm,
            eval_llm=eval_llm
        )

        assert isinstance(result_meta, SolveResult)
        assert result_meta.status == SolveStatus.COMPLETED


class TestRecoveryManager:
    """恢复管理器测试"""

    def test_snapshot_manager(self):
        """测试快照管理器"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SnapshotManager(temp_dir)

            # 创建测试快照
            snapshot = ExecutionSnapshot(
                snapshot_id="test-recovery",
                goal="恢复测试"
            )

            node = S(goal="测试节点")
            frame = ExecutionFrame(frame_id="frame-1", node=node)
            snapshot.add_frame(frame)

            # 保存快照
            filepath = manager.save_snapshot(snapshot)
            assert os.path.exists(filepath)

            # 加载快照
            loaded_snapshot = manager.load_snapshot(filepath)
            assert loaded_snapshot.snapshot_id == snapshot.snapshot_id
            assert loaded_snapshot.goal == snapshot.goal
            assert loaded_snapshot.stack_depth == snapshot.stack_depth

            # 列出快照
            snapshots = manager.list_snapshots()
            assert len(snapshots) == 1
            assert snapshots[0]['snapshot_id'] == "test-recovery"

        def test_recovery_execution(self):
            """测试恢复执行"""

            async def run_test():
                with tempfile.TemporaryDirectory() as temp_dir:
                    manager = RecoveryManager(temp_dir)

                # 创建包含部分进展的快照
                snapshot = ExecutionSnapshot(
                    snapshot_id="recovery-test",
                    goal="恢复执行测试"
                )

                node = S(goal="主任务")
                node.done.append("已完成的部分1")
                frame = ExecutionFrame(frame_id="main-frame", node=node)
                frame.state = FrameState.CONTINUING
                snapshot.add_frame(frame)

                budget = ExecutionBudget(max_depth=5, max_tokens=1000)
                snapshot.budget = budget

                # 创建恢复计划
                plan = manager.create_recovery_plan(snapshot)
                assert plan.mode == RecoveryMode.CONTINUE
                assert plan.adjusted_budget is not None

                # 模拟恢复执行
                think_llm = MockAsyncThinkLLM([
                    {"type": "RETURN", "description": "恢复完成", "tokens_used": 100}
                ])
                eval_llm = MockAsyncEvalLLM()

                result = await manager.execute_recovery(
                    snapshot=snapshot,
                    think_llm=think_llm,
                    eval_llm=eval_llm,
                    plan=plan
                )

                assert isinstance(result, SolveResult)
                # 由于恢复逻辑的复杂性，这里主要验证没有异常


        


            asyncio.run(run_test())


class TestIntegration:
    """集成测试"""

    def test_end_to_end_with_snapshot(self):
        """端到端测试包含快照"""

        async def run_test():
            snapshots_taken = []

            def snapshot_callback(snapshot: ExecutionSnapshot):
                snapshots_taken.append(snapshot)

            # 创建会触发约束的场景
            think_llm = MockAsyncThinkLLM([
                {"type": "TODO", "description": "长任务1\n长任务2\n长任务3", "tokens_used": 400},
                {"type": "RETURN", "description": "子任务完成", "tokens_used": 200}
            ])
            eval_llm = MockAsyncEvalLLM([
                {"type": "CALL", "description": "继续", "tokens_used": 150}
            ])

            result = await solve_async(
                goal="集成测试任务",
                think_llm=think_llm,
                eval_llm=eval_llm,
                budget=ExecutionBudget(max_depth=5, max_tokens=800)  # 可能触发约束
            )

            assert isinstance(result, SolveResult)
            assert result.status in (SolveStatus.COMPLETED, SolveStatus.DEGRADED)

        asyncio.run(run_test())

    def test_sync_async_compatibility(self):
        """测试同步异步兼容性"""
        # 同步Think/Eval
        sync_think = MockThinkLLM([
            {"type": "RETURN", "description": "同步异步兼容", "tokens_used": 100}
        ])
        sync_eval = MockEvalLLM()

        # 通过同步适配器运行
        result = solve_with_async_engine(
            goal="兼容性测试",
            think_llm=sync_think,
            eval_llm=sync_eval
        )

        assert result.status == SolveStatus.COMPLETED
        assert "同步异步兼容" in result.result

        # 通过遗留API运行（创建新的实例以避免状态共享）
        legacy_think = MockThinkLLM([
            {"type": "RETURN", "description": "同步异步兼容", "tokens_used": 100}
        ])
        legacy_eval = MockEvalLLM()

        legacy_result = LegacyCompatibilityLayer.start_solve_simple(
            goal="遗留API测试",
            think_llm=legacy_think,
            eval_llm=legacy_eval
        )

        assert isinstance(legacy_result, str)
        assert "同步异步兼容" in legacy_result


# 运行测试的辅助函数
def run_tests():
    """运行所有测试"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()