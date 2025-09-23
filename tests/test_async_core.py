"""
异步核心功能测试
"""

import asyncio
import json
from typing import Optional

import pytest

# 导入在pytest和直接运行时都能工作
try:
    from src.fractal_think import solve_async, ExecutionBudget, SolveStatus, ExecutionNode, NodeStatus
    from src.fractal_think.examples.mock_operators import AsyncMockThinkLLM, AsyncMockEvalLLM, DetailedAIArtThink, DetailedAIArtEval
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.fractal_think import solve_async, ExecutionBudget, SolveStatus, ExecutionNode, NodeStatus
    from src.fractal_think.examples.mock_operators import AsyncMockThinkLLM, AsyncMockEvalLLM, DetailedAIArtThink, DetailedAIArtEval


class RecordingObserver:
    """记录ExecutionNode生命周期事件的观察者"""

    def __init__(self):
        self.events = []
        self.root: Optional[ExecutionNode] = None

    def on_node_created(self, node: ExecutionNode) -> None:
        if self.root is None and node.parent is None:
            self.root = node
        self.events.append(("created", node.node_id, node.stage, node.status.value))

    def on_node_started(self, node: ExecutionNode) -> None:
        if self.root is None and node.parent is None:
            self.root = node
        self.events.append(("started", node.node_id, node.stage, node.status.value))

    def on_node_completed(self, node: ExecutionNode) -> None:
        self.events.append(("completed", node.node_id, node.stage, node.status.value))

    def on_node_failed(self, node: ExecutionNode) -> None:
        self.events.append(("failed", node.node_id, node.stage, node.status.value))


class FailingThinkLLM:
    """可控的Think算子：第二个子任务首次执行时失败"""

    def __init__(self):
        self.fail_once = True

    async def __call__(self, node, memory=None, tools=None):
        goal = node.goal
        if goal == "root goal":
            return {
                "type": "TODO",
                "description": "[] child-1 goal\n[] child-2 goal",
                "tokens_used": 1,
            }
        if "child-1" in goal:
            return {"type": "RETURN", "description": "child-1 ok", "tokens_used": 1}
        if "child-2" in goal:
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("simulated failure")
            return {"type": "RETURN", "description": "child-2 ok", "tokens_used": 1}
        return {"type": "RETURN", "description": f"done {goal}", "tokens_used": 1}


class SequencedEvalLLM:
    """按顺序调度两个子任务的Eval算子"""

    def __init__(self):
        self.calls = {}

    async def __call__(self, node, memory=None):
        goal = node.goal
        self.calls.setdefault(goal, 0)
        self.calls[goal] += 1

        if goal == "root goal":
            if self.calls[goal] == 1:
                return {"type": "CALL", "description": "child-1 goal", "tokens_used": 1}
            if self.calls[goal] == 2:
                return {"type": "CALL", "description": "child-2 goal", "tokens_used": 1}
            return {"type": "RETURN", "description": "root done", "tokens_used": 1}

        return {"type": "RETURN", "description": f"eval {goal}", "tokens_used": 1}


def test_basic_async_solve():
    """测试基础异步求解功能"""
    think_llm = AsyncMockThinkLLM(simulation_delay=0.01, verbose=False)
    eval_llm = AsyncMockEvalLLM(simulation_delay=0.01, verbose=False)

    result = asyncio.run(
        solve_async(
            goal="简单测试任务",
            think_llm=think_llm,
            eval_llm=eval_llm,
            budget=ExecutionBudget(max_depth=3, max_tokens=1000, max_time=5.0),
        )
    )

    assert result.status == SolveStatus.COMPLETED
    assert result.token_usage.total > 0
    assert think_llm.call_count > 0
    assert eval_llm.call_count >= 0


def test_detailed_ai_art_example():
    """测试详细的AI与艺术示例"""
    think_llm = DetailedAIArtThink(simulation_delay=0.01, verbose=False)
    eval_llm = DetailedAIArtEval(simulation_delay=0.01, verbose=False)

    result = asyncio.run(
        solve_async(
            goal='写一篇"AI与艺术"的短文（800–1200字）',
            think_llm=think_llm,
            eval_llm=eval_llm,
            budget=ExecutionBudget(max_depth=5, max_tokens=2000, max_time=10.0),
        )
    )

    assert result.status == SolveStatus.COMPLETED
    assert "AI与艺术" in result.result
    assert len(result.result) > 500  # 应该是相当长的文章
    assert result.max_depth_reached > 0


def test_budget_constraints():
    """测试预算约束"""
    from src.fractal_think.types import DepthLimitExceeded, ResourceLimitExceeded, TimeLimitExceeded

    think_llm = AsyncMockThinkLLM(simulation_delay=0.01, verbose=False)
    eval_llm = AsyncMockEvalLLM(simulation_delay=0.01, verbose=False)

    # 测试深度限制 - 应该抛出异常
    with pytest.raises(DepthLimitExceeded):
        asyncio.run(
            solve_async(
                goal="复杂多层任务，需要多层处理和深度递归测试",
                think_llm=think_llm,
                eval_llm=eval_llm,
                budget=ExecutionBudget(max_depth=1, max_tokens=1000, max_time=5.0),
            )
        )


def test_token_tracking():
    """测试Token追踪功能"""
    think_llm = AsyncMockThinkLLM(simulation_delay=0.01, verbose=False)
    eval_llm = AsyncMockEvalLLM(simulation_delay=0.01, verbose=False)

    result = asyncio.run(
        solve_async(
            goal="Token追踪测试",
            think_llm=think_llm,
            eval_llm=eval_llm,
            budget=ExecutionBudget(max_depth=3, max_tokens=1000, max_time=5.0),
        )
    )

    assert result.token_usage.total > 0
    assert result.token_usage.think_tokens >= 0
    assert result.token_usage.eval_tokens >= 0
    assert result.token_usage.think_calls > 0


def test_execution_tree_resume_and_observer_events():
    """验证ExecutionNode恢复、事件顺序以及JSON输出"""

    think_llm = FailingThinkLLM()
    eval_llm = SequencedEvalLLM()
    observer = RecordingObserver()

    failed_result = asyncio.run(
        solve_async(
            goal="root goal",
            think_llm=think_llm,
            eval_llm=eval_llm,
            budget=ExecutionBudget(max_depth=5, max_tokens=200, max_time=5.0),
            observers=[observer],
        )
    )

    assert failed_result.status == SolveStatus.FAILED
    root = observer.root
    assert root is not None
    assert len(root.children) == 2
    first_child, failed_child = root.children
    assert first_child.status == NodeStatus.COMPLETED
    assert failed_child.status == NodeStatus.FAILED
    assert failed_child.metadata.get("resume_stage") == "think"

    event_types = [evt[0] for evt in observer.events]
    assert event_types[0] == "created"
    assert event_types[1] == "started"
    assert ("failed", failed_child.node_id, failed_child.stage, failed_child.status.value) in observer.events

    tree_dict = root.to_dict()
    assert tree_dict["children"][1]["status"] == NodeStatus.FAILED.value
    json.dumps(tree_dict)  # 确认可序列化

    observer_resumed = RecordingObserver()
    resumed_result = asyncio.run(
        solve_async(
            goal="root goal",
            think_llm=think_llm,
            eval_llm=eval_llm,
            budget=ExecutionBudget(max_depth=5, max_tokens=200, max_time=5.0),
            execution_tree=tree_dict,
            observers=[observer_resumed],
        )
    )

    assert resumed_result.status == SolveStatus.COMPLETED
    resumed_root = observer_resumed.root
    assert resumed_root is not None
    assert resumed_root.status == NodeStatus.COMPLETED
    assert resumed_root.result_summary == "root done"
    assert len(resumed_root.children) == 2
    assert resumed_root.children[1].status == NodeStatus.COMPLETED
    assert "resume_stage" not in resumed_root.children[1].metadata

    resumed_events = [evt[0] for evt in observer_resumed.events]
    assert resumed_events.count("started") >= 1
    assert resumed_events[-1] == "completed"

    restored_tree = ExecutionNode.from_dict(resumed_root.to_dict())
    assert restored_tree.children[0].result_summary == "child-1 ok"
    assert restored_tree.children[1].result_summary == "child-2 ok"

    already_completed = asyncio.run(
        solve_async(
            goal="root goal",
            think_llm=think_llm,
            eval_llm=eval_llm,
            budget=ExecutionBudget(max_depth=5, max_tokens=200, max_time=5.0),
            execution_tree=resumed_root,
        )
    )

    assert already_completed.status == SolveStatus.COMPLETED
    assert already_completed.result == "root done"