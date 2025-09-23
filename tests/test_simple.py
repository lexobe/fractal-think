"""
简单测试 - 验证基本功能
"""

import asyncio
import pytest

# 导入在pytest和直接运行时都能工作
try:
    from src.fractal_think import (
        solve_async,
        ExecutionBudget,
        SolveStatus,
        S,
        SolveResult,
        TokenUsage,
        ExecutionNode,
        NodeStatus,
    )
    from src.fractal_think.examples.mock_operators import AsyncMockThinkLLM, AsyncMockEvalLLM
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.fractal_think import (
        solve_async,
        ExecutionBudget,
        SolveStatus,
        S,
        SolveResult,
        TokenUsage,
        ExecutionNode,
        NodeStatus,
    )
    from src.fractal_think.examples.mock_operators import AsyncMockThinkLLM, AsyncMockEvalLLM


def test_basic_imports():
    """测试基本导入功能"""
    # 创建基本对象
    node = S(goal="测试目标")
    assert node.goal == "测试目标"
    assert node.level == 0
    assert len(node.done) == 0


def test_execution_node_serialization():
    """验证ExecutionNode的序列化/反序列化与状态迁移"""
    root = ExecutionNode(
        node_id="root",
        goal="测试根目标",
        depth=0,
        stage="think",
        node_type="root",
    )
    root.mark_running(stage="think", timestamp=1.0)
    root.todo = "完成测试"
    child = ExecutionNode(
        node_id="child",
        goal="子任务",
        depth=1,
        stage="think",
        node_type="task",
    )
    root.add_child(child)
    child.mark_running(stage="think", timestamp=2.0)
    child.mark_failed("子任务失败", stage="failed", timestamp=3.0)
    root.done.append("子任务失败")
    root.mark_completed("全部完成", stage="completed", timestamp=4.0)

    payload = root.to_dict()
    restored = ExecutionNode.from_dict(payload)

    assert restored.status == NodeStatus.COMPLETED
    assert restored.result_summary == "全部完成"
    assert restored.children[0].status == NodeStatus.FAILED
    assert restored.children[0].error_message == "子任务失败"


def test_basic_async_solve():
    """运行基本异步测试"""
    think_llm = AsyncMockThinkLLM(simulation_delay=0.01, verbose=False)
    eval_llm = AsyncMockEvalLLM(simulation_delay=0.01, verbose=False)

    result = asyncio.run(
        solve_async(
            goal="简单测试任务",
            think_llm=think_llm,
            eval_llm=eval_llm,
            budget=ExecutionBudget(max_depth=2, max_tokens=500, max_time=5.0),
        )
    )

    assert result.status == SolveStatus.COMPLETED
    assert result.token_usage.total > 0
    assert think_llm.call_count > 0


def test_specification_example():
    """测试规范版算子"""
    try:
        from src.fractal_think.examples.specification_operators import (
            SpecificationAIArtThink, SpecificationAIArtEval
        )
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.fractal_think.examples.specification_operators import (
            SpecificationAIArtThink, SpecificationAIArtEval
        )

    think_llm = SpecificationAIArtThink(simulation_delay=0.01, verbose=False)
    eval_llm = SpecificationAIArtEval(simulation_delay=0.01, verbose=False)

    result = asyncio.run(
        solve_async(
            goal='写一篇"AI与艺术"的短文',
            think_llm=think_llm,
            eval_llm=eval_llm,
            budget=ExecutionBudget(max_depth=3, max_tokens=1000, max_time=10.0),
        )
    )

    assert result.status == SolveStatus.COMPLETED
    assert "AI与艺术" in result.result
    assert len(result.result) > 100  # Updated to match exact specification output
    assert "生成艺术段落已经写完，含DALL·E例，并符合要求。" in result.result  # Exact specification text
    assert "辅助创作段落已经写完，包含了一个能打动人的事例，符合要求。" in result.result  # Exact original text
    assert "艺术评论段落已经写完，提供了深入的分析和见解，符合要求。" in result.result  # Exact original text


if __name__ == "__main__":
    # 直接运行模式（向后兼容）
    test_basic_imports()
    print("✅ 导入测试通过")

    async def run_tests():
        await test_basic_async_solve()
        print("✅ 基本异步测试通过")

        await test_specification_example()
        print("✅ 规范示例测试通过")

    asyncio.run(run_tests())
    print("🎉 所有测试通过！")