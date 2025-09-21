"""
异步核心功能测试
"""

import asyncio
import pytest

# 导入在pytest和直接运行时都能工作
try:
    from src.fractal_think import solve_async, ExecutionBudget, SolveStatus
    from src.fractal_think.examples.mock_operators import AsyncMockThinkLLM, AsyncMockEvalLLM, DetailedAIArtThink, DetailedAIArtEval
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.fractal_think import solve_async, ExecutionBudget, SolveStatus
    from src.fractal_think.examples.mock_operators import AsyncMockThinkLLM, AsyncMockEvalLLM, DetailedAIArtThink, DetailedAIArtEval


@pytest.mark.asyncio
async def test_basic_async_solve():
    """测试基础异步求解功能"""
    think_llm = AsyncMockThinkLLM(simulation_delay=0.01, verbose=False)
    eval_llm = AsyncMockEvalLLM(simulation_delay=0.01, verbose=False)

    result = await solve_async(
        goal="简单测试任务",
        think_llm=think_llm,
        eval_llm=eval_llm,
        budget=ExecutionBudget(max_depth=3, max_tokens=1000, max_time=5.0)
    )

    assert result.status == SolveStatus.COMPLETED
    assert result.token_usage.total > 0
    assert think_llm.call_count > 0
    assert eval_llm.call_count >= 0


@pytest.mark.asyncio
async def test_detailed_ai_art_example():
    """测试详细的AI与艺术示例"""
    think_llm = DetailedAIArtThink(simulation_delay=0.01, verbose=False)
    eval_llm = DetailedAIArtEval(simulation_delay=0.01, verbose=False)

    result = await solve_async(
        goal='写一篇"AI与艺术"的短文（800–1200字）',
        think_llm=think_llm,
        eval_llm=eval_llm,
        budget=ExecutionBudget(max_depth=5, max_tokens=2000, max_time=10.0)
    )

    assert result.status == SolveStatus.COMPLETED
    assert "AI与艺术" in result.result
    assert len(result.result) > 500  # 应该是相当长的文章
    assert result.max_depth_reached > 0


@pytest.mark.asyncio
async def test_budget_constraints():
    """测试预算约束"""
    from src.fractal_think.types import DepthLimitExceeded, ResourceLimitExceeded, TimeLimitExceeded

    think_llm = AsyncMockThinkLLM(simulation_delay=0.01, verbose=False)
    eval_llm = AsyncMockEvalLLM(simulation_delay=0.01, verbose=False)

    # 测试深度限制 - 应该抛出异常
    with pytest.raises(DepthLimitExceeded):
        await solve_async(
            goal="复杂多层任务",
            think_llm=think_llm,
            eval_llm=eval_llm,
            budget=ExecutionBudget(max_depth=0, max_tokens=1000, max_time=5.0)  # 设置为0确保触发
        )


@pytest.mark.asyncio
async def test_token_tracking():
    """测试Token追踪功能"""
    think_llm = AsyncMockThinkLLM(simulation_delay=0.01, verbose=False)
    eval_llm = AsyncMockEvalLLM(simulation_delay=0.01, verbose=False)

    result = await solve_async(
        goal="Token追踪测试",
        think_llm=think_llm,
        eval_llm=eval_llm,
        budget=ExecutionBudget(max_depth=3, max_tokens=1000, max_time=5.0)
    )

    assert result.token_usage.total > 0
    assert result.token_usage.think_tokens >= 0
    assert result.token_usage.eval_tokens >= 0
    assert result.token_usage.think_calls > 0