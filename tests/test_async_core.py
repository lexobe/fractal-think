"""
异步核心功能测试
"""

import asyncio
import pytest

# 导入在pytest和直接运行时都能工作
try:
    from src.fractal_think import (
        solve_async,
        ExecutionBudget,
        SolveStatus,
        Memory,
        FrameStackProtocolError,
    )
    from src.fractal_think.examples.mock_operators import AsyncMockThinkLLM, AsyncMockEvalLLM, DetailedAIArtThink, DetailedAIArtEval
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.fractal_think import (
        solve_async,
        ExecutionBudget,
        SolveStatus,
        Memory,
        FrameStackProtocolError,
    )
    from src.fractal_think.examples.mock_operators import AsyncMockThinkLLM, AsyncMockEvalLLM, DetailedAIArtThink, DetailedAIArtEval


def test_solve_async_requires_frame_stack():
    async def runner():
        think_llm = AsyncMockThinkLLM(simulation_delay=0.01, verbose=False)
        eval_llm = AsyncMockEvalLLM(simulation_delay=0.01, verbose=False)

        memory = Memory()

        with pytest.raises(FrameStackProtocolError):
            await solve_async(
                goal="缺少frame stack",
                think_llm=think_llm,
                eval_llm=eval_llm,
                budget=ExecutionBudget(max_depth=2, max_tokens=100, max_time=1.0),
                memory=memory,
            )

    asyncio.run(runner())


def test_basic_async_solve():
    """测试基础异步求解功能"""

    async def runner():
        think_llm = AsyncMockThinkLLM(simulation_delay=0.01, verbose=False)
        eval_llm = AsyncMockEvalLLM(simulation_delay=0.01, verbose=False)

        memory = Memory()

        result = await solve_async(
            goal="简单测试任务",
            think_llm=think_llm,
            eval_llm=eval_llm,
            budget=ExecutionBudget(max_depth=3, max_tokens=1000, max_time=5.0),
            memory=memory,
            frame_stack=[],
        )

        assert result.status == SolveStatus.COMPLETED
        assert result.token_usage.total > 0
        assert think_llm.call_count > 0
        assert eval_llm.call_count >= 0

    asyncio.run(runner())


def test_detailed_ai_art_example():
    """测试详细的AI与艺术示例"""

    async def runner():
        think_llm = DetailedAIArtThink(simulation_delay=0.01, verbose=False)
        eval_llm = DetailedAIArtEval(simulation_delay=0.01, verbose=False)

        memory = Memory()

        result = await solve_async(
            goal='写一篇"AI与艺术"的短文（800–1200字）',
            think_llm=think_llm,
            eval_llm=eval_llm,
            budget=ExecutionBudget(max_depth=5, max_tokens=2000, max_time=10.0),
            memory=memory,
            frame_stack=[],
        )

        assert result.status == SolveStatus.COMPLETED
        assert "AI与艺术" in result.result
        assert len(result.result) > 500  # 应该是相当长的文章
        assert result.max_depth_reached > 0

    asyncio.run(runner())


def test_budget_constraints():
    """测试预算约束"""

    async def runner():
        from src.fractal_think.types import DepthLimitExceeded

        class DeepThink:
            async def __call__(
                self,
                node,
                memory_text="",
                memory_context=None,
                tools=None,
                frame_stack=None,
            ):
                if node.level < 2:
                    return {
                        "type": "TODO",
                        "description": "[] 深入一层",
                        "tokens_used": 1,
                        "remember": None,
                    }
                return {
                    "type": "RETURN",
                    "description": f"完成: {node.goal}",
                    "tokens_used": 1,
                    "remember": None,
                }

        class DeepEval:
            async def __call__(
                self,
                node,
                memory_text="",
                memory_context=None,
                frame_stack=None,
            ):
                if node.level < 2:
                    return {
                        "type": "CALL",
                        "description": f"扩展: {node.goal}",
                        "tokens_used": 1,
                        "remember": None,
                    }
                return {
                    "type": "RETURN",
                    "description": f"完成评估: {node.goal}",
                    "tokens_used": 1,
                    "remember": None,
                }

        with pytest.raises(DepthLimitExceeded):
            memory = Memory()

            await solve_async(
                goal="复杂多层任务",
                think_llm=DeepThink(),
                eval_llm=DeepEval(),
                budget=ExecutionBudget(max_depth=2, max_tokens=1000, max_time=5.0),
                memory=memory,
                frame_stack=[],
            )

    asyncio.run(runner())


def test_token_tracking():
    """测试Token追踪功能"""

    async def runner():
        think_llm = AsyncMockThinkLLM(simulation_delay=0.01, verbose=False)
        eval_llm = AsyncMockEvalLLM(simulation_delay=0.01, verbose=False)

        memory = Memory()

        result = await solve_async(
            goal="Token追踪测试",
            think_llm=think_llm,
            eval_llm=eval_llm,
            budget=ExecutionBudget(max_depth=3, max_tokens=1000, max_time=5.0),
            memory=memory,
            frame_stack=[],
        )

        assert result.token_usage.total > 0
        assert result.token_usage.think_tokens >= 0
        assert result.token_usage.eval_tokens >= 0
        assert result.token_usage.think_calls > 0

    asyncio.run(runner())


def test_memory_integration_tracks_think_and_eval_entries():
    async def runner():
        class MemoryAwareThink:
            async def __call__(
                self,
                node,
                memory_text="",
                memory_context=None,
                tools=None,
                frame_stack=None,
            ):
                version = int(memory_context.get("version", 0)) if memory_context else 0
                remember_note = f"think::{node.goal}::{version}"
                return {
                    "type": "TODO",
                    "description": "[] 汇总信息",
                    "tokens_used": 1,
                    "remember": remember_note,
                }

        class MemoryAwareEval:
            async def __call__(
                self,
                node,
                memory_text="",
                memory_context=None,
                frame_stack=None,
            ):
                version = int(memory_context.get("version", 0)) if memory_context else 0
                remember_note = f"eval::{node.goal}::{version}"
                return {
                    "type": "RETURN",
                    "description": f"完成: {node.goal}",
                    "tokens_used": 1,
                    "remember": remember_note,
                }

        memory = Memory()

        result = await solve_async(
            goal="测试记忆整合",
            think_llm=MemoryAwareThink(),
            eval_llm=MemoryAwareEval(),
            budget=ExecutionBudget(max_depth=3, max_tokens=100, max_time=5.0),
            memory=memory,
            frame_stack=[],
        )

        assert result.status == SolveStatus.COMPLETED

        entries = memory.dump_entries()
        assert len(entries) >= 2
        stages = {entry.context["stage"] for entry in entries}
        assert {"think", "eval"}.issubset(stages)
        versions = [entry.version for entry in entries]
        assert versions == sorted(versions)

    asyncio.run(runner())
