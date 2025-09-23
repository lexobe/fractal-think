"""Memory component tests."""

import asyncio
import time
import pytest

from src.fractal_think.memory import Memory


def test_memory_recall_and_remember_flow():
    async def runner():
        memory = Memory(max_entry_length=256, max_total_length=512)
        context = {
            "session_id": "test-session",
            "node_id": "node-1",
            "stage": "think",
            "timestamp": time.time(),
        }
        stack = [{"frameId": "root", "depth": 0, "stage": "think", "nodeType": "test"}]

        recall_text = await memory.recall(context, frame_stack=stack)
        assert recall_text == ""
        assert context["version"] == 0

        await memory.remember(context, "第一次记忆内容", frame_stack=stack)
        assert context["version"] == 1

        recall_text = await memory.recall(context, frame_stack=stack)
        assert recall_text.strip() == "第一次记忆内容"

        # 写入另一阶段的记忆，think 调用不应读到 eval 数据
        eval_context = dict(context, stage="eval", node_id="node-1")
        eval_stack = [
            {"frameId": "root", "depth": 0, "stage": "eval", "nodeType": "test"}
        ]
        await memory.remember(eval_context, "Eval 内容", frame_stack=eval_stack)
        recall_text = await memory.recall(context, frame_stack=stack)
        assert "Eval 内容" not in recall_text

        # 再次写入相同内容应保持版本号不变，实现幂等。
        await memory.remember(context, "第一次记忆内容", frame_stack=stack)
        assert context["version"] == 1

        recall_text = await memory.recall(context, frame_stack=stack)
        assert recall_text.count("第一次记忆内容") == 1

    asyncio.run(runner())


def test_memory_validation_errors():
    async def runner():
        memory = Memory()
        base_context = {
            "session_id": "test-session",
            "node_id": "node-2",
            "stage": "think",
            "timestamp": time.time(),
        }

        stack = [{"frameId": "root", "depth": 0, "stage": "think", "nodeType": "test"}]

        with pytest.raises(ValueError):
            await memory.recall({}, frame_stack=stack)

        with pytest.raises(ValueError):
            await memory.recall(dict(base_context), frame_stack=None)

        with pytest.raises(ValueError):
            await memory.remember(dict(base_context, stage="unknown"), "记录", frame_stack=stack)

        with pytest.raises(ValueError):
            await memory.remember(dict(base_context), "   ", frame_stack=stack)

        with pytest.raises(ValueError):
            await memory.remember(dict(base_context), "记录", frame_stack=None)

        with pytest.raises(ValueError):
            await memory.recall(dict(base_context), frame_stack=[{"frameId": "f"}])

    asyncio.run(runner())
