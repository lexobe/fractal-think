"""异步记忆模块，实现 `recall`/`remember` 接口。

本模块提供一个默认的进程内记忆后端，满足项目提出的能力需求：

- 提供可实例化的 :class:`Memory` 类，暴露异步 ``recall`` 与 ``remember`` 方法；
- 调用方需传入包含 ``node_id``、执行 ``stage``（think/eval）与 ``timestamp`` 的上下文，以便追踪；
- ``recall`` 返回可直接拼入 Prompt 的 UTF-8 字符串，并同步写回最新版本号；
- ``remember`` 会在不覆盖并发写入的前提下持久化 UTF-8 文本，并记录版本信息。

默认实现面向单进程使用，若需接入 Redis、数据库等后端，可在遵守接口约束的前提下扩展或替换实现。
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional

__all__ = ["Memory", "MemoryEntry"]

from .frame_stack import FrameStackEntry, FrameStackProtocolError, validate_frame_stack


@dataclass
class MemoryEntry:
    """记忆条目结构体。"""

    content: str
    context: Dict[str, object]
    version: int
    created_at: float


class Memory:
    """默认的进程内异步记忆实现。

    参数：
        max_entry_length: 单条记忆允许的最大长度，超过则截断保留最新内容；
        max_total_length: 返回给同一节点的聚合文本长度上限；
        max_recent_hashes: 记忆去重哈希的保留数量，上限后会淘汰最旧记录。
    """

    def __init__(
        self,
        *,
        max_entry_length: int = 1024,
        max_total_length: Optional[int] = 4096,
        max_recent_hashes: int = 1024,
    ) -> None:
        self._max_entry_length = max(1, max_entry_length)
        self._max_total_length = max_total_length
        self._entries: List[MemoryEntry] = []
        self._lock = asyncio.Lock()
        self._version = 0
        self._recent_hashes: "OrderedDict[str, int]" = OrderedDict()
        self._max_recent_hashes = max(1, max_recent_hashes)

    async def recall(
        self,
        context: Dict[str, object],
        *,
        frame_stack: Optional[List[Dict[str, object]]] = None,
    ) -> str:
        """根据上下文返回最新记忆文本。"""

        self._validate_context(context)
        self._validate_frame_stack(frame_stack)

        async with self._lock:
            latest_version = self._version
            context["version"] = latest_version

            node_id = context.get("node_id")
            stage = context.get("stage")
            # 默认按 node_id + stage 聚合记忆，确保 Think/Eval 不相互污染。
            relevant_entries = [
                entry
                for entry in self._entries
                if entry.context.get("node_id") == node_id
                and entry.context.get("stage") == stage
            ]

            combined = "\n".join(entry.content for entry in relevant_entries)
            if self._max_total_length is not None and len(combined) > self._max_total_length:
                combined = combined[-self._max_total_length :]

            return combined

    async def remember(
        self,
        context: Dict[str, object],
        payload: str,
        *,
        frame_stack: Optional[List[Dict[str, object]]] = None,
    ) -> None:
        """写入新的记忆文本。"""

        self._validate_context(context)
        self._validate_frame_stack(frame_stack)

        if payload is None:
            raise ValueError("payload 不能为空字符串")

        payload_str = str(payload).strip()
        if not payload_str:
            raise ValueError("payload 不能为空字符串")

        if len(payload_str) > self._max_entry_length:
            payload_str = payload_str[: self._max_entry_length]

        payload_hash = self._hash_payload(context, payload_str)

        async with self._lock:
            if self._recent_hashes.get(payload_hash) == self._version:
                # 同一版本重复写入，直接跳过。
                context["version"] = self._version
                return

            if self._recent_hashes.get(payload_hash) is not None:
                # 历史上已写入完全相同的负载，保持幂等。
                context["version"] = self._recent_hashes[payload_hash]
                return

            self._version += 1
            current_version = self._version
            context_copy = dict(context)
            context_copy["version"] = current_version

            entry = MemoryEntry(
                content=payload_str,
                context=context_copy,
                version=current_version,
                created_at=time.time(),
            )
            self._entries.append(entry)

            # 记录哈希，按上限淘汰最旧元素，防止内存无限增长。
            self._recent_hashes[payload_hash] = current_version
            while len(self._recent_hashes) > self._max_recent_hashes:
                self._recent_hashes.popitem(last=False)

            context["version"] = current_version

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def _validate_frame_stack(self, stack: Optional[List[Dict[str, object]]]) -> None:
        if stack is None:
            raise ValueError("frame_stack 必须提供，用于追踪执行路径")

        entries = []
        for item in stack:
            if not isinstance(item, dict):
                raise ValueError("frame_stack 元素必须为字典")
            try:
                entry = FrameStackEntry(
                    frame_id=str(item["frameId"]),
                    depth=int(item["depth"]),
                    stage=str(item["stage"]),
                    node_type=str(item.get("nodeType", "unknown")),
                )
            except KeyError as exc:
                raise ValueError(f"frame_stack 缺少字段: {exc}") from exc
            entries.append(entry)

        try:
            validate_frame_stack(entries)
        except FrameStackProtocolError as exc:
            raise ValueError(f"frame_stack 协议错误: {exc}") from exc

    def _validate_context(self, context: Dict[str, object]) -> None:
        required = {"node_id", "stage", "timestamp"}
        missing = [key for key in required if key not in context]
        if missing:
            raise ValueError(f"context 缺少必需字段: {', '.join(missing)}")

        stage = context.get("stage")
        if stage not in {"think", "eval"}:
            raise ValueError("context['stage'] 必须为 'think' 或 'eval'")

        if not isinstance(context.get("timestamp"), (int, float)):
            raise ValueError("context['timestamp'] 必须是数值型时间戳")

    def _hash_payload(self, context: Dict[str, object], payload: str) -> str:
        hasher = hashlib.sha256()
        # 去重域：node_id + stage + payload。node_id 在默认引擎中已唯一，
        # stage 作为额外维度保留扩展空间，便于上层按阶段自定义策略。
        # 若需要跨会话做强幂等，可扩展包含 session_id 或其他字段。
        hasher.update(str(context.get("node_id")).encode("utf-8"))
        stage = str(context.get("stage"))
        hasher.update(stage.encode("utf-8"))
        hasher.update(payload.encode("utf-8"))
        return hasher.hexdigest()

    # 对外调试接口 -----------------------------------------------------
    def dump_entries(self) -> List[MemoryEntry]:  # pragma: no cover - 调试用途
        return list(self._entries)
