"""Frame Stack 数据结构与校验工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class FrameStackEntry:
    """Frame Stack 单个节点的元数据。"""

    frame_id: str
    depth: int
    stage: str  # think / eval
    node_type: str

    def to_dict(self) -> dict:
        return {
            "frameId": self.frame_id,
            "depth": self.depth,
            "stage": self.stage,
            "nodeType": self.node_type,
        }


class FrameStackProtocolError(RuntimeError):
    """Frame Stack 协议错误。"""


def validate_frame_stack(entries: List[FrameStackEntry]) -> None:
    """校验 Frame Stack 有序性与字段完整性。"""

    if not entries:
        return

    for idx, entry in enumerate(entries):
        if entry.depth < 0:
            raise FrameStackProtocolError(f"frame {entry.frame_id} depth {entry.depth} 非法")
        if entry.stage not in {"think", "eval"}:
            raise FrameStackProtocolError(
                f"frame {entry.frame_id} stage {entry.stage} 非法"
            )
        if idx > 0:
            prev = entries[idx - 1]
            if entry.depth < prev.depth:
                raise FrameStackProtocolError(
                    f"frame stack 深度倒退：{prev.frame_id}({prev.depth}) -> {entry.frame_id}({entry.depth})"
                )
            if entry.depth > prev.depth + 1:
                raise FrameStackProtocolError(
                    f"frame stack 深度跳跃：{prev.frame_id}({prev.depth}) -> {entry.frame_id}({entry.depth})"
                )


def append_frame_entry(
    stack: List[FrameStackEntry],
    entry: FrameStackEntry,
) -> List[FrameStackEntry]:
    """返回追加 entry 后的新堆栈，保持不可变语义。"""

    new_stack = stack + [entry]
    validate_frame_stack(new_stack)
    return new_stack


def pop_frame_entry(stack: List[FrameStackEntry], frame_id: str) -> List[FrameStackEntry]:
    """从堆栈末尾弹出指定 frame_id，确保匹配当前帧。"""

    if not stack:
        raise FrameStackProtocolError("尝试在空堆栈上弹出 frame")

    last = stack[-1]
    if last.frame_id != frame_id:
        raise FrameStackProtocolError(
            f"堆栈顶层 frame {last.frame_id} 与待弹出 {frame_id} 不一致"
        )

    return stack[:-1]


def frame_stack_to_json(stack: List[FrameStackEntry]) -> List[dict]:
    """将堆栈转换为日志友好的 JSON 列表。"""

    return [entry.to_dict() for entry in stack]
