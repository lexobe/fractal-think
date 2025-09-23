"""Execution tree node definitions."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class NodeStatus(Enum):
    """Execution node status lifecycle."""

    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class ExecutionNode:
    """Represents a node in the async execution tree."""

    node_id: str
    goal: str
    depth: int
    stage: str
    node_type: str
    status: NodeStatus = NodeStatus.RUNNING
    parent_id: Optional[str] = None
    parent: Optional["ExecutionNode"] = field(default=None, repr=False)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result_summary: Optional[str] = None
    error_message: Optional[str] = None
    todo: str = ""
    done: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List["ExecutionNode"] = field(default_factory=list)

    def add_child(self, child: "ExecutionNode") -> None:
        """Attach a child node and maintain parent linkage."""

        child.parent = self
        child.parent_id = self.node_id
        self.children.append(child)

    def mark_running(self, stage: Optional[str] = None, timestamp: Optional[float] = None) -> None:
        """Mark the node as running and update start timestamp."""

        if stage:
            self.stage = stage
        self.status = NodeStatus.RUNNING
        if timestamp is not None:
            self.started_at = timestamp
        elif self.started_at is None:
            self.started_at = time.time()
        self.finished_at = None
        self.error_message = None

    def mark_completed(
        self,
        summary: Optional[str] = None,
        *,
        stage: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Mark the node as completed and store summary information."""

        self.status = NodeStatus.COMPLETED
        self.stage = stage or "completed"
        self.result_summary = summary
        self.error_message = None
        self.finished_at = timestamp if timestamp is not None else time.time()

    def mark_failed(
        self,
        message: str,
        *,
        stage: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Mark the node as failed and capture error details."""

        self.status = NodeStatus.FAILED
        self.stage = stage or "failed"
        self.error_message = message
        self.finished_at = timestamp if timestamp is not None else time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the execution node tree into dictionaries."""

        return {
            "node_id": self.node_id,
            "goal": self.goal,
            "depth": self.depth,
            "stage": self.stage,
            "node_type": self.node_type,
            "status": self.status.value,
            "parent_id": self.parent_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result_summary": self.result_summary,
            "error_message": self.error_message,
            "todo": self.todo,
            "done": list(self.done),
            "metadata": self.metadata,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        parent: Optional["ExecutionNode"] = None,
    ) -> "ExecutionNode":
        """Restore an execution node tree from serialised data."""

        status = data.get("status", NodeStatus.RUNNING.value)
        node = cls(
            node_id=data["node_id"],
            goal=data.get("goal", ""),
            depth=data.get("depth", 0 if parent is None else parent.depth + 1),
            stage=data.get("stage", "think"),
            node_type=data.get("node_type", "task"),
            status=NodeStatus(status),
            parent_id=data.get("parent_id"),
            parent=parent,
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            result_summary=data.get("result_summary"),
            error_message=data.get("error_message"),
            todo=data.get("todo", ""),
            done=list(data.get("done", [])),
            metadata=data.get("metadata", {}),
            children=[],
        )

        for child_data in data.get("children", []):
            child = cls.from_dict(child_data, parent=node)
            node.children.append(child)

        return node


__all__ = ["NodeStatus", "ExecutionNode", "ExecutionFrame"]


def __getattr__(name: str):
    if name == "ExecutionFrame":
        warnings.warn(
            "ExecutionFrame 已弃用，请使用 ExecutionNode。",
            DeprecationWarning,
            stacklevel=2,
        )
        return ExecutionNode
    raise AttributeError(name)

