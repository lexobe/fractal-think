"""
异步执行引擎 - ExecutionNode 树 + 显式状态栈

新的实现以 ExecutionNode 树为核心：
- Runtime 栈保存 ExecutionNode，节点状态与序列化统一
- 支持 ExecutionNodeObserver 监听节点生命周期
- 可通过 execution_tree 参数恢复运行
"""

from __future__ import annotations

import asyncio
import uuid
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Union, Iterable

from .interfaces import (
    AsyncThinkLLM,
    AsyncEvalLLM,
    ExecutionNodeObserver,
    create_async_think,
    create_async_eval,
)
from .execution_node import ExecutionNode, NodeStatus
from .common import ExecutionBudget, BudgetManager, UnifiedTokenUsage, UnifiedLogger
from .types import (
    S,
    SolveResult,
    SolveStatus,
    TokenUsage,
    ConstraintViolationError,
)


class ExecutionFailureError(RuntimeError):
    """执行失败异常"""

    def __init__(self, message: str, node: Optional[ExecutionNode] = None):
        super().__init__(message)
        self.node = node


@dataclass
class AsyncExecutionContext:
    """异步执行上下文"""

    session_id: str
    budget_manager: BudgetManager
    token_usage: UnifiedTokenUsage
    logger: UnifiedLogger
    start_time: float = 0.0
    max_depth_reached: int = 0
    observers: List[ExecutionNodeObserver] = field(default_factory=list)


class ExecutionStage(str, Enum):
    """ExecutionNode 的运行阶段"""

    THINK = "think"
    PLANNING = "planning"
    FIRST_EVAL = "first_eval"
    EVAL = "eval"
    RETURNING = "returning"
    FAILED = "failed"


@dataclass
class SubTaskInfo:
    """子任务信息"""

    task_id: str
    description: str
    created_at: float = field(default_factory=time.time)
    status: str = "pending"


@dataclass
class NodeRuntimeState:
    """执行节点的运行时状态"""

    s_node: S
    stage: ExecutionStage = ExecutionStage.THINK
    think_action: Optional[Dict[str, Any]] = None
    eval_action: Optional[Dict[str, Any]] = None
    subtasks: List[SubTaskInfo] = field(default_factory=list)
    current_subtask_index: int = 0
    think_tokens: int = 0
    eval_tokens: int = 0
    failure_reason: Optional[str] = None

    def add_subtask(self, node_id: str, description: str) -> SubTaskInfo:
        task = SubTaskInfo(
            task_id=f"{node_id}_subtask_{len(self.subtasks)}",
            description=description,
        )
        self.subtasks.append(task)
        return task

    def get_current_subtask(self) -> Optional[SubTaskInfo]:
        if 0 <= self.current_subtask_index < len(self.subtasks):
            return self.subtasks[self.current_subtask_index]
        return None

    def advance_subtask(self) -> bool:
        if self.current_subtask_index < len(self.subtasks) - 1:
            self.current_subtask_index += 1
            return True
        return False


class AsyncExecutionEngine:
    """异步执行引擎"""

    def __init__(self, context: AsyncExecutionContext):
        self.context = context
        self.execution_stack: List[ExecutionNode] = []
        self.node_states: Dict[str, NodeRuntimeState] = {}
        self.resume_paths: List[List[ExecutionNode]] = []
        self.root_node: Optional[ExecutionNode] = None

    async def solve(
        self,
        goal: str,
        think_llm: AsyncThinkLLM,
        eval_llm: AsyncEvalLLM,
        memory: Any = None,
        tools: Any = None,
        execution_tree: Optional[Union[ExecutionNode, Dict[str, Any]]] = None,
    ) -> SolveResult:
        """执行异步求解"""

        self.context.start_time = time.time()
        self.execution_stack.clear()
        self.node_states.clear()
        self.resume_paths.clear()

        try:
            if execution_tree is not None:
                self.root_node = self._load_execution_tree(goal, execution_tree)
            else:
                self.root_node = self._initialize_root(goal)

            self.resume_paths = self._build_resume_paths(self.root_node)

            # 如果没有需要执行的节点且根节点已完成，直接返回历史结果
            if not self.resume_paths and self.root_node.status == NodeStatus.COMPLETED:
                summary = self.root_node.result_summary or (
                    self.root_node.done[-1] if self.root_node.done else f"任务完成: {self.root_node.goal}"
                )
                return self._create_result(summary, SolveStatus.COMPLETED)

            result = await self._execute_state_machine(think_llm, eval_llm, memory, tools)
            return self._create_result(result, SolveStatus.COMPLETED)

        except ConstraintViolationError:
            self.context.logger.warning("约束触发，终止执行")
            raise
        except ExecutionFailureError as exc:
            self.context.logger.error(f"执行失败: {exc}")
            return self._create_result(str(exc), SolveStatus.FAILED)
        except Exception as exc:  # pragma: no cover - 防御性
            self.context.logger.error(f"执行异常: {exc}")
            return self._create_result(f"执行异常: {exc}", SolveStatus.FAILED)

    def _initialize_root(self, goal: str) -> ExecutionNode:
        root_s = S(goal=goal)
        root_node = ExecutionNode(
            node_id=str(uuid.uuid4()),
            goal=goal,
            depth=0,
            stage=ExecutionStage.THINK.value,
            node_type="root",
        )
        self.node_states[root_node.node_id] = NodeRuntimeState(s_node=root_s)
        self._sync_node_state(root_node, self.node_states[root_node.node_id])
        self._notify_observers("created", root_node)
        return root_node

    def _load_execution_tree(
        self,
        goal: str,
        execution_tree: Union[ExecutionNode, Dict[str, Any]],
    ) -> ExecutionNode:
        if isinstance(execution_tree, dict):
            root = ExecutionNode.from_dict(execution_tree)
        else:
            root = execution_tree

        root.parent = None
        root.parent_id = None
        self._reset_depths(root, 0)

        if not root.goal:
            root.goal = goal

        self._rebuild_runtime_state(root, None)
        return root

    def _reset_depths(self, node: ExecutionNode, depth: int) -> None:
        node.depth = depth
        for child in node.children:
            child.parent = node
            child.parent_id = node.node_id
            self._reset_depths(child, depth + 1)

    def _rebuild_runtime_state(self, node: ExecutionNode, parent_s: Optional[S]) -> None:
        s_node = S(goal=node.goal, parent=parent_s, todo=node.todo, done=list(node.done))
        state = NodeRuntimeState(s_node=s_node)

        # metadata 恢复
        metadata = dict(node.metadata or {})
        resume_stage = metadata.get("resume_stage")
        if node.status == NodeStatus.FAILED and resume_stage:
            state.stage = self._stage_from_string(resume_stage)
        else:
            state.stage = self._stage_from_string(node.stage)
        state.think_tokens = int(metadata.get("think_tokens", 0))
        state.eval_tokens = int(metadata.get("eval_tokens", 0))
        state.current_subtask_index = int(metadata.get("current_subtask_index", 0))
        subtasks: List[SubTaskInfo] = []
        for idx, item in enumerate(metadata.get("subtasks", [])):
            subtasks.append(
                SubTaskInfo(
                    task_id=item.get("task_id", f"{node.node_id}_subtask_{idx}"),
                    description=item.get("description", ""),
                    created_at=item.get("created_at", time.time()),
                    status=item.get("status", "pending"),
                )
            )
        state.subtasks = subtasks

        self.node_states[node.node_id] = state
        self._sync_node_state(node, state)

        for child in node.children:
            self._rebuild_runtime_state(child, s_node)

    def _build_resume_paths(self, root: ExecutionNode) -> List[List[ExecutionNode]]:
        paths: List[List[ExecutionNode]] = []

        def dfs(node: ExecutionNode, trail: List[ExecutionNode]) -> bool:
            trail.append(node)
            child_pending = False
            for child in node.children:
                if dfs(child, trail):
                    child_pending = True
            needs_resume = node.status in (NodeStatus.RUNNING, NodeStatus.FAILED)
            if needs_resume and not child_pending:
                paths.append(list(trail))
            trail.pop()
            return child_pending or needs_resume

        dfs(root, [])

        paths.sort(key=lambda path: (path[-1].started_at or 0.0, len(path)))
        if not paths and root.status != NodeStatus.COMPLETED:
            return [[root]]
        return paths

    async def _execute_state_machine(
        self,
        think_llm: AsyncThinkLLM,
        eval_llm: AsyncEvalLLM,
        memory: Any,
        tools: Any,
    ) -> str:
        max_iterations = 2000

        for _ in range(max_iterations):
            if not self.execution_stack:
                if self.resume_paths:
                    next_path = self.resume_paths.pop(0)
                    self._push_resume_path(next_path)
                    continue
                break

            current_node = self.execution_stack[-1]
            runtime_state = self.node_states[current_node.node_id]

            self.context.max_depth_reached = max(
                self.context.max_depth_reached,
                current_node.depth,
            )

            await self.context.budget_manager.check_constraints_async(runtime_state.s_node, 0)

            current_stage = runtime_state.stage

            try:
                if current_stage == ExecutionStage.THINK:
                    new_stage = await self._handle_think_state(
                        current_node, runtime_state, think_llm, memory, tools
                    )
                elif current_stage == ExecutionStage.PLANNING:
                    new_stage = await self._handle_planning_state(current_node, runtime_state)
                elif current_stage in (ExecutionStage.FIRST_EVAL, ExecutionStage.EVAL):
                    is_first = current_stage == ExecutionStage.FIRST_EVAL
                    new_stage = await self._handle_eval_state(
                        current_node,
                        runtime_state,
                        eval_llm,
                        memory,
                        is_first=is_first,
                    )
                elif current_stage == ExecutionStage.RETURNING:
                    new_stage = await self._handle_return_state(current_node, runtime_state)
                else:
                    new_stage = current_stage

                runtime_state.stage = new_stage
                current_node.stage = new_stage.value

                if new_stage == ExecutionStage.RETURNING:
                    result = await self._handle_return(current_node, runtime_state)
                    if self.execution_stack and result:
                        parent_node = self.execution_stack[-1]
                        parent_state = self.node_states[parent_node.node_id]
                        parent_state.s_node.done.append(result)
                        self._sync_node_state(parent_node, parent_state)
                        parent_state.stage = ExecutionStage.EVAL
                        parent_node.stage = parent_state.stage.value
                    elif result:
                        return result
                elif new_stage == ExecutionStage.FAILED:
                    self._handle_failure(current_node, runtime_state, current_stage)

            except Exception as exc:
                self.context.logger.error(f"帧处理异常: {exc}")
                raise

        if self.execution_stack or self.resume_paths:
            raise RuntimeError("执行未在最大迭代限制内完成")

        if self.root_node:
            if self.root_node.result_summary:
                return self.root_node.result_summary
            if self.root_node.done:
                return self.root_node.done[-1]
            return f"任务完成: {self.root_node.goal}"
        return "任务完成"

    def _push_resume_path(self, path: Iterable[ExecutionNode]) -> None:
        existing = {node.node_id for node in self.execution_stack}
        for node in path:
            if node.node_id in existing:
                continue
            runtime = self.node_states[node.node_id]
            node.mark_running(stage=runtime.stage.value)
            self._notify_observers("started", node)
            self.execution_stack.append(node)
            existing.add(node.node_id)

    async def _handle_think_state(
        self,
        node: ExecutionNode,
        runtime: NodeRuntimeState,
        think_llm: AsyncThinkLLM,
        memory: Any,
        tools: Any,
    ) -> ExecutionStage:
        self.context.logger.debug(f"Node {node.node_id}: 执行Think")

        try:
            think_result = await think_llm(runtime.s_node, memory, tools)
            if asyncio.iscoroutine(think_result):
                raise TypeError("Think算子返回了未await的协程")

            tokens_used = think_result.get("tokens_used", 0)
            runtime.think_tokens += tokens_used
            await self.context.token_usage.add_think_async(tokens_used)
            await self.context.budget_manager.consume_budget_async(tokens_delta=tokens_used)

            action_type = think_result.get("type", "TODO")
            description = think_result.get("description", "")

            if action_type == "RETURN":
                runtime.s_node.done.append(description)
                node.result_summary = description
                self._sync_node_state(node, runtime)
                return ExecutionStage.RETURNING
            elif action_type == "TODO":
                runtime.think_action = think_result
                runtime.s_node.todo = description
                self._sync_node_state(node, runtime)
                return ExecutionStage.PLANNING
            else:
                raise ValueError(f"未知的Think action type: {action_type}")

        except ConstraintViolationError:
            raise
        except Exception as exc:
            runtime.failure_reason = f"Think操作失败: {exc}"
            return ExecutionStage.FAILED
        finally:
            try:
                await self.context.budget_manager.check_constraints_async(runtime.s_node, 0)
            except ConstraintViolationError:
                raise

    async def _handle_planning_state(
        self,
        node: ExecutionNode,
        runtime: NodeRuntimeState,
    ) -> ExecutionStage:
        self.context.logger.debug(f"Node {node.node_id}: 解析计划")

        try:
            if not runtime.think_action:
                runtime.failure_reason = "没有Think结果可供解析"
                return ExecutionStage.FAILED

            plan_description = runtime.think_action.get("description", "")
            subtask_lines: List[str] = []

            for line in plan_description.split("\n"):
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue
                if (
                    raw.lstrip().startswith("[]")
                    or raw.lstrip().startswith(tuple(f"{i}." for i in range(1, 10)))
                    or raw.startswith(("- ", "* ", "• "))
                    or (raw.startswith(("步骤", "任务", "Step", "Task")) and "：" in raw)
                ):
                    task_text = raw.lstrip()
                    if task_text.startswith("[]"):
                        task_text = task_text[2:].strip()
                    elif task_text.startswith(("- ", "* ", "• ")):
                        task_text = task_text[2:].strip()
                    elif any(task_text.startswith(f"{i}.") for i in range(1, 10)):
                        task_text = task_text.split(".", 1)[1].strip()
                    elif task_text.startswith(("步骤", "任务", "Step", "Task")) and "：" in task_text:
                        task_text = task_text.split("：", 1)[1].strip()
                    if task_text:
                        runtime.add_subtask(node.node_id, task_text)

            if runtime.subtasks:
                runtime.stage = ExecutionStage.FIRST_EVAL
                node.stage = runtime.stage.value
                self._sync_node_state(node, runtime)
                return ExecutionStage.FIRST_EVAL

            runtime.s_node.done.append(plan_description)
            node.result_summary = plan_description
            self._sync_node_state(node, runtime)
            return ExecutionStage.RETURNING

        except Exception as exc:
            runtime.failure_reason = f"计划解析失败: {exc}"
            return ExecutionStage.FAILED

    async def _handle_eval_state(
        self,
        node: ExecutionNode,
        runtime: NodeRuntimeState,
        eval_llm: AsyncEvalLLM,
        memory: Any,
        *,
        is_first: bool,
    ) -> ExecutionStage:
        self.context.logger.debug(
            f"Node {node.node_id}: 执行{'首次' if is_first else ''}Eval"
        )

        try:
            eval_result = await eval_llm(runtime.s_node, memory)
            if asyncio.iscoroutine(eval_result):
                raise TypeError("Eval算子返回了未await的协程")

            tokens_used = eval_result.get("tokens_used", 0)
            runtime.eval_tokens += tokens_used
            await self.context.token_usage.add_eval_async(tokens_used)
            await self.context.budget_manager.consume_budget_async(tokens_delta=tokens_used)

            action_type = eval_result.get("type", "RETURN")
            description = eval_result.get("description", "")

            if action_type == "CALL":
                child_s = S(goal=description, parent=runtime.s_node)
                child_node = ExecutionNode(
                    node_id=str(uuid.uuid4()),
                    goal=description,
                    depth=node.depth + 1,
                    stage=ExecutionStage.THINK.value,
                    node_type="task",
                )
                node.add_child(child_node)
                self.node_states[child_node.node_id] = NodeRuntimeState(s_node=child_s)
                self._sync_node_state(child_node, self.node_states[child_node.node_id])
                self._notify_observers("created", child_node)
                self.execution_stack.append(child_node)
                child_node.mark_running(stage=ExecutionStage.THINK.value)
                self._notify_observers("started", child_node)
                return ExecutionStage.EVAL

            if action_type == "RETURN":
                runtime.s_node.done.append(description)
                node.result_summary = description
                self._sync_node_state(node, runtime)
                return ExecutionStage.RETURNING

            raise ValueError(f"未知的Eval action type: {action_type}")

        except ConstraintViolationError:
            raise
        except Exception as exc:
            runtime.failure_reason = f"Eval操作失败: {exc}"
            return ExecutionStage.FAILED
        finally:
            try:
                await self.context.budget_manager.check_constraints_async(runtime.s_node, 0)
            except ConstraintViolationError:
                raise

    async def _handle_return_state(
        self,
        node: ExecutionNode,
        runtime: NodeRuntimeState,
    ) -> ExecutionStage:
        # RETURNING 阶段的处理由 _handle_return 统一完成
        return ExecutionStage.RETURNING

    async def _handle_return(
        self,
        node: ExecutionNode,
        runtime: NodeRuntimeState,
    ) -> Optional[str]:
        self.context.logger.debug(f"Node {node.node_id}: 处理返回")

        if self.execution_stack and self.execution_stack[-1] == node:
            self.execution_stack.pop()

        summary = (
            runtime.s_node.done[-1]
            if runtime.s_node.done
            else f"任务完成: {runtime.s_node.goal}"
        )

        node.mark_completed(summary=summary)
        self._sync_node_state(node, runtime)
        node.metadata.pop("resume_stage", None)
        self._notify_observers("completed", node)

        if node is self.root_node:
            self.root_node.result_summary = summary

        return summary

    def _handle_failure(
        self,
        node: ExecutionNode,
        runtime: NodeRuntimeState,
        previous_stage: ExecutionStage,
    ) -> None:
        message = runtime.failure_reason or f"任务失败: {runtime.s_node.goal}"
        self._sync_node_state(node, runtime)
        metadata = dict(node.metadata or {})
        metadata["resume_stage"] = previous_stage.value
        node.metadata = metadata
        node.mark_failed(message)
        self._notify_observers("failed", node)
        raise ExecutionFailureError(message, node)

    def _stage_from_string(self, stage: str) -> ExecutionStage:
        try:
            return ExecutionStage(stage)
        except ValueError:
            if stage == "completed":
                return ExecutionStage.RETURNING
            if stage == "failed":
                return ExecutionStage.FAILED
            return ExecutionStage.THINK

    def _sync_node_state(self, node: ExecutionNode, runtime: NodeRuntimeState) -> None:
        node.todo = runtime.s_node.todo
        node.done = list(runtime.s_node.done)
        metadata = dict(node.metadata or {})
        metadata.update(
            {
                "think_tokens": runtime.think_tokens,
                "eval_tokens": runtime.eval_tokens,
                "current_subtask_index": runtime.current_subtask_index,
                "subtasks": [
                    {
                        "task_id": sub.task_id,
                        "description": sub.description,
                        "created_at": sub.created_at,
                        "status": sub.status,
                    }
                    for sub in runtime.subtasks
                ],
            }
        )
        node.metadata = metadata

    def _notify_observers(self, event: str, node: ExecutionNode) -> None:
        callback_name = f"on_node_{event}"
        for observer in list(self.context.observers):
            callback = getattr(observer, callback_name, None)
            if not callback:
                continue
            try:
                callback(node)
            except Exception as exc:  # pragma: no cover - 观察者异常不应中断执行
                self.context.logger.debug(f"Observer error on {event}: {exc}")

    def _create_result(self, result: str, status: SolveStatus) -> SolveResult:
        execution_time = time.time() - self.context.start_time

        token_usage = TokenUsage(
            total=self.context.token_usage.total,
            think_calls=self.context.token_usage.think_calls,
            eval_calls=self.context.token_usage.eval_calls,
            think_tokens=self.context.token_usage.think_tokens,
            eval_tokens=self.context.token_usage.eval_tokens,
        )

        return SolveResult(
            status=status,
            result=result,
            token_usage=token_usage,
            execution_time=execution_time,
            max_depth_reached=self.context.max_depth_reached,
        )


async def solve_async(
    goal: str,
    think_llm: Union[AsyncThinkLLM, Any],
    eval_llm: Union[AsyncEvalLLM, Any],
    budget: Optional[ExecutionBudget] = None,
    logger: Optional[UnifiedLogger] = None,
    memory: Any = None,
    tools: Any = None,
    execution_tree: Optional[Union[ExecutionNode, Dict[str, Any]]] = None,
    observers: Optional[List[ExecutionNodeObserver]] = None,
) -> SolveResult:
    """
    异步求解主入口函数

    Args:
        goal: 求解目标
        think_llm: Think算子（支持自动适配同步算子）
        eval_llm: Eval算子（支持自动适配同步算子）
        budget: 执行预算约束
        logger: 日志器
        memory: 内存上下文
        tools: 工具上下文
        execution_tree: 历史 ExecutionNode 树或序列化字典
        observers: ExecutionNodeObserver 列表

    Returns:
        SolveResult: 求解结果
    """

    if budget is None:
        budget = ExecutionBudget()
    if logger is None:
        logger = UnifiedLogger()

    async_think = create_async_think(think_llm)
    async_eval = create_async_eval(eval_llm)

    context = AsyncExecutionContext(
        session_id=str(uuid.uuid4()),
        budget_manager=BudgetManager(budget),
        token_usage=UnifiedTokenUsage(),
        logger=logger,
        observers=list(observers or []),
    )

    engine = AsyncExecutionEngine(context)
    return await engine.solve(
        goal,
        async_think,
        async_eval,
        memory,
        tools,
        execution_tree=execution_tree,
    )

