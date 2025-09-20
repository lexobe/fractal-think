"""
异步执行引擎 - 显式栈式状态机 + 事件驱动调度

实现从同步递归到异步事件驱动的架构转换：
- 显式ExecutionFrame栈管理
- 异步Think/Eval调度
- BudgetManager约束检查
- 状态快照和恢复支持
"""

import asyncio
import uuid
import time
import copy
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass, field

from .interfaces import AsyncThinkLLM, AsyncEvalLLM, create_async_think, create_async_eval
from .frame import ExecutionFrame, FrameState, SubTaskInfo
from .snapshot import ExecutionSnapshot
from .common import ExecutionBudget, BudgetManager, UnifiedTokenUsage, UnifiedLogger, ExecutionMode
try:
    from ..thinkon_core import S, SolveResult, SolveStatus, ThinkLLM, EvalLLM
except ImportError:
    # 当作为顶层模块运行时的回退导入
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from thinkon_core import S, SolveResult, SolveStatus, ThinkLLM, EvalLLM


@dataclass
class AsyncExecutionContext:
    """异步执行上下文"""
    think_llm: AsyncThinkLLM
    eval_llm: AsyncEvalLLM
    budget_manager: BudgetManager
    token_usage: UnifiedTokenUsage
    logger: UnifiedLogger

    # 执行状态
    frame_stack: List[ExecutionFrame]
    active_frame_id: Optional[str] = None

    # 执行统计
    start_time: float = field(default_factory=time.time)
    max_depth_reached: int = 0

    # 快照相关
    snapshot_callback: Optional[Callable[[ExecutionSnapshot], None]] = None
    auto_snapshot: bool = False


class AsyncExecutionEngine:
    """异步执行引擎 - 核心状态机实现"""

    def __init__(self, context: AsyncExecutionContext):
        self.context = context
        self._running = False
        self._current_snapshot_id: Optional[str] = None

    async def solve_async(self, goal: str, budget: Optional[ExecutionBudget] = None,
                         memory: Any = None, tools: Any = None) -> SolveResult:
        """异步求解入口 - 新任务"""
        if budget is None:
            budget = ExecutionBudget()

        # 创建根节点和根帧
        root_node = S(goal=goal)
        root_frame = ExecutionFrame(
            frame_id=str(uuid.uuid4()),
            node=root_node,
            state=FrameState.CREATED
        )

        # 初始化执行上下文
        self.context.frame_stack = [root_frame]
        self.context.active_frame_id = root_frame.frame_id
        self.context.budget_manager = BudgetManager(budget)
        self.context.token_usage.set_async_mode()

        self.context.logger.info(f"开始异步求解: {goal}")

        try:
            return await self._execute_state_machine()
        except Exception as e:
            self.context.logger.error(f"异步执行失败: {e}")
            execution_time = time.time() - self.context.start_time
            return SolveResult(
                status=SolveStatus.FAILED,
                result="执行异常终止",
                token_usage=self.context.token_usage,
                execution_time=execution_time,
                max_depth_reached=self.context.max_depth_reached,
                constraint_triggered=f"ExecutionError: {str(e)}"
            )

    async def resume_from_snapshot(self, snapshot: ExecutionSnapshot) -> SolveResult:
        """从快照恢复执行"""
        self.context.logger.info(f"从快照恢复执行: {snapshot.snapshot_id}")

        # 恢复执行状态
        self.context.frame_stack = snapshot.frame_stack
        self.context.active_frame_id = snapshot.active_frame_id

        if snapshot.budget:
            self.context.budget_manager = BudgetManager(snapshot.budget)

        if snapshot.global_token_usage:
            self.context.token_usage = snapshot.global_token_usage
            self.context.token_usage.set_async_mode()

        self._current_snapshot_id = snapshot.snapshot_id

        try:
            return await self._execute_state_machine()
        except Exception as e:
            self.context.logger.error(f"恢复执行失败: {e}")
            execution_time = time.time() - self.context.start_time
            return SolveResult(
                status=SolveStatus.FAILED,
                result="恢复执行异常",
                token_usage=self.context.token_usage,
                execution_time=execution_time,
                max_depth_reached=self.context.max_depth_reached,
                constraint_triggered=f"ResumeError: {str(e)}"
            )

    async def _execute_state_machine(self) -> SolveResult:
        """核心状态机执行循环"""
        self._running = True

        while self._running and self.context.frame_stack:
            current_frame = self._get_current_frame()
            if not current_frame:
                break

            # 更新最大深度
            current_depth = current_frame.node.level
            if current_depth > self.context.max_depth_reached:
                self.context.max_depth_reached = current_depth

            # 约束检查
            constraint_error = await self.context.budget_manager.check_constraints_async(
                current_frame.node, self.context.token_usage.total
            )
            if constraint_error:
                return await self._handle_constraint_violation(constraint_error, current_frame)

            # 状态转换
            try:
                next_state = await self._process_frame_state(current_frame)
                if next_state:
                    current_frame.state = next_state

                # 自动快照检查
                if self.context.auto_snapshot:
                    await self._check_auto_snapshot(current_frame)

            except Exception as e:
                self.context.logger.error(f"状态处理异常: {e}")
                current_frame.set_failed(str(e))
                return await self._handle_frame_failure(current_frame)

        # 执行完成
        execution_time = time.time() - self.context.start_time

        if self.context.frame_stack:
            root_frame = self.context.frame_stack[0]
            if root_frame.state == FrameState.COMPLETED:
                result = root_frame.node.done[-1] if root_frame.node.done else "任务完成"
                return SolveResult(
                    status=SolveStatus.COMPLETED,
                    result=result,
                    token_usage=self.context.token_usage,
                    execution_time=execution_time,
                    max_depth_reached=self.context.max_depth_reached
                )
            else:
                partial_results = []
                for frame in self.context.frame_stack:
                    partial_results.extend(frame.node.done)
                return SolveResult(
                    status=SolveStatus.DEGRADED,
                    result="任务未完全完成",
                    token_usage=self.context.token_usage,
                    execution_time=execution_time,
                    max_depth_reached=self.context.max_depth_reached,
                    partial_results=partial_results
                )

        return SolveResult(
            status=SolveStatus.FAILED,
            result="执行栈为空",
            token_usage=self.context.token_usage,
            execution_time=execution_time,
            max_depth_reached=self.context.max_depth_reached
        )

    async def _process_frame_state(self, frame: ExecutionFrame) -> Optional[FrameState]:
        """处理单个帧的状态转换"""
        if frame.state == FrameState.CREATED:
            return await self._handle_created_state(frame)
        elif frame.state == FrameState.THINKING:
            return await self._handle_thinking_state(frame)
        elif frame.state == FrameState.PLANNING:
            return await self._handle_planning_state(frame)
        elif frame.state == FrameState.FIRST_EVAL:
            return await self._handle_first_eval_state(frame)
        elif frame.state == FrameState.CALLING:
            return await self._handle_calling_state(frame)
        elif frame.state == FrameState.WAITING:
            return await self._handle_waiting_state(frame)
        elif frame.state == FrameState.CONTINUING:
            return await self._handle_continuing_state(frame)
        elif frame.state == FrameState.RETURNING:
            return await self._handle_returning_state(frame)
        elif frame.state in (FrameState.COMPLETED, FrameState.FAILED):
            return await self._handle_terminal_state(frame)

        return None

    async def _handle_created_state(self, frame: ExecutionFrame) -> FrameState:
        """处理CREATED状态 - 转到THINKING"""
        self.context.logger.debug(f"Frame {frame.frame_id}: CREATED -> THINKING")
        return FrameState.THINKING

    async def _handle_thinking_state(self, frame: ExecutionFrame) -> FrameState:
        """处理THINKING状态 - 执行Think操作"""
        self.context.logger.debug(f"Frame {frame.frame_id}: 执行Think操作")

        try:
            think_result = await self.context.think_llm(
                frame.node, memory=None, tools=None
            )

            # 记录token消耗
            tokens_used = think_result.get("tokens_used", 0)
            frame.add_frame_tokens(think_tokens=tokens_used)
            await self.context.token_usage.add_think_async(tokens_used)
            await self.context.budget_manager.consume_budget_async(tokens_delta=tokens_used)

            # 解析Think结果
            action_type = think_result.get("type", "TODO")
            description = think_result.get("description", "")

            if action_type == "RETURN":
                # 直接返回
                frame.node.done.append(description)
                return FrameState.RETURNING
            elif action_type == "TODO":
                # 需要分解，转到PLANNING
                frame.think_action = think_result
                # 规范要求：Plan_made 必须更新 S.todo
                frame.node.todo = description
                return FrameState.PLANNING
            else:
                raise ValueError(f"未知的Think action type: {action_type}")

        except Exception as e:
            frame.set_failed(f"Think操作失败: {e}")
            return FrameState.FAILED

    async def _handle_planning_state(self, frame: ExecutionFrame) -> FrameState:
        """处理PLANNING状态 - 解析计划并创建子任务"""
        self.context.logger.debug(f"Frame {frame.frame_id}: 解析计划创建子任务")

        try:
            if frame.think_action:
                plan_description = frame.think_action.get("description", "")

                # 智能计划解析 - 只保留真正的任务行
                subtask_lines = []
                for line in plan_description.split('\n'):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    # 识别任务行：支持多种格式
                    # 1. [] 格式（规范示例）
                    # 2. 数字格式（1. 2. 3.）
                    # 3. 步骤格式（步骤1：步骤2：）
                    # 4. 列表格式（- * •）
                    if (line.lstrip().startswith('[]') or
                        line.lstrip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) or
                        line.startswith(('- ', '* ', '• ')) or
                        line.startswith(('步骤', '任务', 'Step', 'Task')) and '：' in line):  # 支持步骤格式
                        # 清理任务描述，去掉前缀
                        task_text = line.lstrip()
                        if task_text.startswith('[]'):
                            task_text = task_text[2:].strip()
                        elif task_text.startswith(('- ', '* ', '• ')):
                            task_text = task_text[2:].strip()
                        elif any(task_text.startswith(f'{i}.') for i in range(1, 10)):
                            # 去掉数字前缀 "1. 2. 3." 等
                            task_text = task_text.split('.', 1)[1].strip()
                        elif task_text.startswith(('步骤', '任务', 'Step', 'Task')) and '：' in task_text:
                            # 去掉步骤前缀 "步骤1：" "任务A：" 等
                            task_text = task_text.split('：', 1)[1].strip()

                        subtask_lines.append(task_text)

                for task in subtask_lines:
                    if task:
                        frame.add_subtask(task)

                if frame.subtasks:
                    return FrameState.FIRST_EVAL
                else:
                    # 没有子任务，直接完成
                    frame.node.done.append(plan_description)
                    return FrameState.RETURNING
            else:
                frame.set_failed("PLANNING状态缺少think_action")
                return FrameState.FAILED

        except Exception as e:
            frame.set_failed(f"计划解析失败: {e}")
            return FrameState.FAILED

    async def _handle_first_eval_state(self, frame: ExecutionFrame) -> FrameState:
        """处理FIRST_EVAL状态 - 首次Eval决策"""
        return await self._execute_eval_logic(frame, FrameState.CALLING)

    async def _handle_calling_state(self, frame: ExecutionFrame) -> FrameState:
        """处理CALLING状态 - 创建子任务帧"""
        if frame.has_pending_subtasks:
            current_subtask = frame.current_subtask
            if current_subtask:
                # 创建子帧
                child_node = S(
                    goal=current_subtask.goal,
                    parent=frame.node
                )
                child_frame = ExecutionFrame(
                    frame_id=str(uuid.uuid4()),
                    node=child_node,
                    parent_frame_id=frame.frame_id,
                    state=FrameState.CREATED
                )

                # 推入栈
                self.context.frame_stack.append(child_frame)
                self.context.active_frame_id = child_frame.frame_id
                current_subtask.frame_id = child_frame.frame_id

                return FrameState.WAITING

        # 没有待处理子任务，直接返回
        return FrameState.RETURNING

    async def _handle_waiting_state(self, frame: ExecutionFrame) -> FrameState:
        """处理WAITING状态 - 等待子任务完成"""
        # 检查当前子任务是否完成
        current_subtask = frame.current_subtask
        if current_subtask and current_subtask.frame_id:
            child_frame = self._get_frame_by_id(current_subtask.frame_id)
            if child_frame and child_frame.is_completed:
                # 子任务完成，收集结果
                if child_frame.state == FrameState.COMPLETED:
                    result = child_frame.node.done[-1] if child_frame.node.done else "子任务完成"
                    frame.complete_current_subtask(result, child_frame.frame_id)
                elif child_frame.state == FrameState.FAILED:
                    error_msg = child_frame.error_msg or "子任务失败"
                    frame.fail_current_subtask(error_msg)

                # 移除子帧并转到CONTINUING
                self._remove_frame(child_frame.frame_id)
                return FrameState.CONTINUING

        # 子任务尚未完成，保持等待
        return FrameState.WAITING

    async def _handle_continuing_state(self, frame: ExecutionFrame) -> FrameState:
        """处理CONTINUING状态 - 继续执行或Eval决策"""
        return await self._execute_eval_logic(frame, FrameState.CALLING)

    async def _execute_eval_logic(self, frame: ExecutionFrame, default_next: FrameState) -> FrameState:
        """执行Eval逻辑 - 决策下一步行动"""
        try:
            eval_result = await self.context.eval_llm(frame.node, memory=None)

            # 记录token消耗
            tokens_used = eval_result.get("tokens_used", 0)
            frame.add_frame_tokens(eval_tokens=tokens_used)
            await self.context.token_usage.add_eval_async(tokens_used)
            await self.context.budget_manager.consume_budget_async(tokens_delta=tokens_used)

            # 解析Eval结果
            action_type = eval_result.get("type", "CALL")
            description = eval_result.get("description", "")

            frame.current_eval_action = eval_result

            if action_type == "RETURN":
                # 完成当前任务
                frame.node.done.append(description)
                return FrameState.RETURNING
            elif action_type == "CALL":
                # 继续执行子任务
                return default_next
            else:
                raise ValueError(f"未知的Eval action type: {action_type}")

        except Exception as e:
            frame.set_failed(f"Eval操作失败: {e}")
            return FrameState.FAILED

    async def _handle_returning_state(self, frame: ExecutionFrame) -> FrameState:
        """处理RETURNING状态 - 完成当前帧"""
        frame.set_completed()

        # 如果是根帧，结束执行
        if not frame.parent_frame_id:
            self._running = False

        return FrameState.COMPLETED

    async def _handle_terminal_state(self, frame: ExecutionFrame) -> Optional[FrameState]:
        """处理终态 - 清理和返回"""
        if frame.parent_frame_id:
            # 非根帧，切换到父帧
            self.context.active_frame_id = frame.parent_frame_id
        else:
            # 根帧完成，结束执行
            self._running = False

        return None

    async def _handle_constraint_violation(self, constraint_error: str,
                                         frame: ExecutionFrame) -> SolveResult:
        """处理约束违规 - 触发快照和优雅降级"""
        self.context.logger.warning(f"约束触发: {constraint_error}")

        # 创建约束触发快照
        snapshot = await self._create_snapshot(
            trigger_reason="constraint_violation",
            trigger_frame_id=frame.frame_id,
            constraint_triggered=constraint_error
        )

        # 调用快照回调
        if self.context.snapshot_callback:
            self.context.snapshot_callback(snapshot)

        # 构建降级结果
        partial_results = []
        failure_path = []
        current = frame.node
        while current is not None:
            failure_path.append(current.goal)
            if current.done:
                partial_results.extend(current.done)
            current = current.parent
        failure_path.reverse()

        degraded_result = f"约束触发优雅降级: {constraint_error}\n"
        if partial_results:
            degraded_result += f"部分完成: {'; '.join(partial_results[-3:])}"
        else:
            degraded_result += "尚无部分结果"

        execution_time = time.time() - self.context.start_time

        return SolveResult(
            status=SolveStatus.DEGRADED,
            result=degraded_result,
            token_usage=self.context.token_usage,
            execution_time=execution_time,
            max_depth_reached=self.context.max_depth_reached,
            constraint_triggered=constraint_error,
            partial_results=partial_results,
            failure_path=failure_path,
            failure_level=frame.node.level,
            failure_node_goal=frame.node.goal,
            failure_node_done=frame.node.done.copy()
        )

    async def _handle_frame_failure(self, frame: ExecutionFrame) -> SolveResult:
        """处理帧失败"""
        error_msg = frame.error_msg or "帧执行失败"
        execution_time = time.time() - self.context.start_time

        # 构建失败路径
        failure_path = []
        current = frame.node
        while current is not None:
            failure_path.append(current.goal)
            current = current.parent
        failure_path.reverse()

        return SolveResult(
            status=SolveStatus.FAILED,
            result=f"执行失败: {error_msg}",
            token_usage=self.context.token_usage,
            execution_time=execution_time,
            max_depth_reached=self.context.max_depth_reached,
            constraint_triggered=frame.constraint_triggered,
            failure_path=failure_path,
            failure_level=frame.node.level,
            failure_node_goal=frame.node.goal,
            failure_node_done=frame.node.done.copy()
        )

    async def _create_snapshot(self, trigger_reason: str = "manual",
                             trigger_frame_id: Optional[str] = None,
                             constraint_triggered: Optional[str] = None) -> ExecutionSnapshot:
        """创建执行快照"""
        snapshot_id = self._current_snapshot_id or str(uuid.uuid4())

        snapshot = ExecutionSnapshot(
            snapshot_id=snapshot_id,
            goal=self.context.frame_stack[0].node.goal if self.context.frame_stack else "",
            frame_stack=copy.deepcopy(self.context.frame_stack),
            active_frame_id=self.context.active_frame_id,
            budget=copy.deepcopy(self.context.budget_manager.budget),
            global_token_usage=copy.deepcopy(self.context.token_usage)
        )

        snapshot.set_trigger_info(trigger_reason, trigger_frame_id, constraint_triggered)
        return snapshot

    async def _check_auto_snapshot(self, frame: ExecutionFrame):
        """检查是否需要自动快照"""
        # 简单的自动快照策略 - 每完成5个子任务
        if len(frame.node.done) % 5 == 0 and len(frame.node.done) > 0:
            snapshot = await self._create_snapshot("auto_checkpoint", frame.frame_id)
            if self.context.snapshot_callback:
                self.context.snapshot_callback(snapshot)

    def _get_current_frame(self) -> Optional[ExecutionFrame]:
        """获取当前活跃帧"""
        if self.context.active_frame_id:
            return self._get_frame_by_id(self.context.active_frame_id)
        return None

    def _get_frame_by_id(self, frame_id: str) -> Optional[ExecutionFrame]:
        """根据ID获取帧"""
        for frame in self.context.frame_stack:
            if frame.frame_id == frame_id:
                return frame
        return None

    def _remove_frame(self, frame_id: str):
        """移除指定帧"""
        self.context.frame_stack = [f for f in self.context.frame_stack if f.frame_id != frame_id]


# 工厂函数
async def solve_async(goal: str,
                     think_llm: Union[ThinkLLM, AsyncThinkLLM],
                     eval_llm: Union[EvalLLM, AsyncEvalLLM],
                     budget: Optional[ExecutionBudget] = None,
                     logger: Optional[UnifiedLogger] = None,
                     memory: Any = None,
                     tools: Any = None) -> SolveResult:
    """异步求解工厂函数"""

    # 创建异步算子
    async_think = create_async_think(think_llm)
    async_eval = create_async_eval(eval_llm)

    # 创建执行上下文
    if logger is None:
        logger = UnifiedLogger(mode=ExecutionMode.ASYNC)

    token_usage = UnifiedTokenUsage()
    token_usage.set_async_mode()

    context = AsyncExecutionContext(
        think_llm=async_think,
        eval_llm=async_eval,
        budget_manager=BudgetManager(budget or ExecutionBudget()),
        token_usage=token_usage,
        logger=logger,
        frame_stack=[],
        start_time=time.time(),
        max_depth_reached=0
    )

    # 创建执行引擎
    engine = AsyncExecutionEngine(context)

    # 执行求解
    return await engine.solve_async(goal, budget, memory, tools)