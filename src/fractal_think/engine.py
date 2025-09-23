"""
异步执行引擎 - 显式栈式状态机 + 事件驱动调度

实现高效的异步分形思考执行引擎：
- 显式ExecutionFrame栈管理
- 异步Think/Eval调度
- BudgetManager约束检查
- 智能计划解析
"""

import asyncio
import uuid
import time
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass

from .interfaces import AsyncThinkLLM, AsyncEvalLLM, create_async_think, create_async_eval
from .frame import ExecutionFrame, FrameState, identify_node_type
from .frame_stack import (
    FrameStackEntry,
    FrameStackProtocolError,
    append_frame_entry,
    pop_frame_entry,
    frame_stack_to_json,
    validate_frame_stack,
)
from .common import ExecutionBudget, BudgetManager, UnifiedTokenUsage, UnifiedLogger, ExecutionMode
from .types import (
    S,
    SolveResult,
    SolveStatus,
    TokenUsage,
    ConstraintViolationError,
)


class ExecutionFailureError(RuntimeError):
    """执行失败异常"""

    def __init__(self, message: str, frame: Optional['ExecutionFrame'] = None):
        super().__init__(message)
        self.frame = frame


@dataclass
class AsyncExecutionContext:
    """异步执行上下文"""
    session_id: str
    budget_manager: BudgetManager
    token_usage: UnifiedTokenUsage
    logger: UnifiedLogger
    start_time: float = 0.0
    max_depth_reached: int = 0


class AsyncExecutionEngine:
    """异步执行引擎"""

    def __init__(self, context: AsyncExecutionContext):
        self.context = context
        self.frame_stack: List[ExecutionFrame] = []

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------
    def _memory_available(self, memory: Any) -> bool:
        return (
            memory is not None
            and hasattr(memory, "recall")
            and hasattr(memory, "remember")
        )

    def _build_memory_context(self, frame: ExecutionFrame, stage: str) -> Dict[str, Any]:
        return {
            "session_id": self.context.session_id,
            "node_id": frame.frame_id,
            "stage": stage,
            "timestamp": time.time(),
            # goal 信息仅用于调试追踪，若担心体积可在自定义实现中剔除或压缩
            "goal": frame.node.goal,
            "depth": frame.depth,
        }

    async def _prepare_memory_inputs(
        self,
        frame: ExecutionFrame,
        memory: Any,
        stage: str,
        frame_stack: List[FrameStackEntry],
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if not self._memory_available(memory):
            return "", None

        context = self._build_memory_context(frame, stage)
        try:
            memory_text = await memory.recall(
                context,
                frame_stack=frame_stack_to_json(frame_stack),
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.context.logger.warning(
                f"记忆读取失败[{stage}] frame={frame.frame_id}: {exc} context={context}",
                extra={
                    "frameStack": frame_stack_to_json(frame_stack),
                    "frameId": frame.frame_id,
                },
            )
            context.setdefault("version", 0)
            return "", context

        context.setdefault("version", 0)
        return memory_text or "", context

    async def _remember_if_needed(
        self,
        frame: ExecutionFrame,
        memory: Any,
        memory_context: Optional[Dict[str, Any]],
        payload: Any,
        stage: str,
        frame_stack: List[FrameStackEntry],
    ) -> None:
        if not self._memory_available(memory) or memory_context is None:
            return

        if payload is None:
            return

        payload_str = str(payload).strip()
        if not payload_str:
            return

        try:
            await memory.remember(
                memory_context,
                payload_str,
                frame_stack=frame_stack_to_json(frame_stack),
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.context.logger.error(
                f"记忆写入失败[{stage}] frame={frame.frame_id}: {exc} context={memory_context}",
                extra={
                    "frameStack": frame_stack_to_json(frame_stack),
                    "frameId": frame.frame_id,
                },
            )

    async def solve(
        self,
        goal: str,
        think_llm: AsyncThinkLLM,
        eval_llm: AsyncEvalLLM,
        memory: Any = None,
        tools: Any = None,
        frame_stack: Optional[List[FrameStackEntry]] = None,
    ) -> SolveResult:
        """执行异步求解"""
        self.context.start_time = time.time()

        if frame_stack is None:
            raise FrameStackProtocolError("solve_async 调用必须提供 frameStack")

        try:
            # 创建根节点和根帧
            root_node = S(goal=goal)
            root_frame = ExecutionFrame(
                frame_id=str(uuid.uuid4()),
                node=root_node,
                depth=0
            )
            self.frame_stack.append(root_frame)

            engine_stack = frame_stack

            # 执行状态机循环
            result = await self._execute_state_machine(
                think_llm,
                eval_llm,
                memory,
                tools,
                engine_stack,
            )

            return self._create_result(result, SolveStatus.COMPLETED)

        except ConstraintViolationError:
            # 约束终止：不包装，直接向上抛出
            self.context.logger.warning("约束触发，终止执行")
            raise
        except ExecutionFailureError as e:
            # 执行失败：返回FAILED状态
            self.context.logger.error(f"执行失败: {e}")
            return self._create_result(str(e), SolveStatus.FAILED)
        except Exception as e:
            self.context.logger.error(f"执行异常: {e}")
            return self._create_result(f"执行异常: {e}", SolveStatus.FAILED)

    async def _execute_state_machine(
        self,
        think_llm: AsyncThinkLLM,
        eval_llm: AsyncEvalLLM,
        memory: Any,
        tools: Any,
        frame_stack: List[FrameStackEntry],
    ) -> str:
        """执行状态机主循环"""
        max_iterations = 1000  # 防止无限循环

        for iteration in range(max_iterations):
            if not self.frame_stack:
                break

            current_frame = self.frame_stack[-1]

            # 更新最大深度
            self.context.max_depth_reached = max(
                self.context.max_depth_reached,
                current_frame.depth
            )

            # 检查约束
            await self.context.budget_manager.check_constraints_async(
                current_frame.node,
                0  # 临时token消耗
            )

            # 根据当前帧状态构建堆栈条目
            current_stage = "think" if current_frame.state == FrameState.THINK else "eval"
            stack_entry = FrameStackEntry(
                frame_id=current_frame.frame_id,
                depth=current_frame.depth,
                stage=current_stage,
                node_type=identify_node_type(current_frame.node),
            )
            frame_stack_with_entry = append_frame_entry(frame_stack, stack_entry)

            # 执行状态转换
            try:
                new_state = await self._handle_frame_state(
                    current_frame,
                    think_llm,
                    eval_llm,
                    memory,
                    tools,
                    frame_stack_with_entry,
                )
                current_frame.state = new_state

                # 离开当前帧前弹出堆栈条目
                frame_stack = pop_frame_entry(frame_stack_with_entry, current_frame.frame_id)

                # 处理状态转换
                if new_state == FrameState.RETURNING:
                    result = await self._handle_return(current_frame)
                    if self.frame_stack and result:
                        # 还有父帧，将结果传递给父帧
                        parent_frame = self.frame_stack[-1]
                        parent_frame.node.done.append(result)
                        parent_frame.state = FrameState.EVAL
                    elif result:
                        # 根帧完成，直接返回结果
                        return result
                elif new_state == FrameState.FAILED:
                    self._handle_failure(current_frame)

            except Exception as e:
                self.context.logger.error(f"帧处理异常: {e}")
                raise

        if self.frame_stack:
            self.context.logger.warning(f"达到最大迭代次数，剩余帧: {len(self.frame_stack)}")
            raise RuntimeError("执行未在最大迭代限制内完成")

        return "任务完成"

    async def _handle_frame_state(
        self,
        frame: ExecutionFrame,
        think_llm: AsyncThinkLLM,
        eval_llm: AsyncEvalLLM,
        memory: Any,
        tools: Any,
        frame_stack: List[FrameStackEntry],
    ) -> FrameState:
        """处理帧状态"""
        if frame.state == FrameState.THINK:
            return await self._handle_think_state(frame, think_llm, memory, tools, frame_stack)
        elif frame.state == FrameState.PLANNING:
            return await self._handle_planning_state(frame)
        elif frame.state == FrameState.FIRST_EVAL:
            return await self._handle_eval_state(frame, eval_llm, memory, frame_stack, is_first=True)
        elif frame.state == FrameState.EVAL:
            return await self._handle_eval_state(frame, eval_llm, memory, frame_stack, is_first=False)
        else:
            return frame.state

    async def _handle_think_state(
        self,
        frame: ExecutionFrame,
        think_llm: AsyncThinkLLM,
        memory: Any,
        tools: Any,
        frame_stack: List[FrameStackEntry],
    ) -> FrameState:
        """处理THINK状态"""
        self.context.logger.debug(f"Frame {frame.frame_id}: 执行Think操作")

        memory_text, memory_context = await self._prepare_memory_inputs(
            frame,
            memory,
            stage="think",
            frame_stack=frame_stack,
        )

        try:
            think_result = await think_llm(
                frame.node,
                memory_text,
                memory_context,
                tools,
                frame_stack=frame_stack_to_json(frame_stack),
            )

            # 检测是否意外返回了未await的协程
            if asyncio.iscoroutine(think_result):
                raise TypeError(
                    f"Think算子返回了未await的协程。请确保在Think函数中正确使用await。"
                    f"检查think_llm实现是否正确await了异步调用。"
                )

            # 记录token消耗
            tokens_used = think_result.get("tokens_used", 0)
            frame.add_frame_tokens(think_tokens=tokens_used)
            await self.context.token_usage.add_think_async(tokens_used)
            await self.context.budget_manager.consume_budget_async(tokens_delta=tokens_used)

            # 解析Think结果
            action_type = think_result.get("type", "TODO")
            description = think_result.get("description", "")
            remember_payload = think_result.get("remember") if isinstance(think_result, dict) else None

            await self._remember_if_needed(
                frame,
                memory,
                memory_context,
                remember_payload,
                stage="think",
                frame_stack=frame_stack,
            )

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

        except ConstraintViolationError:
            # 约束违反异常直接向上抛出，不转换为FAILED状态
            raise
        except Exception as e:
            self.context.logger.error(
                "Think操作异常",
                extra={
                    "frameStack": frame_stack_to_json(frame_stack),
                    "frameId": frame.frame_id,
                },
            )
            frame.set_failed(f"Think操作失败: {e}")
            return FrameState.FAILED
        finally:
            # 消耗token后立即检查约束，确保最后一步也能触发终止
            # 放在finally中确保无论Think成功还是失败都会检查
            try:
                await self.context.budget_manager.check_constraints_async(frame.node, 0)
            except ConstraintViolationError:
                # 约束违反异常直接向上抛出
                raise

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
                frame.set_failed("没有Think结果可供解析")
                return FrameState.FAILED

        except Exception as e:
            frame.set_failed(f"计划解析失败: {e}")
            return FrameState.FAILED

    async def _handle_eval_state(
        self,
        frame: ExecutionFrame,
        eval_llm: AsyncEvalLLM,
        memory: Any,
        frame_stack: List[FrameStackEntry],
        is_first: bool = False
    ) -> FrameState:
        """处理EVAL状态"""
        self.context.logger.debug(
            f"Frame {frame.frame_id}: 执行{'首次' if is_first else ''}Eval操作"
        )

        memory_text, memory_context = await self._prepare_memory_inputs(
            frame,
            memory,
            stage="eval",
            frame_stack=frame_stack,
        )

        try:
            eval_result = await eval_llm(
                frame.node,
                memory_text,
                memory_context,
                frame_stack=frame_stack_to_json(frame_stack),
            )

            # 检测是否意外返回了未await的协程
            if asyncio.iscoroutine(eval_result):
                raise TypeError(
                    f"Eval算子返回了未await的协程。请确保在Eval函数中正确使用await。"
                    f"检查eval_llm实现是否正确await了异步调用。"
                )

            # 记录token消耗
            tokens_used = eval_result.get("tokens_used", 0)
            frame.add_frame_tokens(eval_tokens=tokens_used)
            await self.context.token_usage.add_eval_async(tokens_used)
            await self.context.budget_manager.consume_budget_async(tokens_delta=tokens_used)

            # 解析Eval结果
            action_type = eval_result.get("type", "RETURN")
            description = eval_result.get("description", "")
            remember_payload = eval_result.get("remember") if isinstance(eval_result, dict) else None

            await self._remember_if_needed(
                frame,
                memory,
                memory_context,
                remember_payload,
                stage="eval",
                frame_stack=frame_stack,
            )

            if action_type == "CALL":
                # 创建子任务帧
                child_node = S(goal=description, parent=frame.node)
                child_frame = ExecutionFrame(
                    frame_id=str(uuid.uuid4()),
                    node=child_node,
                    depth=frame.depth + 1
                )
                self.frame_stack.append(child_frame)
                self.context.logger.info(
                    "创建子Frame",
                    extra={
                        "frameStack": frame_stack_to_json(frame_stack),
                        "childFrameId": child_frame.frame_id,
                    },
                )
                return FrameState.EVAL  # 父帧保持EVAL状态，等待子帧完成

            elif action_type == "RETURN":
                # 完成当前任务
                frame.node.done.append(description)
                return FrameState.RETURNING
            else:
                raise ValueError(f"未知的Eval action type: {action_type}")

        except ConstraintViolationError:
            # 约束违反异常直接向上抛出，不转换为FAILED状态
            raise
        except Exception as e:
            self.context.logger.error(
                "Eval操作异常",
                extra={
                    "frameStack": frame_stack_to_json(frame_stack),
                    "frameId": frame.frame_id,
                },
            )
            frame.set_failed(f"Eval操作失败: {e}")
            return FrameState.FAILED
        finally:
            # 消耗token后立即检查约束，确保最后一步也能触发终止
            # 放在finally中确保无论Eval成功还是失败都会检查
            try:
                await self.context.budget_manager.check_constraints_async(frame.node, 0)
            except ConstraintViolationError:
                # 约束违反异常直接向上抛出
                raise

    async def _handle_return(self, frame: ExecutionFrame) -> Optional[str]:
        """处理RETURNING状态"""
        self.context.logger.debug(f"Frame {frame.frame_id}: 处理返回")

        # 移除当前帧
        if self.frame_stack and self.frame_stack[-1] == frame:
            self.frame_stack.pop()

        # 返回结果
        if frame.node.done:
            return frame.node.done[-1]
        else:
            return f"任务完成: {frame.node.goal}"

    def _handle_failure(self, frame: ExecutionFrame) -> None:
        """处理失败状态"""
        failure_message = frame.failure_reason or f"任务失败: {frame.node.goal}"
        self.context.logger.error(f"Frame {frame.frame_id}: 执行失败 - {failure_message}")
        raise ExecutionFailureError(failure_message, frame)

    def _create_result(self, result: str, status: SolveStatus) -> SolveResult:
        """创建求解结果"""
        execution_time = time.time() - self.context.start_time

        # 转换token统计格式
        token_usage = TokenUsage(
            total=self.context.token_usage.total,
            think_calls=self.context.token_usage.think_calls,
            eval_calls=self.context.token_usage.eval_calls,
            think_tokens=self.context.token_usage.think_tokens,
            eval_tokens=self.context.token_usage.eval_tokens
        )

        return SolveResult(
            status=status,
            result=result,
            token_usage=token_usage,
            execution_time=execution_time,
            max_depth_reached=self.context.max_depth_reached
        )


async def solve_async(
    goal: str,
    think_llm: Union[AsyncThinkLLM, Any],
    eval_llm: Union[AsyncEvalLLM, Any],
    budget: Optional[ExecutionBudget] = None,
    logger: Optional[UnifiedLogger] = None,
    memory: Any = None,
    tools: Any = None,
    frame_stack: Optional[List[FrameStackEntry]] = None,
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
        frame_stack: 现有执行堆栈（只读数组）

    Returns:
        SolveResult: 求解结果

    Raises:
        FrameStackProtocolError: 当未提供 frame_stack 或协议校验失败时抛出
    """
    # 设置默认参数
    if budget is None:
        budget = ExecutionBudget()
    if logger is None:
        logger = UnifiedLogger()

    if memory is not None:
        missing = [attr for attr in ("recall", "remember") if not hasattr(memory, attr)]
        if missing:
            raise TypeError(
                "memory implementation must expose async recall() and remember() methods"
            )

    if frame_stack is None:
        raise FrameStackProtocolError("solve_async 需要提供 frame_stack 参数")
    validate_frame_stack(frame_stack)

    # 转换为异步接口
    async_think = create_async_think(think_llm)
    async_eval = create_async_eval(eval_llm)

    # 创建执行上下文
    context = AsyncExecutionContext(
        session_id=str(uuid.uuid4()),
        budget_manager=BudgetManager(budget),
        token_usage=UnifiedTokenUsage(),
        logger=logger
    )

    # 创建引擎并执行
    engine = AsyncExecutionEngine(context)
    return await engine.solve(goal, async_think, async_eval, memory, tools, frame_stack)
