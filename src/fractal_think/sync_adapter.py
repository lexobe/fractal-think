"""
同步适配器 - 向后兼容支持

提供与原有同步API完全兼容的接口，内部使用异步引擎：
- solve() 函数兼容性
- start_solve() 入口点兼容性
- 自动async/sync转换
- 透明的性能优化
"""

import asyncio
import threading
from typing import Optional, Any, Union, List
import logging

from .engine import solve_async
from .interfaces import create_async_think, create_async_eval, ThinkLLM, EvalLLM
from .common import ExecutionBudget, UnifiedLogger, ExecutionMode, UnifiedTokenUsage, BudgetManager, convert_legacy_constraints
from .types import S, SolveResult, SolveStatus, TokenUsage


class SyncAsyncAdapter:
    """同步到异步的适配器 - 处理事件循环管理"""

    def __init__(self):
        self._loop = None
        self._thread = None
        self._lock = threading.Lock()

    def run_async_in_sync(self, coro):
        """在同步上下文中运行异步代码"""
        try:
            # 尝试获取当前事件循环
            loop = asyncio.get_running_loop()
            # 如果已有事件循环在运行，在线程中运行
            return self._run_in_thread(coro)
        except RuntimeError:
            # 没有运行的事件循环，直接运行
            return asyncio.run(coro)

    def _run_in_thread(self, coro):
        """在独立线程中运行异步代码"""
        with self._lock:
            if self._loop is None or self._loop.is_closed():
                self._start_event_loop_thread()

            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            return future.result()

    def _start_event_loop_thread(self):
        """启动事件循环线程"""
        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()

        # 等待循环启动
        import time
        while self._loop is None:
            time.sleep(0.001)


# 全局适配器实例
_sync_adapter = SyncAsyncAdapter()


def solve_with_async_engine(
    goal: str,
    think_llm: ThinkLLM,
    eval_llm: EvalLLM,
    budget: Optional[Union[ExecutionBudget, Any]] = None,
    logger: Optional[logging.Logger] = None,
    memory: Any = None,
    tools: Any = None
) -> SolveResult:
    """使用异步引擎的同步solve实现 - 完全向后兼容"""

    async def _async_solve():
        # 兼容性处理：将旧的Constraints转换为ExecutionBudget
        if budget is not None and not isinstance(budget, ExecutionBudget):
            converted_budget = convert_legacy_constraints(budget)
        else:
            converted_budget = budget

        # 创建统一日志器
        unified_logger = UnifiedLogger(logger, ExecutionMode.SYNC) if logger else UnifiedLogger()

        # 转换算子接口：同步 -> 异步
        async_think_llm = create_async_think(think_llm)
        async_eval_llm = create_async_eval(eval_llm)

        return await solve_async(
            goal=goal,
            think_llm=async_think_llm,
            eval_llm=async_eval_llm,
            budget=converted_budget,
            logger=unified_logger,
            memory=memory,
            tools=tools
        )

    # 在同步上下文中运行异步代码
    return _sync_adapter.run_async_in_sync(_async_solve())


def enhanced_solve(
    node: S,
    think_llm: ThinkLLM,
    eval_llm: EvalLLM,
    constraints: Optional[Union[ExecutionBudget, Any]] = None,
    logger: Optional[logging.Logger] = None,
    token_usage: Optional[TokenUsage] = None
) -> SolveResult:
    """增强的solve函数 - 兼容原_solve_internal签名"""

    # 参数转换和兼容性处理
    if constraints is not None and not isinstance(constraints, ExecutionBudget):
        budget = convert_legacy_constraints(constraints)
    else:
        budget = constraints or ExecutionBudget()
    goal = node.goal

    async def _async_enhanced_solve():
        # 创建异步算子
        async_think = create_async_think(think_llm)
        async_eval = create_async_eval(eval_llm)

        # 创建统一组件
        unified_logger = UnifiedLogger(logger, ExecutionMode.ASYNC) if logger else UnifiedLogger()
        unified_token_usage = UnifiedTokenUsage()

        # 如果传入了token_usage，合并初始状态
        if token_usage:
            unified_token_usage.think_calls = token_usage.think_calls
            unified_token_usage.eval_calls = token_usage.eval_calls
            unified_token_usage.think_tokens = token_usage.think_tokens
            unified_token_usage.eval_tokens = token_usage.eval_tokens

        # 如果node已有parent关系，需要特殊处理
        if node.parent:
            # 构建完整的节点层次结构
            ancestors = []
            current = node
            while current:
                ancestors.append(current)
                current = current.parent
            ancestors.reverse()

            # 从根节点开始恢复
            root_goal = ancestors[0].goal
            result = await solve_async(
                goal=root_goal,
                think_llm=async_think,
                eval_llm=async_eval,
                budget=budget,
                logger=unified_logger
            )

            # TODO: 实现更复杂的中间状态恢复逻辑
            return result
        else:
            # 简单情况：直接求解
            return await solve_async(
                goal=goal,
                think_llm=async_think,
                eval_llm=async_eval,
                budget=budget,
                logger=unified_logger
            )

    return _sync_adapter.run_async_in_sync(_async_enhanced_solve())


class LegacyCompatibilityLayer:
    """遗留兼容性层 - 提供与原thinkon_core完全一致的API"""

    @staticmethod
    def solve(
        goal: str,
        think_llm: ThinkLLM,
        eval_llm: EvalLLM,
        constraints: Optional[Union[ExecutionBudget, Any]] = None,
        logger: Optional[logging.Logger] = None
    ) -> str:
        """原solve函数的兼容实现 - 返回字符串结果"""
        result = solve_with_async_engine(
            goal=goal,
            think_llm=think_llm,
            eval_llm=eval_llm,
            budget=constraints,
            logger=logger
        )
        return result.result

    @staticmethod
    def solve_with_meta(
        goal: str,
        think_llm: ThinkLLM,
        eval_llm: EvalLLM,
        constraints: Optional[Union[ExecutionBudget, Any]] = None,
        logger: Optional[logging.Logger] = None
    ) -> SolveResult:
        """原solve_with_meta函数的兼容实现"""
        return solve_with_async_engine(
            goal=goal,
            think_llm=think_llm,
            eval_llm=eval_llm,
            budget=constraints,
            logger=logger
        )

    @staticmethod
    def start_solve(
        goal: str,
        think_llm: ThinkLLM,
        eval_llm: EvalLLM,
        max_depth: int = 5,
        max_tokens: int = 5000,
        max_time: float = 30.0,
        logger: Optional[logging.Logger] = None
    ) -> SolveResult:
        """原start_solve函数的兼容实现"""
        budget = ExecutionBudget(
            max_depth=max_depth,
            max_tokens=max_tokens,
            max_time=max_time
        )

        return solve_with_async_engine(
            goal=goal,
            think_llm=think_llm,
            eval_llm=eval_llm,
            budget=budget,
            logger=logger
        )

    @staticmethod
    def start_solve_simple(
        goal: str,
        think_llm: ThinkLLM,
        eval_llm: EvalLLM,
        logger: Optional[logging.Logger] = None
    ) -> str:
        """原start_solve_simple函数的兼容实现"""
        result = LegacyCompatibilityLayer.start_solve(
            goal=goal,
            think_llm=think_llm,
            eval_llm=eval_llm,
            logger=logger
        )
        return result.result


class AsyncOptimizedLayer:
    """异步优化层 - 提供性能优化的新API"""

    @staticmethod
    async def solve_async_optimized(
        goal: str,
        think_llm: Union[ThinkLLM, 'AsyncThinkLLM'],
        eval_llm: Union[EvalLLM, 'AsyncEvalLLM'],
        budget: Optional[ExecutionBudget] = None,
        logger: Optional[logging.Logger] = None,
        memory: Any = None,
        tools: Any = None,
        snapshot_callback: Optional[callable] = None,
        auto_snapshot: bool = False
    ) -> SolveResult:
        """异步优化的solve - 支持快照和恢复"""

        unified_logger = UnifiedLogger(logger, ExecutionMode.ASYNC) if logger else UnifiedLogger()

        result = await solve_async(
            goal=goal,
            think_llm=think_llm,
            eval_llm=eval_llm,
            budget=budget,
            logger=unified_logger,
            memory=memory,
            tools=tools
        )

        return result

    @staticmethod
    async def solve_batch_async(
        goals: List[str],
        think_llm: Union[ThinkLLM, 'AsyncThinkLLM'],
        eval_llm: Union[EvalLLM, 'AsyncEvalLLM'],
        budget_per_task: Optional[ExecutionBudget] = None,
        max_concurrent: int = 3,
        logger: Optional[logging.Logger] = None
    ) -> List[SolveResult]:
        """批量异步求解 - 并发执行多个任务"""

        semaphore = asyncio.Semaphore(max_concurrent)

        async def solve_single(goal: str) -> SolveResult:
            async with semaphore:
                return await AsyncOptimizedLayer.solve_async_optimized(
                    goal=goal,
                    think_llm=think_llm,
                    eval_llm=eval_llm,
                    budget=budget_per_task,
                    logger=logger
                )

        # 并发执行所有任务
        tasks = [solve_single(goal) for goal in goals]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(SolveResult(
                    result=f"批量任务{i}执行异常: {str(result)}",
                    status=SolveStatus.FAILED,
                    token_usage=TokenUsage(total=0, think_calls=0, eval_calls=0, think_tokens=0, eval_tokens=0),
                    execution_time=0.0,
                    max_depth_reached=0,
                    constraint_triggered=None
                ))
            else:
                processed_results.append(result)

        return processed_results


# 默认导出兼容层
solve = LegacyCompatibilityLayer.solve
solve_with_meta = LegacyCompatibilityLayer.solve_with_meta
start_solve = LegacyCompatibilityLayer.start_solve
start_solve_simple = LegacyCompatibilityLayer.start_solve_simple

# 导出优化层
solve_async_optimized = AsyncOptimizedLayer.solve_async_optimized
solve_batch_async = AsyncOptimizedLayer.solve_batch_async