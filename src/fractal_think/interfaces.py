"""
异步算子接口定义
"""

import asyncio
from typing import Protocol, runtime_checkable, Any, Dict, Union, Optional, List

from .types import S

@runtime_checkable
class ThinkLLM(Protocol):
    """同步Think算子协议"""
    def __call__(
        self,
        node: S,
        memory_text: str = "",
        memory_context: Optional[Dict[str, Any]] = None,
        tools: Any = None,
        frame_stack: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        ...

@runtime_checkable
class EvalLLM(Protocol):
    """同步Eval算子协议"""
    def __call__(
        self,
        node: S,
        memory_text: str = "",
        memory_context: Optional[Dict[str, Any]] = None,
        frame_stack: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        ...


@runtime_checkable
class AsyncThinkLLM(Protocol):
    """异步Think算子协议"""

    async def __call__(
        self,
        node: S,
        memory_text: str = "",
        memory_context: Optional[Dict[str, Any]] = None,
        tools: Any = None,
        frame_stack: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        异步Think调用

        Returns:
            Dict包含: {
                "type": "TODO"|"RETURN",
                "description": str,
                "tokens_used": int,
                "remember": Optional[str],  # 非空字符串触发记忆写入
            }
        """
        ...


@runtime_checkable
class AsyncEvalLLM(Protocol):
    """异步Eval算子协议"""

    async def __call__(
        self,
        node: S,
        memory_text: str = "",
        memory_context: Optional[Dict[str, Any]] = None,
        frame_stack: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        异步Eval调用

        Returns:
            Dict包含: {
                "type": "CALL"|"RETURN",
                "description": str,
                "tokens_used": int,
                "remember": Optional[str],  # 非空字符串触发记忆写入
            }
        """
        ...


class SyncToAsyncAdapter:
    """同步算子到异步算子的适配器"""

    def __init__(self, sync_llm: Union[ThinkLLM, EvalLLM]):
        self.sync_llm = sync_llm

    async def __call__(
        self,
        node: S,
        memory_text: str = "",
        memory_context: Optional[Dict[str, Any]] = None,
        tools: Any = None,
        frame_stack: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """将同步调用包装为异步"""
        # 在线程池中执行同步调用，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        if hasattr(self.sync_llm, '__call__'):
            if tools is not None:
                # Think算子
                result = await loop.run_in_executor(
                    None,
                    lambda: self.sync_llm(
                        node,
                        memory_text,
                        memory_context,
                        tools,
                        frame_stack,
                    ),
                )
            else:
                # Eval算子
                result = await loop.run_in_executor(
                    None,
                    lambda: self.sync_llm(
                        node,
                        memory_text,
                        memory_context,
                        frame_stack,
                    ),
                )
            return result
        else:
            raise TypeError(f"Unsupported LLM type: {type(self.sync_llm)}")


class AsyncThinkAdapter(SyncToAsyncAdapter):
    """同步Think算子适配器"""

    def __init__(self, sync_think: ThinkLLM):
        super().__init__(sync_think)

    async def __call__(
        self,
        node: S,
        memory_text: str = "",
        memory_context: Optional[Dict[str, Any]] = None,
        tools: Any = None,
        frame_stack: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return await super().__call__(node, memory_text, memory_context, tools, frame_stack)


class AsyncEvalAdapter(SyncToAsyncAdapter):
    """同步Eval算子适配器"""

    def __init__(self, sync_eval: EvalLLM):
        super().__init__(sync_eval)

    async def __call__(
        self,
        node: S,
        memory_text: str = "",
        memory_context: Optional[Dict[str, Any]] = None,
        frame_stack: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return await super().__call__(node, memory_text, memory_context, None, frame_stack)


# 工厂函数
def create_async_think(sync_or_async_think: Union[ThinkLLM, AsyncThinkLLM]) -> AsyncThinkLLM:
    """创建异步Think算子"""
    # 首先检查函数本身是否是协程函数（async def函数）
    if asyncio.iscoroutinefunction(sync_or_async_think):
        return sync_or_async_think
    # 然后检查是否是对象的__call__方法是协程函数
    elif hasattr(sync_or_async_think, '__call__') and asyncio.iscoroutinefunction(sync_or_async_think.__call__):
        return sync_or_async_think
    else:
        return AsyncThinkAdapter(sync_or_async_think)


def create_async_eval(sync_or_async_eval: Union[EvalLLM, AsyncEvalLLM]) -> AsyncEvalLLM:
    """创建异步Eval算子"""
    # 首先检查函数本身是否是协程函数（async def函数）
    if asyncio.iscoroutinefunction(sync_or_async_eval):
        return sync_or_async_eval
    # 然后检查是否是对象的__call__方法是协程函数
    elif hasattr(sync_or_async_eval, '__call__') and asyncio.iscoroutinefunction(sync_or_async_eval.__call__):
        return sync_or_async_eval
    else:
        return AsyncEvalAdapter(sync_or_async_eval)
