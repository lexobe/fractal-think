"""
示例Mock算子 - 用于演示和测试
"""

import asyncio
import time
from typing import Dict, Any, Optional

from ..types import S


class MockThinkLLM:
    """简单的Mock Think算子"""

    def __init__(self, simulation_delay: float = 0.1, verbose: bool = False):
        self.simulation_delay = simulation_delay
        self.verbose = verbose
        self.call_count = 0

    def __call__(self, node: S, memory: Any = None, tools: Any = None) -> Dict[str, Any]:
        self.call_count += 1
        time.sleep(self.simulation_delay)

        if self.verbose:
            print(f"[Think] 目标: {node.goal}")

        # 简单逻辑：如果目标很简单就直接返回，否则分解
        if len(node.goal) < 20 or node.level >= 3:
            return {
                "type": "RETURN",
                "description": f"已完成: {node.goal}",
                "tokens_used": 50
            }
        else:
            return {
                "type": "TODO",
                "description": f"[] 步骤1：分析{node.goal}\n[] 步骤2：实现{node.goal}\n[] 步骤3：总结{node.goal}",
                "tokens_used": 100
            }


class MockEvalLLM:
    """简单的Mock Eval算子"""

    def __init__(self, simulation_delay: float = 0.05, verbose: bool = False):
        self.simulation_delay = simulation_delay
        self.verbose = verbose
        self.call_count = 0

    def __call__(self, node: S, memory: Any = None) -> Dict[str, Any]:
        self.call_count += 1
        time.sleep(self.simulation_delay)

        if self.verbose:
            print(f"[Eval] 评估: {node.goal}")

        # 简单逻辑：随机决定是否需要更多子任务
        if node.level >= 2 or len(node.done) > 0:
            return {
                "type": "RETURN",
                "description": f"完成评估: {node.goal}",
                "tokens_used": 30
            }
        else:
            return {
                "type": "CALL",
                "description": f"子任务：详细分析{node.goal}",
                "tokens_used": 60
            }


class AsyncMockThinkLLM:
    """异步Mock Think算子"""

    def __init__(self, simulation_delay: float = 0.1, verbose: bool = False):
        self.simulation_delay = simulation_delay
        self.verbose = verbose
        self.call_count = 0

    async def __call__(self, node: S, memory: Any = None, tools: Any = None) -> Dict[str, Any]:
        self.call_count += 1
        await asyncio.sleep(self.simulation_delay)

        if self.verbose:
            print(f"[AsyncThink] 目标: {node.goal}")

        # 简单逻辑：如果目标很简单就直接返回，否则分解
        if len(node.goal) < 20 or node.level >= 3:
            return {
                "type": "RETURN",
                "description": f"已完成: {node.goal}",
                "tokens_used": 50
            }
        else:
            return {
                "type": "TODO",
                "description": f"[] 步骤1：分析{node.goal}\n[] 步骤2：实现{node.goal}\n[] 步骤3：总结{node.goal}",
                "tokens_used": 100
            }


class AsyncMockEvalLLM:
    """异步Mock Eval算子"""

    def __init__(self, simulation_delay: float = 0.05, verbose: bool = False):
        self.simulation_delay = simulation_delay
        self.verbose = verbose
        self.call_count = 0

    async def __call__(self, node: S, memory: Any = None) -> Dict[str, Any]:
        self.call_count += 1
        await asyncio.sleep(self.simulation_delay)

        if self.verbose:
            print(f"[AsyncEval] 评估: {node.goal}")

        # 简单逻辑：随机决定是否需要更多子任务
        if node.level >= 2 or len(node.done) > 0:
            return {
                "type": "RETURN",
                "description": f"完成评估: {node.goal}",
                "tokens_used": 30
            }
        else:
            return {
                "type": "CALL",
                "description": f"子任务：详细分析{node.goal}",
                "tokens_used": 60
            }


class DetailedAIArtThink:
    """详细的AI与艺术Think算子 - 复现规范示例"""

    def __init__(self, simulation_delay: float = 0.3, verbose: bool = True):
        self.simulation_delay = simulation_delay
        self.verbose = verbose
        self.call_count = 0

    async def __call__(self, node: S, memory: Any = None, tools: Any = None) -> Dict[str, Any]:
        self.call_count += 1
        await asyncio.sleep(self.simulation_delay)

        if self.verbose:
            print(f"[DetailedThink-{self.call_count}] 目标: {node.goal[:50]}...")

        goal_lower = node.goal.lower()

        # 根据不同的目标给出不同的分解策略
        if "ai与艺术" in goal_lower or "ai和艺术" in goal_lower:
            return {
                "type": "TODO",
                "description": """[] 研究AI在艺术创作中的应用现状
[] 分析传统艺术与AI艺术的关系
[] 探讨AI艺术的创新价值和局限性""",
                "tokens_used": 80
            }
        elif "研究" in goal_lower and "应用现状" in goal_lower:
            return {
                "type": "RETURN",
                "description": "AI在艺术创作中的应用现状：从早期的算法艺术到现代的深度学习生成模型，AI技术正在革命性地改变艺术创作的方式。目前主要应用包括图像生成、音乐创作、诗歌写作等领域，工具如DALL-E、Midjourney、GPT等已被广泛使用。",
                "tokens_used": 50
            }
        elif "分析" in goal_lower and "关系" in goal_lower:
            return {
                "type": "RETURN",
                "description": "传统艺术与AI艺术的关系分析：AI艺术并非要取代传统艺术，而是为艺术创作提供了新的工具和可能性。传统艺术强调人类情感表达和技法传承，AI艺术则能够快速生成、无限变化，两者相互补充，共同拓展了艺术表达的边界。",
                "tokens_used": 55
            }
        elif "探讨" in goal_lower and ("价值" in goal_lower or "局限" in goal_lower):
            return {
                "type": "RETURN",
                "description": "AI艺术的创新价值和局限性：价值在于降低创作门槛、提供灵感启发、实现大规模个性化创作；局限性包括缺乏深层情感理解、存在训练数据偏见、可能削弱人类创造力等。关键是找到人机协作的平衡点。",
                "tokens_used": 60
            }
        else:
            # 默认策略：简单分解或直接完成
            if node.level >= 2:
                return {
                    "type": "RETURN",
                    "description": f"已完成任务：{node.goal}",
                    "tokens_used": 30
                }
            else:
                return {
                    "type": "TODO",
                    "description": f"[] 分析任务：{node.goal}\n[] 执行核心工作\n[] 整理输出结果",
                    "tokens_used": 40
                }


class DetailedAIArtEval:
    """详细的AI与艺术Eval算子 - 复现规范示例"""

    def __init__(self, simulation_delay: float = 0.2, verbose: bool = True):
        self.simulation_delay = simulation_delay
        self.verbose = verbose
        self.call_count = 0

    async def __call__(self, node: S, memory: Any = None) -> Dict[str, Any]:
        self.call_count += 1
        await asyncio.sleep(self.simulation_delay)

        if self.verbose:
            print(f"[DetailedEval-{self.call_count}] 评估: {node.goal[:50]}...")

        # 检查是否有子任务完成结果
        if node.done:
            if self.verbose:
                print(f"[DetailedEval] 发现 {len(node.done)} 个完成项")

        goal_lower = node.goal.lower()

        # 根据不同情况决定是否需要进一步分解
        if ("ai与艺术" in goal_lower or "ai和艺术" in goal_lower) and len(node.done) >= 3:
            # 主任务有3个子任务完成，可以整合
            if self.verbose:
                print(f"[DetailedEval] 整合文章，已完成{len(node.done)}项")
            final_content = self._integrate_ai_art_content(node.done)
            return {
                "type": "RETURN",
                "description": final_content,
                "tokens_used": 80
            }
        elif len(node.done) > 0 and node.level >= 1:
            # 子任务层级，有完成项就返回
            return {
                "type": "RETURN",
                "description": f"子任务完成：{node.done[-1]}",
                "tokens_used": 25
            }
        elif node.level >= 2:
            # 深度限制，直接返回
            return {
                "type": "RETURN",
                "description": f"达到深度限制，返回当前结果：{node.goal}",
                "tokens_used": 20
            }
        else:
            # 需要更深入的分析
            return {
                "type": "CALL",
                "description": f"深入分析：{node.goal}的具体实现方案",
                "tokens_used": 35
            }

    def _integrate_ai_art_content(self, done_items):
        """整合AI与艺术文章内容"""
        content = """# AI与艺术：技术与创意的完美融合

随着人工智能技术的飞速发展，AI在艺术创作领域的应用日益广泛，正在深刻改变着我们对艺术创作的理解和实践。

## AI艺术的现状与发展

""" + done_items[0] + """

这些工具的出现，不仅提高了艺术创作的效率，也为艺术家提供了前所未有的创作可能性。

## 传统艺术与AI艺术的对话

""" + done_items[1] + """

这种互补关系正在重新定义艺术创作的本质和意义。

## 价值与挑战并存

""" + done_items[2] + """

因此，我们需要以开放而谨慎的态度对待AI艺术的发展。

## 未来展望

随着技术的不断进步，AI与艺术的结合将更加深入。未来的艺术创作可能是一个人机协作的过程，艺术家利用AI工具拓展创意边界，同时保持人类独有的情感表达和审美判断。

AI与艺术的融合不是终点，而是一个新的起点。它开启了艺术创作的新篇章，让我们有理由期待更加丰富多彩的艺术未来。

(全文约900字)"""

        return content