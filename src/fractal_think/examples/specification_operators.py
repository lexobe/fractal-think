"""
规范版算子 - 严格按照thinkon.md规范实现AI与艺术示例
"""

import asyncio
from typing import Dict, Any, Optional

from ..types import S


class SpecificationAIArtThink:
    """规范版AI与艺术Think算子 - 严格按照thinkon.md实现"""

    def __init__(self, simulation_delay: float = 0.1, verbose: bool = True):
        self.simulation_delay = simulation_delay
        self.verbose = verbose
        self.call_count = 0

    async def __call__(
        self,
        node: S,
        memory_text: str = "",
        memory_context: Optional[Dict[str, Any]] = None,
        tools: Any = None,
        frame_stack: Optional[list] = None,
    ) -> Dict[str, Any]:
        self.call_count += 1
        await asyncio.sleep(self.simulation_delay)

        if self.verbose:
            print(f"[SpecThink-{self.call_count}] 分析: {node.goal[:30]}...")

        goal_lower = node.goal.lower()

        # 主任务：写AI与艺术短文 - 完全按照thinkon.md section 4.1
        if "ai与艺术" in goal_lower and "短文" in goal_lower:
            return {
                "type": "TODO",
                "description": """分成三个段落撰写：
 [] 生成艺术段落
 [] 辅助创作段落
 [] 艺术评论段落""",
                "tokens_used": 60,
                "remember": None,
            }

        # 第一段：生成艺术段落 - 完全按照thinkon.md section 4.1
        elif "生成艺术段落" in node.goal:
            return {
                "type": "RETURN",
                "description": "生成艺术段落已经写完，含DALL·E例，并符合要求。",
                "tokens_used": 45,
                "remember": None,
            }

        # 第二段：辅助创作段落 - 完全按照thinkon.md规范原文
        elif "辅助创作段落" in node.goal:
            return {
                "type": "RETURN",
                "description": "辅助创作段落已经写完，包含了一个能打动人的事例，符合要求。",
                "tokens_used": 50,
                "remember": None,
            }

        # 第三段：艺术评论段落 - 完全按照thinkon.md规范原文
        elif "艺术评论段落" in node.goal:
            return {
                "type": "RETURN",
                "description": "艺术评论段落已经写完，提供了深入的分析和见解，符合要求。",
                "tokens_used": 55,
                "remember": None,
            }

        # 其他情况：直接完成，不进入递归分支
        else:
            return {
                "type": "RETURN",
                "description": f"任务完成: {node.goal}",
                "tokens_used": 25,
                "remember": None,
            }


class SpecificationAIArtEval:
    """规范版AI与艺术Eval算子 - 严格按照thinkon.md实现"""

    def __init__(self, simulation_delay: float = 0.1, verbose: bool = True):
        self.simulation_delay = simulation_delay
        self.verbose = verbose
        self.call_count = 0

    async def __call__(
        self,
        node: S,
        memory_text: str = "",
        memory_context: Optional[Dict[str, Any]] = None,
        frame_stack: Optional[list] = None,
    ) -> Dict[str, Any]:
        self.call_count += 1
        await asyncio.sleep(self.simulation_delay)

        if self.verbose:
            print(f"[SpecEval-{self.call_count}] 评估: {node.goal[:30]}...")

        goal_lower = node.goal.lower()

        # 检查是否有完成的子任务
        if node.done and self.verbose:
            print(f"[SpecEval] 已完成 {len(node.done)} 个子任务")

        # 主任务：AI与艺术短文
        if ("ai与艺术" in goal_lower or "ai和艺术" in goal_lower) and "短文" in goal_lower:
            if len(node.done) >= 3:
                # 三个段落都完成，整合文章
                if self.verbose:
                    print(f"[SpecEval] 整合最终文章")
                final_article = self._create_final_article(node.done)
                return {
                    "type": "RETURN",
                    "description": final_article,
                    "tokens_used": 70,
                    "remember": None,
                }
            else:
                # 还需要更多段落，继续处理下一个子任务
                return {
                    "type": "CALL",
                    "description": self._get_next_subtask(len(node.done)),
                    "tokens_used": 30,
                    "remember": None,
                }

        # 子任务：直接返回完成
        else:
            return {
                "type": "RETURN",
                "description": f"子任务完成: {node.goal}",
                "tokens_used": 20,
                "remember": None,
            }

    def _get_next_subtask(self, completed_count: int) -> str:
        """获取下一个子任务描述 - 完全按照thinkon.md section 4.1"""
        subtasks = [
            "生成艺术段落：含DALL·E例，要求不少于100字，且能引起兴趣",
            "辅助创作段落：写一个能打动人的事例",
            "艺术评论段落"
        ]

        if completed_count < len(subtasks):
            return subtasks[completed_count]
        else:
            return "完成文章写作"

    def _create_final_article(self, done_items):
        """创建最终文章 - 完全按照thinkon.md section 4.1 规范"""
        # 按照规范，最终Eval应该返回完整文档，这里组合三个段落的完成状态
        # 严格按照thinkon.md文档，只返回完成状态的组合，不添加自定义内容
        article = f"""AI与艺术短文完成状态：

{done_items[0]}

{done_items[1]}

{done_items[2]}

完整文档已根据计划生成完毕。"""

        return article
