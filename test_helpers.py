"""
共享的测试辅助模块

提供用于规范示例和单元测试的通用 Think/Eval 算子实现，
避免在多个文件中重复相同的逻辑。
"""

import asyncio
from typing import Dict, Any, List
from thinkon_core import S


class AIArtAsyncThinkBase:
    """AI与艺术异步Think算子基类"""

    def __init__(self, simulation_delay: float = 0.01, verbose: bool = False):
        self.simulation_delay = simulation_delay
        self.call_count = 0
        self.call_history = []
        self.verbose = verbose

    async def __call__(self, node: S, memory=None, tools=None) -> Dict[str, Any]:
        self.call_count += 1
        goal = node.goal
        self.call_history.append(f"Think#{self.call_count}: {goal}")

        if self.verbose:
            print(f"[Think #{self.call_count}] 处理节点: {goal}")

        # 模拟异步思考延时
        await asyncio.sleep(self.simulation_delay)

        # 根节点: 返回三段落计划
        if "AI与艺术" in goal and "短文" in goal:
            plan = """分成三个段落撰写：
 [] 生成艺术段落
 [] 辅助创作段落
 [] 艺术评论段落"""

            result = {
                "type": "TODO",
                "description": plan,
                "tokens_used": 150
            }
            if self.verbose:
                print(f"[Think #{self.call_count}] 制定计划: {plan}")
            return result

        # 各段落的具体实现由子类提供
        elif "生成艺术段落" in goal:
            return self._get_art_paragraph_result()
        elif "辅助创作段落" in goal:
            return self._get_creation_paragraph_result()
        elif "艺术评论段落" in goal:
            return self._get_review_paragraph_result()
        else:
            # 默认处理
            result = {
                "type": "RETURN",
                "description": f"默认完成: {goal}",
                "tokens_used": 80
            }
            if self.verbose:
                print(f"[Think #{self.call_count}] 默认完成: {goal}")
            return result

    def _get_art_paragraph_result(self) -> Dict[str, Any]:
        """子类重写：生成艺术段落"""
        raise NotImplementedError

    def _get_creation_paragraph_result(self) -> Dict[str, Any]:
        """子类重写：辅助创作段落"""
        raise NotImplementedError

    def _get_review_paragraph_result(self) -> Dict[str, Any]:
        """子类重写：艺术评论段落"""
        raise NotImplementedError


class AIArtAsyncEvalBase:
    """AI与艺术异步Eval算子基类"""

    def __init__(self, simulation_delay: float = 0.01, verbose: bool = False):
        self.simulation_delay = simulation_delay
        self.call_count = 0
        self.call_history = []
        self.verbose = verbose

    async def __call__(self, node: S, memory=None) -> Dict[str, Any]:
        self.call_count += 1
        goal = node.goal
        done_count = len(node.done)
        self.call_history.append(f"Eval#{self.call_count}: {goal} (done={done_count})")

        if self.verbose:
            print(f"[Eval #{self.call_count}] 评估节点: {goal}")
            print(f"[Eval #{self.call_count}] 当前done: {done_count} 项")

        # 模拟异步评估延时
        await asyncio.sleep(self.simulation_delay)

        # 根节点的评估逻辑
        if "AI与艺术" in goal and "短文" in goal:
            if done_count == 0:
                result = {
                    "type": "CALL",
                    "description": "生成艺术段落：含DALL·E例，要求不少于100字，且能引起兴趣",
                    "tokens_used": 70
                }
                if self.verbose:
                    print(f"[Eval #{self.call_count}] 首启阶段 - 创建第一个子任务")
                return result

            elif done_count == 1:
                result = {
                    "type": "CALL",
                    "description": "辅助创作段落：写一个能打动人的事例",
                    "tokens_used": 65
                }
                if self.verbose:
                    print(f"[Eval #{self.call_count}] 续步阶段 - 创建第二个子任务")
                return result

            elif done_count == 2:
                result = {
                    "type": "CALL",
                    "description": "艺术评论段落：思考AI艺术的哲学意义",
                    "tokens_used": 68
                }
                if self.verbose:
                    print(f"[Eval #{self.call_count}] 续步阶段 - 创建第三个子任务")
                return result

            elif done_count >= 3:
                # 完整文章由子类实现
                return self._get_final_article_result(node)

        # 子任务默认直接返回
        else:
            result = {
                "type": "RETURN",
                "description": f"子任务完成: {goal}",
                "tokens_used": 60
            }
            if self.verbose:
                print(f"[Eval #{self.call_count}] 子任务完成")
            return result

    def _get_final_article_result(self, node: S) -> Dict[str, Any]:
        """子类重写：生成最终文章"""
        raise NotImplementedError


# 详细版本（用于示例演示）
class DetailedAIArtThink(AIArtAsyncThinkBase):
    """详细版AI与艺术Think算子 - 用于演示"""

    def _get_art_paragraph_result(self) -> Dict[str, Any]:
        content = """AI正在革命性地改变艺术创作领域。以DALL·E为代表的AI绘画工具，能够根据文字描述生成令人惊叹的视觉作品，让艺术创作不再局限于传统的画笔和颜料。这种技术突破了想象力的边界，使得每个人都能成为视觉艺术的创造者，无论是否拥有传统绘画技能。AI生成的作品常常展现出人类艺术家难以企及的奇异美感和创新视角。"""

        result = {
            "type": "RETURN",
            "description": content,
            "tokens_used": 120
        }
        if self.verbose:
            print(f"[Think #{self.call_count}] 生成艺术段落内容")
        return result

    def _get_creation_paragraph_result(self) -> Dict[str, Any]:
        content = """在音乐创作领域，AI展现出了同样令人瞩目的能力。作曲家David Cope开发的EMI系统能够模仿巴赫、莫扎特等大师的风格创作音乐，其作品曾在音乐会上演奏时让观众误以为是真正的古典杰作。这个例子生动地展示了AI如何成为艺术家的得力助手，不是要取代人类创造力，而是拓展和增强我们的艺术表达能力。"""

        result = {
            "type": "RETURN",
            "description": content,
            "tokens_used": 130
        }
        if self.verbose:
            print(f"[Think #{self.call_count}] 生成辅助创作段落内容")
        return result

    def _get_review_paragraph_result(self) -> Dict[str, Any]:
        content = """然而，AI艺术的兴起也引发了深刻的哲学思考。什么是真正的艺术？创造力是否是人类独有的特质？当机器能够创作出感动人心的作品时，我们需要重新审视艺术的本质。也许答案在于：艺术的价值不仅在于技巧的精湛，更在于创作背后的情感、经历和人文思考。AI可以成为强大的创作工具，但艺术的灵魂仍然来自于人类的生命体验和情感共鸣。"""

        result = {
            "type": "RETURN",
            "description": content,
            "tokens_used": 140
        }
        if self.verbose:
            print(f"[Think #{self.call_count}] 生成艺术评论段落内容")
        return result


class DetailedAIArtEval(AIArtAsyncEvalBase):
    """详细版AI与艺术Eval算子 - 用于演示"""

    def _get_final_article_result(self, node: S) -> Dict[str, Any]:
        article = f"""# AI与艺术：创新与思考的交融

{node.done[0]}

{node.done[1]}

{node.done[2]}

---
本文探讨了AI在艺术领域的应用与影响，从技术创新到哲学思考，展现了AI与艺术融合的多重维度。（总计约{len(''.join(node.done))}字）"""

        result = {
            "type": "RETURN",
            "description": article,
            "tokens_used": 90
        }
        if self.verbose:
            print(f"[Eval #{self.call_count}] 完成整篇文章")
        return result


# 简化版本（用于测试）
class TestableAIArtThink(AIArtAsyncThinkBase):
    """简化版AI与艺术Think算子 - 用于测试"""

    def _get_art_paragraph_result(self) -> Dict[str, Any]:
        return {
            "type": "RETURN",
            "description": "AI艺术段落内容：DALL·E等工具正在改变艺术创作...",
            "tokens_used": 120
        }

    def _get_creation_paragraph_result(self) -> Dict[str, Any]:
        return {
            "type": "RETURN",
            "description": "辅助创作段落内容：AI成为艺术家的得力助手...",
            "tokens_used": 130
        }

    def _get_review_paragraph_result(self) -> Dict[str, Any]:
        return {
            "type": "RETURN",
            "description": "艺术评论段落内容：AI艺术引发的哲学思考...",
            "tokens_used": 140
        }


class TestableAIArtEval(AIArtAsyncEvalBase):
    """简化版AI与艺术Eval算子 - 用于测试"""

    def _get_final_article_result(self, node: S) -> Dict[str, Any]:
        article = f"完整的AI与艺术短文，包含三个段落：{'; '.join(node.done[:3])}"

        result = {
            "type": "RETURN",
            "description": article,
            "tokens_used": 90
        }
        return result