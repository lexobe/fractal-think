"""
基于规则的Think和Eval算子示例

这些示例算子使用简单的关键字匹配和状态机逻辑，
适用于演示框架功能和快速原型开发。
在生产环境中，建议使用基于LLM的算子实现。
"""

from typing import Any, Dict
import sys
import os

# 优雅的导入方式
try:
    # 优先尝试直接导入（当作为包使用时）
    from thinkon_core import S
except ImportError:
    try:
        # 尝试相对导入（当作为子模块使用时）
        from ..thinkon_core import S
    except ImportError:
        # 最后回退到路径操作（用于直接运行脚本）
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from thinkon_core import S


class RuleBasedThink:
    """
    基于规则的Think算子示例
    根据关键字把目标拆成2-3个子任务并汇总
    """

    def __init__(self):
        self.call_count = 0

    def __call__(self, node: S, memory: Any = None, tools: Any = None) -> Dict[str, Any]:
        self.call_count += 1

        goal = node.goal.lower()

        # 简单任务直接返回
        if any(keyword in goal for keyword in ["简单", "直接", "立即"]):
            return {
                "type": "RETURN",
                "description": f"简单任务已完成：{node.goal}",
                "tokens_used": 5
            }

        # 复杂任务制定计划
        if any(keyword in goal for keyword in ["复杂", "困难", "多步骤", "详细"]):
            plan = f"""针对目标"{node.goal}"的执行计划：
1. 需求分析和理解
2. 方案设计和规划
3. 具体实施和验证"""
            return {
                "type": "TODO",
                "description": plan,
                "tokens_used": 15
            }

        # 包含"写"的任务
        if "写" in goal:
            plan = f"""写作任务"{node.goal}"分解：
1. 收集素材和确定大纲
2. 撰写主体内容
3. 修改和完善"""
            return {
                "type": "TODO",
                "description": plan,
                "tokens_used": 12
            }

        # 默认制定简单计划
        return {
            "type": "TODO",
            "description": f"执行目标：{node.goal}。分解为准备、执行、验证三个步骤。",
            "tokens_used": 8
        }


class RuleBasedEval:
    """
    基于规则的Eval算子示例
    基于todo和done状态决定下一步行动
    """

    def __init__(self):
        self.call_count = 0

    def __call__(self, node: S, memory: Any = None) -> Dict[str, Any]:
        self.call_count += 1

        # 没有计划直接返回
        if not node.todo:
            return {
                "type": "RETURN",
                "description": f"目标完成：{node.goal}",
                "tokens_used": 3
            }

        done_count = len(node.done)

        # 解析计划步骤以提高可解释性
        todo_lines = [line.strip() for line in node.todo.split('\n') if line.strip()]
        total_steps = len([line for line in todo_lines if any(marker in line for marker in ["1.", "2.", "3.", "步骤", "："])])

        # 根据已完成数量决定下一步
        if done_count == 0:
            # 开始第一个子任务
            if "1." in node.todo or "需求分析" in node.todo:
                return {
                    "type": "CALL",
                    "description": "执行需求分析和理解",
                    "tokens_used": 6
                }
            elif "收集素材" in node.todo:
                return {
                    "type": "CALL",
                    "description": "收集素材和确定大纲",
                    "tokens_used": 6
                }
            else:
                return {
                    "type": "CALL",
                    "description": "开始执行第一个步骤",
                    "tokens_used": 5
                }

        elif done_count == 1:
            # 第二个子任务
            if "方案设计" in node.todo:
                return {
                    "type": "CALL",
                    "description": "方案设计和规划",
                    "tokens_used": 6
                }
            elif "撰写主体" in node.todo:
                return {
                    "type": "CALL",
                    "description": "撰写主体内容",
                    "tokens_used": 6
                }
            else:
                return {
                    "type": "CALL",
                    "description": "执行第二个步骤",
                    "tokens_used": 5
                }

        elif done_count == 2:
            # 第三个子任务
            if "实施和验证" in node.todo:
                return {
                    "type": "CALL",
                    "description": "具体实施和验证",
                    "tokens_used": 6
                }
            elif "修改和完善" in node.todo:
                return {
                    "type": "CALL",
                    "description": "修改和完善",
                    "tokens_used": 6
                }
            else:
                return {
                    "type": "CALL",
                    "description": "执行第三个步骤",
                    "tokens_used": 5
                }

        else:
            # 任务完成 - 使用步骤信息提供更好的摘要
            if total_steps > 0 and done_count >= total_steps:
                summary = f"任务完成，成功执行{total_steps}个规划步骤："
            else:
                summary = f"任务完成，包含{done_count}个步骤的结果："

            for i, result in enumerate(node.done[:3], 1):
                summary += f" {i}){result[:30]}..."
            if len(node.done) > 3:
                summary += f" 等{len(node.done)}项"

            return {
                "type": "RETURN",
                "description": summary,
                "tokens_used": 8
            }