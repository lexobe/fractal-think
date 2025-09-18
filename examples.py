"""
分形思考（Fractal Thinkon）示例实现
展示如何使用 thinkon_core.py 框架
"""

import json
import logging
import random
from typing import Any, Dict
from thinkon_core import S, Down, ReturnUp, solve, TokenLimitExceeded


# 示例实现：基于计划锚点的策略
class PlanBasedThinkLLM:
    """基于计划锚点的 Think 实现，返回包含 token 消耗的 Dict"""
    
    def __init__(self, base_tokens: int = 50):
        self.base_tokens = base_tokens

    def __call__(self, node: S, memory: Any = None, tools: Any = None) -> Dict[str, Any]:
        # 简化实现：如果 context 中有未完成的计划项，则下潜
        if "[P" in node.context and "✓" not in node.context:
            # 提取第一个未完成的计划项
            lines = node.context.split('\n')
            for line in lines:
                if line.strip().startswith('[P') and '✓' not in line:
                    # 提取计划项内容
                    if ']' in line:
                        plan_content = line.split(']', 1)[1].strip()
                        child = S(goal=plan_content, context="")
                        # 在父层 context 中记录下潜理由
                        node.context += f"\n>>> Think: 选择下潜到 {line.strip()}"
                        
                        # 模拟 token 消耗（基于任务复杂度）
                        tokens_used = self.base_tokens + len(plan_content) // 10 + random.randint(5, 15)
                        
                        return {
                            "action_type": "down",
                            "tokens_used": tokens_used,
                            "data": {
                                "goal": plan_content,
                                "context": ""
                            },
                            "reasoning": f"发现未完成计划项: {line.strip()}，决定下潜处理",
                            "confidence": 0.9,
                            "metadata": {
                                "parent_level": node.level,
                                "plan_item": line.strip()
                            }
                        }

        # 否则返回完成结果
        result = f"完成目标: {node.goal}"
        tokens_used = self.base_tokens + len(result) // 10 + random.randint(3, 10)
        
        return {
            "action_type": "return_up",
            "tokens_used": tokens_used,
            "result": result,
            "confidence": 1.0,
            "reasoning": "所有计划项已完成或无更多任务",
            "metadata": {
                "completion_level": node.level,
                "total_tasks": len([l for l in node.context.split('\n') if l.strip().startswith('[P')])
            }
        }


class PlanBasedEvalLLM:
    """基于计划锚点的 Eval 实现，返回包含 token 消耗的 Dict"""
    
    def __init__(self, base_tokens: int = 40):
        self.base_tokens = base_tokens

    def __call__(self, node: S, child_result: ReturnUp, memory: Any = None) -> Dict[str, Any]:
        # 标记完成的子任务 - 找到第一个未完成的项目并标记为完成
        lines = node.context.split('\n')
        updated_lines = []
        marked_complete = False

        for line in lines:
            if line.strip().startswith('[P') and '✓' not in line and not marked_complete:
                # 找到第一个未完成的项目并标记为完成
                prefix = line.split(']')[0] + ']'
                plan_content = line.split(']', 1)[1].strip()
                updated_lines.append(f"{prefix} ✓ {plan_content}")
                marked_complete = True
            else:
                updated_lines.append(line)

        node.context = '\n'.join(updated_lines)
        node.context += f"\n>>> Eval: 接收子层结果 - {child_result.result}"

        # 检查是否还有未完成的计划项
        remaining_tasks = [line for line in node.context.split('\n')
                          if line.strip().startswith('[P') and '✓' not in line]

        if remaining_tasks:
            # 继续下一个任务
            next_task = remaining_tasks[0]
            if ']' in next_task:
                task_content = next_task.split(']', 1)[1].strip()
                child = S(goal=task_content, context="")
                node.context += f"\n>>> Eval: 继续下潜到下一任务 {next_task.strip()}"
                
                # 模拟 token 消耗
                tokens_used = self.base_tokens + len(task_content) // 10 + random.randint(5, 12)
                
                return {
                    "action_type": "down",
                    "tokens_used": tokens_used,
                    "data": {
                        "goal": task_content,
                        "context": ""
                    },
                    "reasoning": f"子任务完成，继续下一个任务: {next_task.strip()}",
                    "confidence": 0.8,
                    "evaluation": {
                        "child_result_quality": "good",
                        "task_progress": f"{len([l for l in node.context.split('\n') if l.strip().startswith('[P') and '✓' in l])}/{len([l for l in node.context.split('\n') if l.strip().startswith('[P')])}"
                    },
                    "metadata": {
                        "remaining_tasks": len(remaining_tasks),
                        "current_level": node.level
                    }
                }

        # 所有任务完成
        result = f"所有计划项已完成: {node.goal}"
        tokens_used = self.base_tokens + len(result) // 10 + random.randint(3, 8)
        
        return {
            "action_type": "return_up",
            "tokens_used": tokens_used,
            "result": result,
            "confidence": 1.0,
            "reasoning": "所有子任务已完成，本层可以返回",
            "evaluation": {
                "child_result_quality": "good",
                "completion_rate": 1.0,
                "remaining_tasks": 0
            },
            "metadata": {
                "completed_level": node.level,
                "child_confidence": child_result.confidence
            }
        }


class NestedThinkLLM:
    """支持嵌套任务的 Think 实现"""
    
    def __init__(self, base_tokens: int = 45):
        self.base_tokens = base_tokens

    def __call__(self, node: S, memory: Any = None, tools: Any = None) -> Dict[str, Any]:
        # 如果是叶子任务（包含具体技术词汇），直接完成
        if any(word in node.goal.lower() for word in ['界面', 'api', '数据库']):
            # 如果没有子计划，说明是具体实现
            if '[P' not in node.context:
                result = f"已实现: {node.goal}"
                tokens_used = self.base_tokens + len(result) // 10 + random.randint(5, 12)
                
                return {
                    "action_type": "return_up",
                    "tokens_used": tokens_used,
                    "result": result,
                    "confidence": 0.9,
                    "reasoning": "叶子任务直接完成",
                    "metadata": {
                        "task_type": "leaf",
                        "completion_level": node.level
                    }
                }

        # 检查是否有未完成的计划项
        if "[P" in node.context and "✓" not in node.context:
            lines = node.context.split('\n')
            for line in lines:
                if line.strip().startswith('[P') and '✓' not in line:
                    if ']' in line:
                        plan_content = line.split(']', 1)[1].strip()

                        # 为特定任务创建子计划
                        child_context = ""
                        if "前端界面" in plan_content:
                            child_context = """《子任务计划》:
[P1] 设计用户界面布局
[P2] 实现交互功能
[P3] 优化响应式设计"""
                        elif "后端 API" in plan_content:
                            child_context = """《子任务计划》:
[P1] 设计 API 架构
[P2] 实现核心接口
[P3] 添加认证授权"""
                        elif "数据库" in plan_content:
                            child_context = """《子任务计划》:
[P1] 设计数据模型
[P2] 创建数据表
[P3] 配置连接池"""

                        child = S(goal=plan_content, context=child_context)
                        node.context += f"\n>>> Think: 选择下潜到 {line.strip()}"
                        
                        # 模拟复杂任务的更高 token 消耗
                        tokens_used = self.base_tokens + len(plan_content) // 8 + len(child_context) // 20 + random.randint(8, 20)
                        
                        return {
                            "action_type": "down",
                            "tokens_used": tokens_used,
                            "data": {
                                "goal": plan_content,
                                "context": child_context
                            },
                            "reasoning": f"复杂任务需要分解: {line.strip()}",
                            "confidence": 0.85,
                            "metadata": {
                                "task_type": "complex",
                                "has_subplan": bool(child_context),
                                "parent_level": node.level
                            }
                        }

        # 否则返回完成结果
        result = f"完成目标: {node.goal}"
        tokens_used = self.base_tokens + len(result) // 10 + random.randint(3, 10)
        
        return {
            "action_type": "return_up",
            "tokens_used": tokens_used,
            "result": result,
            "confidence": 1.0,
            "reasoning": "所有任务已完成",
            "metadata": {
                "completion_level": node.level
            }
        }


def create_example_root() -> S:
    """创建示例根节点，模拟 thinkon.md 中的例子"""
    return S(
        goal="写一篇关于'AI 与艺术'的短文",
        context="""《目标》: 写一篇关于'AI 与艺术'的短文
《完成判据》: 三段齐备且连贯，需覆盖生成艺术/辅助创作/艺术评论
《计划锚点 v1》:
[P1] 写生成艺术段落：介绍 AI 生成艺术的技术和案例
[P2] 写辅助创作段落：探讨 AI 如何辅助人类艺术创作
[P3] 写艺术评论段落：分析 AI 对艺术评论和鉴赏的影响"""
    )


def create_nested_example() -> S:
    """创建嵌套任务示例，展示深层递归能力"""
    return S(
        goal="构建一个完整的 Web 应用",
        context="""《目标》: 构建一个完整的 Web 应用
《完成判据》: 前端、后端、数据库三部分齐全且可运行
《计划锚点 v1》:
[P1] 设计前端界面
[P2] 开发后端 API
[P3] 配置数据库"""
    )


def run_basic_example():
    """运行基础计划任务示例"""
    print("=== 示例1：基础分形思考递归框架演示（Dict 格式 + Token 追踪）===")
    root1 = create_example_root()
    think_llm = PlanBasedThinkLLM()
    eval_llm = PlanBasedEvalLLM()

    print(f"根节点: {root1.goal}")
    print(f"初始 context:\n{root1.context}")
    print("\n" + "="*50)

    try:
        result1, total_tokens = solve(root1, think_llm, eval_llm, max_tokens=500)
        print(f"\n最终结果: {result1.result}")
        print(f"总 Token 消耗: {total_tokens}")

        # 统计完成的任务
        completed_tasks = [line for line in root1.context.split('\n')
                          if line.strip().startswith('[P') and '✓' in line]
        print(f"已完成任务数: {len(completed_tasks)}")

    except TokenLimitExceeded as e:
        print(f"Token 限制超出: {e}")
    except Exception as e:
        print(f"执行出错: {e}")


def run_json_example():
    """运行 JSON 格式展示示例"""
    print("=== 示例2：JSON 格式展示（S 作为 dict）===")
    root2 = create_example_root()
    
    print(f"根节点: {root2.goal}")
    print(f"根节点作为 dict: {json.dumps(dict(root2), ensure_ascii=False, indent=2)}")
    print(f"根节点 to_json(): {json.dumps(root2.to_json(), ensure_ascii=False, indent=2)}")
    print("\n" + "="*50)

    try:
        think_llm = PlanBasedThinkLLM()
        eval_llm = PlanBasedEvalLLM()
        
        result2, total_tokens = solve(root2, think_llm, eval_llm, max_tokens=300)
        print(f"\n最终结果: {result2.result}")
        print(f"最终结果 JSON: {json.dumps(result2.to_json(), ensure_ascii=False, indent=2)}")
        print(f"总 Token 消耗: {total_tokens}")

    except TokenLimitExceeded as e:
        print(f"Token 限制超出: {e}")
    except Exception as e:
        print(f"执行出错: {e}")
        import traceback
        traceback.print_exc()


def run_nested_example():
    """运行嵌套任务示例"""
    print("=== 示例3：嵌套递归任务演示（低 Token 限制测试）===")
    root3 = create_nested_example()
    nested_think_llm = NestedThinkLLM()
    eval_llm = PlanBasedEvalLLM()

    print(f"根节点: {root3.goal}")
    print(f"初始 context:\n{root3.context}")
    print("\n" + "="*50)

    try:
        result3, total_tokens = solve(root3, nested_think_llm, eval_llm, max_tokens=800)
        print(f"\n最终结果: {result3.result}")
        print(f"总 Token 消耗: {total_tokens}")

        # 显示执行轨迹
        print("\n=== 执行轨迹（最后10行）===")
        context_lines = root3.context.split('\n')
        trace_lines = [line for line in context_lines if line.strip().startswith('>>>')]
        for line in trace_lines[-10:]:
            print(line)

    except TokenLimitExceeded as e:
        print(f"Token 限制超出: {e}")
        # 显示已完成的轨迹
        print("\n=== 已完成的执行轨迹 ===")
        context_lines = root3.context.split('\n')
        trace_lines = [line for line in context_lines if line.strip().startswith('>>>')]
        for line in trace_lines:
            print(line)
    except Exception as e:
        print(f"执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 配置 logging 输出级别
    logging.getLogger('FractalThinkon').setLevel(logging.INFO)
    
    # 运行所有示例
    run_basic_example()
    print("\n" + "="*80)
    
    run_json_example()
    print("\n" + "="*80)
    
    run_nested_example()
