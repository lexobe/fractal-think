"""
异步核心示例 - 演示新架构的关键特性

本示例展示：
1. 基础异步求解
2. 快照创建和恢复
3. 批量并发处理
4. 约束管理和优雅降级
5. 同步兼容性
"""

import asyncio
import time
import json
from typing import Dict, Any

# 导入异步核心组件
from async_core.engine import solve_async
from async_core.sync_adapter import LegacyCompatibilityLayer, solve_with_async_engine
from async_core.recovery import RecoveryManager, RecoveryMode
from async_core.common import ExecutionBudget, UnifiedLogger, ExecutionMode
from async_core.interfaces import AsyncThinkLLM, AsyncEvalLLM

# 导入基础组件
from thinkon_core import S, SolveResult, SolveStatus


class DemoAsyncThinkLLM:
    """演示用异步Think算子"""

    def __init__(self, delay: float = 0.1):
        self.delay = delay
        self.call_count = 0

    async def __call__(self, node: S, memory=None, tools=None) -> Dict[str, Any]:
        self.call_count += 1
        await asyncio.sleep(self.delay)  # 模拟异步LLM调用

        goal = node.goal.lower()

        # 根据任务复杂度决定分解策略
        if "复杂" in goal or "详细" in goal or len(goal) > 30:
            # 需要分解的复杂任务
            if "分析" in goal:
                subtasks = [
                    "1. 收集相关数据和信息",
                    "2. 识别关键因素和模式",
                    "3. 深入分析数据关联性",
                    "4. 得出结论和建议"
                ]
            elif "设计" in goal:
                subtasks = [
                    "1. 需求分析和用户调研",
                    "2. 架构设计和技术选型",
                    "3. 详细设计和原型开发",
                    "4. 测试验证和优化改进"
                ]
            elif "规划" in goal:
                subtasks = [
                    "1. 现状评估和目标定义",
                    "2. 资源分析和约束识别",
                    "3. 方案制定和风险评估",
                    "4. 执行计划和监控机制"
                ]
            else:
                subtasks = [
                    "1. 分析问题的核心要素",
                    "2. 制定解决方案",
                    "3. 验证方案可行性"
                ]

            return {
                "type": "TODO",
                "description": "\n".join(subtasks),
                "tokens_used": 120 + len(goal) * 2
            }
        else:
            # 简单任务直接完成
            result = f"已完成任务: {goal}"
            if "数据" in goal:
                result += " - 收集了相关数据并进行初步分析"
            elif "方案" in goal:
                result += " - 制定了可行的解决方案"
            elif "评估" in goal:
                result += " - 完成了全面的评估分析"

            return {
                "type": "RETURN",
                "description": result,
                "tokens_used": 80 + len(goal)
            }


class DemoAsyncEvalLLM:
    """演示用异步Eval算子"""

    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.call_count = 0

    async def __call__(self, node: S, memory=None) -> Dict[str, Any]:
        self.call_count += 1
        await asyncio.sleep(self.delay)

        completed_count = len(node.done)
        total_work = len(node.todo) if hasattr(node, 'todo') else 0

        # 根据完成情况决定下一步行动
        if completed_count == 0:
            # 刚开始，继续执行
            return {
                "type": "CALL",
                "description": "开始执行第一个子任务",
                "tokens_used": 60
            }
        elif completed_count < 3:
            # 还需要更多信息
            return {
                "type": "CALL",
                "description": f"已完成{completed_count}个子任务，继续收集信息",
                "tokens_used": 65
            }
        else:
            # 有足够信息，可以返回结果
            summary = f"基于{completed_count}个子任务的结果: " + "; ".join(node.done[-2:])
            return {
                "type": "RETURN",
                "description": f"任务完成。{summary}",
                "tokens_used": 80 + len(summary)
            }


async def demo_basic_async_solve():
    """演示1: 基础异步求解"""
    print("=== 演示1: 基础异步求解 ===")

    think_llm = DemoAsyncThinkLLM(delay=0.1)
    eval_llm = DemoAsyncEvalLLM(delay=0.05)

    start_time = time.time()

    result = await solve_async(
        goal="设计一个复杂的用户管理系统",
        think_llm=think_llm,
        eval_llm=eval_llm,
        budget=ExecutionBudget(
            max_depth=6,
            max_tokens=5000,
            max_time=30.0
        )
    )

    elapsed = time.time() - start_time

    print(f"执行结果: {result.result}")
    print(f"执行状态: {result.status}")
    print(f"Token消耗: {result.token_usage.total}")
    print(f"Think调用: {result.token_usage.think_calls}")
    print(f"Eval调用: {result.token_usage.eval_calls}")
    print(f"执行时间: {elapsed:.2f}秒")
    print()

    return result


async def demo_constraint_handling():
    """演示2: 约束管理和优雅降级"""
    print("=== 演示2: 约束管理和优雅降级 ===")

    think_llm = DemoAsyncThinkLLM(delay=0.1)
    eval_llm = DemoAsyncEvalLLM(delay=0.05)

    # 设置很严格的限制来触发约束
    result = await solve_async(
        goal="进行详细的市场分析和竞争对手研究",
        think_llm=think_llm,
        eval_llm=eval_llm,
        budget=ExecutionBudget(
            max_depth=3,      # 很浅的深度限制
            max_tokens=800,   # 很低的token限制
            max_time=5.0      # 很短的时间限制
        )
    )

    print(f"执行结果: {result.result}")
    print(f"执行状态: {result.status}")
    print(f"约束触发: {result.constraint_triggered}")
    print(f"Token消耗: {result.token_usage.total}")
    print()

    return result


async def demo_snapshot_and_recovery():
    """演示3: 快照创建和恢复"""
    print("=== 演示3: 快照创建和恢复 ===")

    # 创建恢复管理器
    recovery_manager = RecoveryManager("./demo_snapshots")

    think_llm = DemoAsyncThinkLLM(delay=0.1)
    eval_llm = DemoAsyncEvalLLM(delay=0.05)

    # 第一次执行 - 会因约束触发而中断
    print("第一次执行 (会触发约束):")
    result1 = await solve_async(
        goal="制定详细的产品规划和技术路线图",
        think_llm=think_llm,
        eval_llm=eval_llm,
        budget=ExecutionBudget(max_tokens=1200, max_time=8.0)
    )

    print(f"第一次结果: {result1.status}")
    if result1.constraint_triggered:
        print(f"触发约束: {result1.constraint_triggered}")

    # 模拟快照创建（实际中会在约束触发时自动创建）
    print("\n创建模拟快照...")
    # 这里为了演示，我们直接创建一个假的快照
    # 实际使用中，快照会在执行过程中自动生成

    print("\n列出可恢复的快照:")
    snapshots = recovery_manager.list_recoverable_snapshots()
    for snap in snapshots:
        print(f"- {snap['snapshot_id']}: {snap['goal'][:50]}...")

    print()
    return result1


async def demo_batch_processing():
    """演示4: 批量并发处理"""
    print("=== 演示4: 批量并发处理 ===")

    from async_core.sync_adapter import AsyncOptimizedLayer

    think_llm = DemoAsyncThinkLLM(delay=0.1)
    eval_llm = DemoAsyncEvalLLM(delay=0.05)

    # 批量任务
    tasks = [
        "分析用户需求和痛点",
        "设计产品核心功能",
        "制定技术实施方案",
        "评估项目风险和成本",
        "规划上线和推广策略"
    ]

    start_time = time.time()

    results = await AsyncOptimizedLayer.solve_batch_async(
        goals=tasks,
        think_llm=think_llm,
        eval_llm=eval_llm,
        budget_per_task=ExecutionBudget(max_tokens=1500, max_time=15.0),
        max_concurrent=3
    )

    elapsed = time.time() - start_time

    print(f"批量处理{len(tasks)}个任务，耗时{elapsed:.2f}秒")
    for i, result in enumerate(results):
        print(f"任务{i+1}: {result.status} - {result.result[:60]}...")

    total_tokens = sum(r.token_usage.total for r in results)
    print(f"总Token消耗: {total_tokens}")
    print()

    return results


def demo_sync_compatibility():
    """演示5: 同步兼容性"""
    print("=== 演示5: 同步兼容性 ===")

    # 创建同步版本的算子
    class SyncThinkLLM:
        def __call__(self, node, memory=None, tools=None):
            if "复杂" in node.goal:
                return {
                    "type": "TODO",
                    "description": "1. 分析需求\n2. 设计方案\n3. 实施计划",
                    "tokens_used": 150
                }
            else:
                return {
                    "type": "RETURN",
                    "description": f"完成同步任务: {node.goal}",
                    "tokens_used": 100
                }

    class SyncEvalLLM:
        def __call__(self, node, memory=None):
            if len(node.done) >= 2:
                return {
                    "type": "RETURN",
                    "description": "同步评估完成",
                    "tokens_used": 80
                }
            else:
                return {
                    "type": "CALL",
                    "description": "继续同步执行",
                    "tokens_used": 60
                }

    sync_think = SyncThinkLLM()
    sync_eval = SyncEvalLLM()

    # 方式1: 使用遗留兼容层
    print("方式1: 遗留兼容API")
    result1 = LegacyCompatibilityLayer.solve(
        goal="优化系统性能",
        think_llm=sync_think,
        eval_llm=sync_eval
    )
    print(f"遗留API结果: {result1}")

    # 方式2: 使用同步适配器
    print("\n方式2: 同步适配器")
    result2 = solve_with_async_engine(
        goal="复杂的数据处理任务",
        think_llm=sync_think,
        eval_llm=sync_eval,
        budget=ExecutionBudget(max_tokens=2000)
    )
    print(f"适配器结果: {result2.result}")
    print(f"适配器状态: {result2.status}")
    print()

    return result1, result2


async def demo_custom_logger():
    """演示6: 自定义日志和监控"""
    print("=== 演示6: 自定义日志和监控 ===")

    import logging

    # 设置自定义日志器
    logger = logging.getLogger("fractal_demo")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    unified_logger = UnifiedLogger(logger, ExecutionMode.ASYNC)

    think_llm = DemoAsyncThinkLLM(delay=0.1)
    eval_llm = DemoAsyncEvalLLM(delay=0.05)

    result = await solve_async(
        goal="带监控的任务执行",
        think_llm=think_llm,
        eval_llm=eval_llm,
        budget=ExecutionBudget(max_tokens=1000),
        logger=unified_logger
    )

    print(f"监控执行结果: {result.status}")
    print()

    return result


async def main():
    """主演示函数"""
    print("Fractal Thinkon 异步核心演示")
    print("=" * 50)

    try:
        # 运行所有演示
        await demo_basic_async_solve()
        await demo_constraint_handling()
        await demo_snapshot_and_recovery()
        await demo_batch_processing()
        demo_sync_compatibility()
        await demo_custom_logger()

        print("=== 演示完成 ===")
        print("异步核心提供了以下关键能力:")
        print("✓ 高性能异步执行")
        print("✓ 智能约束管理")
        print("✓ 状态快照和恢复")
        print("✓ 批量并发处理")
        print("✓ 完整向后兼容")
        print("✓ 灵活的日志监控")

    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 创建演示快照目录
    import os
    os.makedirs("./demo_snapshots", exist_ok=True)

    # 运行演示
    asyncio.run(main())

    # 清理演示文件
    import shutil
    if os.path.exists("./demo_snapshots"):
        shutil.rmtree("./demo_snapshots")