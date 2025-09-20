#!/usr/bin/env python3
"""
异步规范示例 - 复现 thinkon.md 第 4.1 节"AI与艺术"短文的分形流程

运行方式：
  从项目根目录运行: python examples/async_norm_example.py
  或使用模块方式: python -m examples.async_norm_example

本示例使用 async_core 异步执行框架，完整还原规范示例的执行流程，
输出与规范一致的 todo、done 以及最终结果。
"""

import asyncio
import time
from typing import Dict, Any

# 导入模块
try:
    # 尝试正常包导入
    from async_core.engine import solve_async
    from async_core.sync_adapter import solve_with_async_engine
    from async_core.common import ExecutionBudget
    from thinkon_core import S, SolveResult, SolveStatus
    from test_helpers import DetailedAIArtThink, DetailedAIArtEval
except ImportError:
    # 回退到路径导入（适配直接运行脚本的情况）
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from async_core.engine import solve_async
    from async_core.sync_adapter import solve_with_async_engine
    from async_core.common import ExecutionBudget
    from thinkon_core import S, SolveResult, SolveStatus
    from test_helpers import DetailedAIArtThink, DetailedAIArtEval


async def run_async_specification_example():
    """运行异步规范示例"""
    print("🚀 启动异步规范示例：AI与艺术短文分形流程")
    print("=" * 60)

    # 创建异步算子
    think_llm = DetailedAIArtThink(simulation_delay=0.3, verbose=True)
    eval_llm = DetailedAIArtEval(simulation_delay=0.2, verbose=True)

    # 设置执行约束
    budget = ExecutionBudget(
        max_depth=5,
        max_tokens=2000,
        max_time=10.0
    )

    print(f"📋 执行约束: max_depth={budget.max_depth}, max_tokens={budget.max_tokens}, max_time={budget.max_time}s")
    print()

    start_time = time.time()

    # 执行异步求解
    result = await solve_async(
        goal='写一篇"AI与艺术"的短文（800–1200字）',
        think_llm=think_llm,
        eval_llm=eval_llm,
        budget=budget
    )

    execution_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("📊 执行结果分析")
    print("=" * 60)

    print(f"✅ 执行状态: {result.status}")
    print(f"⏱️  执行时间: {execution_time:.2f}s (引擎记录: {result.execution_time:.3f}s)")
    print(f"📊 最大深度: {result.max_depth_reached}")
    print(f"🔢 Token统计: {result.token_usage.total}")
    print(f"   - Think调用: {result.token_usage.think_calls} 次，消耗: {result.token_usage.think_tokens}")
    print(f"   - Eval调用: {result.token_usage.eval_calls} 次，消耗: {result.token_usage.eval_tokens}")
    print(f"💾 算子调用统计:")
    print(f"   - Think实际调用: {think_llm.call_count} 次")
    print(f"   - Eval实际调用: {eval_llm.call_count} 次")

    if result.constraint_triggered:
        print(f"⚠️  约束触发: {result.constraint_triggered}")

    if result.partial_results:
        print(f"📝 部分结果: {len(result.partial_results)} 项")

    print(f"\n📄 最终结果:")
    print("-" * 40)
    print(result.result)

    return result


def run_sync_wrapper_example():
    """运行同步包装器示例"""
    print("\n🔄 同步包装器测试")
    print("=" * 40)

    think_llm = DetailedAIArtThink(simulation_delay=0.1, verbose=False)  # 较短延时
    eval_llm = DetailedAIArtEval(simulation_delay=0.05, verbose=False)

    result = solve_with_async_engine(
        goal='写一篇"AI与艺术"的短文（800–1200字）',
        think_llm=think_llm,
        eval_llm=eval_llm,
        budget=ExecutionBudget(max_depth=5, max_tokens=2000, max_time=10.0)
    )

    print(f"同步包装器结果状态: {result.status}")
    print(f"Token统计: {result.token_usage.total}")

    return result


async def main():
    """主程序"""
    try:
        # 运行异步示例
        async_result = await run_async_specification_example()

        # 运行同步包装器示例
        sync_result = run_sync_wrapper_example()

        print("\n🎯 规范验证")
        print("=" * 40)

        # 验证结果
        success_criteria = [
            (async_result.status == SolveStatus.COMPLETED, "异步执行状态为COMPLETED"),
            (async_result.token_usage.total > 0, "Token统计正常"),
            (async_result.max_depth_reached > 0, "深度跟踪正常"),
            (len(async_result.partial_results) == 0, "无部分结果（完整完成）"),
            ("AI与艺术" in async_result.result, "结果包含主题内容"),
            (len(async_result.result) > 500, "文章长度合理")
        ]

        passed = 0
        for check, description in success_criteria:
            status = "✅" if check else "❌"
            print(f"{status} {description}")
            if check:
                passed += 1

        print(f"\n📈 验证结果: {passed}/{len(success_criteria)} 项通过")

        if passed == len(success_criteria):
            print("🎉 规范示例完美复现！")
        else:
            print("⚠️  部分验证未通过，需要进一步调试")

    except Exception as e:
        print(f"❌ 执行异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())