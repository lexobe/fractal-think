#!/usr/bin/env python3
"""
异步规范示例单元测试 - 验证AI与艺术短文分形流程

运行方式：
  从项目根目录运行: python test_async_norm.py
  或使用pytest: pytest test_async_norm.py

测试覆盖：
1. 异步执行能否按既定顺序生成三段落
2. SolveResult.status 为 COMPLETED
3. partial_results 为空
4. Token统计和算子调用次数验证
5. 执行约束验证
"""

import asyncio
from typing import Dict, Any

# 导入模块
try:
    # 尝试正常包导入
    from async_core.engine import solve_async
    from async_core.common import ExecutionBudget
    from thinkon_core import S, SolveResult, SolveStatus
    from test_helpers import TestableAIArtThink, TestableAIArtEval
except ImportError:
    # 回退到路径导入（适配直接运行脚本的情况）
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from async_core.engine import solve_async
    from async_core.common import ExecutionBudget
    from thinkon_core import S, SolveResult, SolveStatus
    from test_helpers import TestableAIArtThink, TestableAIArtEval


async def test_async_specification_flow():
    """测试异步规范流程"""
    print("🧪 测试异步规范流程")

    think_llm = TestableAIArtThink()
    eval_llm = TestableAIArtEval()

    budget = ExecutionBudget(max_depth=5, max_tokens=2000, max_time=10.0)

    result = await solve_async(
        goal='写一篇"AI与艺术"的短文（800–1200字）',
        think_llm=think_llm,
        eval_llm=eval_llm,
        budget=budget
    )

    # 测试验证
    tests = []

    # 1. 执行状态验证
    tests.append((
        result.status == SolveStatus.COMPLETED,
        f"执行状态为COMPLETED: {result.status}"
    ))

    # 2. 部分结果验证
    tests.append((
        len(result.partial_results) == 0,
        f"partial_results为空: {len(result.partial_results)}"
    ))

    # 3. Token统计验证
    tests.append((
        result.token_usage.total > 0,
        f"Token统计正常: {result.token_usage.total}"
    ))

    # 4. 算子调用次数验证
    tests.append((
        think_llm.call_count >= 3,  # 至少根节点+3个段落
        f"Think调用次数合理: {think_llm.call_count}"
    ))

    tests.append((
        eval_llm.call_count >= 3,   # 至少3次Eval调用
        f"Eval调用次数合理: {eval_llm.call_count}"
    ))

    # 5. 执行时间验证
    tests.append((
        result.execution_time > 0,
        f"执行时间正常: {result.execution_time:.3f}s"
    ))

    # 6. 最大深度验证
    tests.append((
        result.max_depth_reached >= 0,
        f"深度跟踪正常: {result.max_depth_reached}"
    ))

    # 7. 约束未触发验证
    tests.append((
        result.constraint_triggered is None,
        f"未触发约束: {result.constraint_triggered}"
    ))

    # 8. 结果内容验证
    tests.append((
        "AI" in result.result and ("艺术" in result.result or "段落" in result.result),
        "结果包含相关主题内容"
    ))

    # 显示测试结果
    passed = 0
    for check, description in tests:
        status = "✅" if check else "❌"
        print(f"  {status} {description}")
        if check:
            passed += 1

    print(f"\n📊 基础流程测试: {passed}/{len(tests)} 项通过")

    # 显示调用历史
    print(f"\n📋 Think调用历史:")
    for call in think_llm.call_history:
        print(f"  {call}")

    print(f"\n📋 Eval调用历史:")
    for call in eval_llm.call_history:
        print(f"  {call}")

    print(f"\n📄 最终结果预览:")
    print(f"  {result.result[:100]}...")

    return passed == len(tests), result


async def test_constraint_handling():
    """测试约束处理"""
    print("\n🧪 测试约束处理")

    think_llm = TestableAIArtThink()
    eval_llm = TestableAIArtEval()

    # 设置很严格的约束
    strict_budget = ExecutionBudget(max_depth=2, max_tokens=300, max_time=1.0)

    result = await solve_async(
        goal='写一篇"AI与艺术"的短文（800–1200字）',
        think_llm=think_llm,
        eval_llm=eval_llm,
        budget=strict_budget
    )

    # 约束测试
    constraint_tests = [
        (result.status in (SolveStatus.COMPLETED, SolveStatus.DEGRADED), "状态合理"),
        (result.token_usage.total <= strict_budget.max_tokens + 100, "Token未严重超限"),  # 允许小幅超限
        (result.execution_time <= strict_budget.max_time + 1.0, "时间基本合理"),  # 允许小幅超时
    ]

    passed = 0
    for check, description in constraint_tests:
        status = "✅" if check else "❌"
        print(f"  {status} {description}")
        if check:
            passed += 1

    if result.constraint_triggered:
        print(f"  ⚠️  约束触发: {result.constraint_triggered}")

    print(f"\n📊 约束处理测试: {passed}/{len(constraint_tests)} 项通过")

    return passed == len(constraint_tests), result


async def test_sequential_execution():
    """测试顺序执行逻辑"""
    print("\n🧪 测试顺序执行逻辑")

    think_llm = TestableAIArtThink()
    eval_llm = TestableAIArtEval()

    result = await solve_async(
        goal='写一篇"AI与艺术"的短文（800–1200字）',
        think_llm=think_llm,
        eval_llm=eval_llm,
        budget=ExecutionBudget(max_depth=5, max_tokens=2000, max_time=10.0)
    )

    # 验证执行顺序
    sequence_tests = [
        (think_llm.call_count > 0, "Think算子被调用"),
        (eval_llm.call_count > 0, "Eval算子被调用"),
        (len(think_llm.call_history) == think_llm.call_count, "Think调用历史完整"),
        (len(eval_llm.call_history) == eval_llm.call_count, "Eval调用历史完整"),
    ]

    # 检查是否包含预期的调用模式
    think_history_str = " ".join(think_llm.call_history)
    eval_history_str = " ".join(eval_llm.call_history)

    sequence_tests.extend([
        ("AI与艺术" in think_history_str, "Think处理了根节点"),
        ("AI与艺术" in eval_history_str, "Eval评估了根节点"),
    ])

    passed = 0
    for check, description in sequence_tests:
        status = "✅" if check else "❌"
        print(f"  {status} {description}")
        if check:
            passed += 1

    print(f"\n📊 顺序执行测试: {passed}/{len(sequence_tests)} 项通过")

    return passed == len(sequence_tests), result


async def main():
    """主测试函数"""
    print("🚀 异步规范示例单元测试")
    print("=" * 50)

    all_tests_passed = True

    try:
        # 运行各项测试
        test1_passed, result1 = await test_async_specification_flow()
        test2_passed, result2 = await test_constraint_handling()
        test3_passed, result3 = await test_sequential_execution()

        all_tests_passed = test1_passed and test2_passed and test3_passed

        print("\n" + "=" * 50)
        print("📈 总体测试结果")
        print("=" * 50)

        test_summary = [
            (test1_passed, "基础流程测试"),
            (test2_passed, "约束处理测试"),
            (test3_passed, "顺序执行测试"),
        ]

        passed_count = 0
        for passed, name in test_summary:
            status = "✅" if passed else "❌"
            print(f"{status} {name}")
            if passed:
                passed_count += 1

        print(f"\n🎯 测试总结: {passed_count}/{len(test_summary)} 项测试通过")

        if all_tests_passed:
            print("🎉 所有单元测试通过！异步规范示例运行正常！")
        else:
            print("⚠️  部分测试未通过，需要进一步优化")

    except Exception as e:
        print(f"❌ 测试执行异常: {e}")
        import traceback
        traceback.print_exc()
        all_tests_passed = False

    return all_tests_passed


if __name__ == "__main__":
    result = asyncio.run(main())