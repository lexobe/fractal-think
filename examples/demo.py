#!/usr/bin/env python3
"""
Fractal Thinkon 核心库演示脚本

运行方式：
    python examples/demo.py
    或
    python -m examples.demo

演示内容：
1. 简单任务执行（Think直接返回）
2. 复杂任务执行（需要计划和子任务）
3. 写作任务执行（特定领域示例）
4. 约束机制演示（深度、资源、时间限制）
5. 错误处理演示
"""

import sys
import os

# 优雅的导入方式
try:
    # 优先尝试直接导入（当作为包使用时）
    from thinkon_core import (
        S, Constraints, start_solve, start_solve_simple,
        DepthLimitExceeded, ResourceLimitExceeded, TimeLimitExceeded,
        setup_logger
    )
    from strategies import RuleBasedThink, RuleBasedEval
except ImportError:
    try:
        # 尝试相对导入（当作为子模块使用时）
        from ..thinkon_core import (
            S, Constraints, start_solve, start_solve_simple,
            DepthLimitExceeded, ResourceLimitExceeded, TimeLimitExceeded,
            setup_logger
        )
        from ..strategies import RuleBasedThink, RuleBasedEval
    except ImportError:
        # 最后回退到路径操作（用于直接运行脚本）
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from thinkon_core import (
            S, Constraints, start_solve, start_solve_simple,
            DepthLimitExceeded, ResourceLimitExceeded, TimeLimitExceeded,
            setup_logger
        )
        from strategies import RuleBasedThink, RuleBasedEval


def demo_simple_task():
    """演示简单任务执行"""
    print("=== 演示1: 简单任务执行 ===")

    think_strategy = RuleBasedThink()
    eval_strategy = RuleBasedEval()

    solve_result = start_solve(
        goal="简单数学计算",
        think_llm=think_strategy,
        eval_llm=eval_strategy,
        constraints=Constraints(max_depth=3)
    )

    print(f"结果: {solve_result.result}")
    print(f"状态: {solve_result.status.value}")
    print(f"Token消耗: {solve_result.token_usage.total}")
    print(f"Think调用次数: {think_strategy.call_count}")
    print(f"Eval调用次数: {eval_strategy.call_count}")
    print()


def demo_complex_task():
    """演示复杂任务执行"""
    print("=== 演示2: 复杂任务执行 ===")

    think_strategy = RuleBasedThink()
    eval_strategy = RuleBasedEval()

    result = start_solve(
        goal="复杂项目管理系统设计",
        think_llm=think_strategy,
        eval_llm=eval_strategy,
        constraints=Constraints(max_depth=5, max_tokens=200)
    )

    print(f"结果: {result}")
    print(f"Think调用次数: {think_strategy.call_count}")
    print(f"Eval调用次数: {eval_strategy.call_count}")
    print()


def demo_writing_task():
    """演示写作任务执行"""
    print("=== 演示3: 写作任务执行 ===")

    think_strategy = RuleBasedThink()
    eval_strategy = RuleBasedEval()

    result = start_solve(
        goal="写一篇关于人工智能的技术文章",
        think_llm=think_strategy,
        eval_llm=eval_strategy,
        constraints=Constraints(max_depth=4, max_tokens=150)
    )

    print(f"结果: {result}")
    print(f"Think调用次数: {think_strategy.call_count}")
    print(f"Eval调用次数: {eval_strategy.call_count}")
    print()


def demo_constraints():
    """演示约束机制"""
    print("=== 演示4: 约束机制 ===")

    # 深度约束演示
    print("4.1 深度约束演示")
    try:
        think_strategy = RuleBasedThink()
        eval_strategy = RuleBasedEval()

        result = start_solve(
            goal="复杂的递归任务",
            think_llm=think_strategy,
            eval_llm=eval_strategy,
            constraints=Constraints(max_depth=2)  # 很小的深度限制
        )
        print(f"结果（优雅降级）: {result}")
    except Exception as e:
        print(f"异常: {e}")

    # Token约束演示
    print("\n4.2 Token约束演示")
    try:
        think_strategy = RuleBasedThink()
        eval_strategy = RuleBasedEval()

        result = start_solve(
            goal="复杂的计算任务",
            think_llm=think_strategy,
            eval_llm=eval_strategy,
            constraints=Constraints(max_depth=10, max_tokens=20)  # 很小的token限制
        )
        print(f"结果（优雅降级）: {result}")
    except Exception as e:
        print(f"异常: {e}")

    # 时间约束演示
    print("\n4.3 时间约束演示")
    try:
        # 创建一个会模拟耗时的Think策略
        class SlowThink:
            def __call__(self, node, memory=None, tools=None):
                import time
                time.sleep(0.1)  # 模拟耗时操作
                return {"type": "TODO", "description": "耗时计划", "tokens_used": 5}

        think_strategy = SlowThink()
        eval_strategy = RuleBasedEval()

        result = start_solve(
            goal="耗时任务",
            think_llm=think_strategy,
            eval_llm=eval_strategy,
            constraints=Constraints(max_depth=10, max_tokens=1000, max_time=0.05)  # 很短的时间限制
        )
        print(f"结果（优雅降级）: {result}")
    except Exception as e:
        print(f"异常: {e}")

    print()


def demo_state_tracking():
    """演示状态跟踪和强不变式"""
    print("=== 演示5: 状态跟踪和强不变式 ===")

    class TrackingEval:
        """追踪状态变化的Eval策略"""
        def __init__(self):
            self.state_history = []

        def __call__(self, node: S, memory=None):
            # 记录每次调用时的状态
            self.state_history.append({
                'level': node.level,
                'goal': node.goal,
                'todo_length': len(node.todo),
                'done_count': len(node.done),
                'done_items': node.done.copy()
            })

            # 简单的执行逻辑
            if len(node.done) == 0:
                return {"type": "CALL", "description": "第一个子任务", "tokens_used": 5}
            elif len(node.done) == 1:
                return {"type": "CALL", "description": "第二个子任务", "tokens_used": 5}
            else:
                return {"type": "RETURN", "description": f"完成，包含{len(node.done)}个结果", "tokens_used": 3}

    think_strategy = RuleBasedThink()
    eval_strategy = TrackingEval()

    result = start_solve(
        goal="状态跟踪演示任务",
        think_llm=think_strategy,
        eval_llm=eval_strategy,
        constraints=Constraints(max_depth=4)
    )

    print(f"最终结果: {result}")
    print(f"\n状态变化历史:")
    for i, state in enumerate(eval_strategy.state_history):
        print(f"  调用{i+1}: Level={state['level']}, Goal='{state['goal'][:30]}...', "
              f"Done={state['done_count']}, Items={state['done_items']}")
    print()


def demo_error_handling():
    """演示错误处理"""
    print("=== 演示6: 错误处理 ===")

    # 无效Think响应
    print("6.1 无效Think响应")
    class BadThink:
        def __call__(self, node, memory=None, tools=None):
            return {"invalid": "response"}  # 缺少必需字段

    try:
        result = start_solve(
            goal="错误处理测试",
            think_llm=BadThink(),
            eval_llm=RuleBasedEval(),
            constraints=Constraints(max_depth=2)
        )
    except ValueError as e:
        print(f"  ✓ 捕获到预期错误: {e}")

    # 无效Eval响应
    print("\n6.2 无效Eval响应")
    class BadEval:
        def __call__(self, node, memory=None):
            return {"type": "INVALID_TYPE", "description": "test"}

    try:
        result = start_solve(
            goal="错误处理测试",
            think_llm=RuleBasedThink(),
            eval_llm=BadEval(),
            constraints=Constraints(max_depth=2)
        )
    except ValueError as e:
        print(f"  ✓ 捕获到预期错误: {e}")

    print()


def demo_json_serialization():
    """演示JSON序列化功能"""
    print("=== 演示7: JSON序列化 ===")

    # 创建状态结构
    root = S(goal="根任务")
    root.todo = "执行三个步骤的计划"
    root.done = ["步骤1完成", "步骤2完成"]

    child = S(goal="子任务", parent=root)
    child.todo = "子任务计划"
    child.done = ["子步骤完成"]

    # 序列化
    root_dict = root.to_dict()
    child_dict = child.to_dict()

    print("根节点序列化:")
    import json
    print(json.dumps(root_dict, indent=2, ensure_ascii=False))

    print("\n子节点序列化:")
    print(json.dumps(child_dict, indent=2, ensure_ascii=False))

    # 反序列化
    restored_root = S.from_dict(root_dict)
    restored_child = S.from_dict(child_dict, parent=restored_root)

    print(f"\n反序列化验证:")
    print(f"  根节点目标: {restored_root.goal}")
    print(f"  根节点完成项: {restored_root.done}")
    print(f"  子节点目标: {restored_child.goal}")
    print(f"  子节点层级: {restored_child.level}")
    print()


def main():
    """主函数，运行所有演示"""
    print("🧠 Fractal Thinkon 核心库演示程序")
    print("=" * 50)

    # 设置日志级别
    logger = setup_logger(level=10)  # DEBUG级别，显示详细信息

    try:
        demo_simple_task()
        demo_complex_task()
        demo_writing_task()
        demo_constraints()
        demo_state_tracking()
        demo_error_handling()
        demo_json_serialization()

        print("🎉 所有演示完成！")
        print("\n📚 使用说明:")
        print("  - 导入: from thinkon_core import S, start_solve, RuleBasedThink, RuleBasedEval")
        print("  - 创建自定义策略: 实现 ThinkLLM 和 EvalLLM 协议")
        print("  - 调用: result = start_solve(goal, think_llm, eval_llm, constraints)")
        print("  - 更多示例请参考本文件源码")

    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()