"""
Fractal Thinkon 框架测试用例
验证实现是否符合 thinkon.md 规范
"""

import json
from thinkon import (
    S, ThinkLLM, EvalLLM, solve, start_solve, start_solve_simple, Constraints,
    parse_think_response, parse_eval_response,
    ReturnAction, PlanAction, CallAction, SolveResult, SolveStatus,
    DepthLimitExceeded, ResourceLimitExceeded, TimeLimitExceeded
)
from strategies import RuleBasedThink, RuleBasedEval


class TestThinkStrategy:
    """测试用的Think策略"""

    def __init__(self):
        self.call_count = 0

    def __call__(self, node: S, memory=None, tools=None):
        self.call_count += 1

        # 简单目标直接返回结果
        if "简单" in node.goal:
            return {"type": "RETURN", "description": f"简单任务已完成：{node.goal}"}

        # 复杂目标制定计划
        if "复杂" in node.goal:
            return {"type": "TODO", "description": "步骤1：分析需求；步骤2：设计方案；步骤3：实施方案"}

        # 默认返回结果
        return {"type": "RETURN", "description": f"任务完成：{node.goal}"}


class TestEvalStrategy:
    """测试用的Eval策略"""

    def __init__(self):
        self.call_count = 0

    def __call__(self, node: S, memory=None):
        self.call_count += 1

        # 如果有计划但没有执行记录，创建子任务
        if node.todo and len(node.done) == 0:
            return {"type": "CALL", "description": "执行步骤1：分析需求"}

        # 执行了一次后，继续下一步
        elif node.todo and len(node.done) == 1:
            return {"type": "CALL", "description": "执行步骤2：设计方案"}

        # 执行了两次后，完成任务
        elif node.todo and len(node.done) >= 2:
            return {"type": "RETURN", "description": f"复杂任务完成，包含{len(node.done)}个步骤的结果"}

        # 默认返回
        return {"type": "RETURN", "description": "任务已完成"}


def test_s_structure():
    """测试S数据结构"""
    print("=== 测试S数据结构 ===")

    # 创建根节点
    root = S(goal="根目标")
    assert root.goal == "根目标"
    assert root.parent is None
    assert root.todo == ""
    assert root.done == []
    print("✓ 根节点创建成功")

    # 创建子节点
    child = S(goal="子目标", parent=root)
    assert child.parent is root
    print("✓ 父子关系建立成功")

    # 测试状态更新
    root.todo = "制定计划"
    root.done.append("步骤1完成")
    assert root.todo == "制定计划"
    assert len(root.done) == 1
    print("✓ 状态更新成功")

    # 测试JSON转换
    data = root.to_dict()
    assert data["goal"] == "根目标"
    assert data["todo"] == "制定计划"
    assert len(data["done"]) == 1
    print("✓ JSON转换成功")


def test_json_parsing():
    """测试JSON解析功能"""
    print("\n=== 测试JSON解析 ===")

    # 测试Think响应解析
    think_return = {"type": "RETURN", "description": "任务完成"}
    action = parse_think_response(think_return)
    assert isinstance(action, ReturnAction)
    assert action.description == "任务完成"
    print("✓ Think RETURN解析成功")

    think_todo = {"type": "TODO", "description": "制定计划"}
    action = parse_think_response(think_todo)
    assert isinstance(action, PlanAction)
    assert action.description == "制定计划"
    print("✓ Think TODO解析成功")

    # 测试Eval响应解析
    eval_call = {"type": "CALL", "description": "执行子任务"}
    action = parse_eval_response(eval_call)
    assert isinstance(action, CallAction)
    assert action.description == "执行子任务"
    print("✓ Eval CALL解析成功")

    eval_return = {"type": "RETURN", "description": "任务完成"}
    action = parse_eval_response(eval_return)
    assert isinstance(action, ReturnAction)
    assert action.description == "任务完成"
    print("✓ Eval RETURN解析成功")

    # 测试错误情况
    try:
        parse_think_response({"type": "INVALID", "description": "test"})
        assert False, "应该抛出异常"
    except ValueError:
        print("✓ 无效type处理成功")


def test_simple_execution():
    """测试简单执行流程"""
    print("\n=== 测试简单执行 ===")

    think_strategy = TestThinkStrategy()
    eval_strategy = TestEvalStrategy()

    # 测试简单任务（直接返回）
    result = start_solve_simple(
        goal="简单任务",
        think_llm=think_strategy,
        eval_llm=eval_strategy,
        constraints=Constraints(max_depth=3)
    )

    assert "简单任务已完成" in result
    assert think_strategy.call_count == 1
    assert eval_strategy.call_count == 0  # 直接返回，不调用Eval
    print(f"✓ 简单任务执行成功：{result}")


def test_complex_execution():
    """测试复杂执行流程"""
    print("\n=== 测试复杂执行 ===")

    think_strategy = TestThinkStrategy()
    eval_strategy = TestEvalStrategy()

    # 测试复杂任务（需要计划和子任务）
    result = start_solve_simple(
        goal="复杂任务",
        think_llm=think_strategy,
        eval_llm=eval_strategy,
        constraints=Constraints(max_depth=5)
    )

    assert "复杂任务完成" in result
    assert think_strategy.call_count >= 3  # 根任务 + 2个子任务
    assert eval_strategy.call_count >= 3   # 至少3次Eval调用
    print(f"✓ 复杂任务执行成功：{result}")


def test_strong_invariant():
    """测试强不变式：子层返回后必须先入档到done"""
    print("\n=== 测试强不变式 ===")

    class InvariantTestEval:
        def __init__(self):
            self.states_seen = []

        def __call__(self, node: S, memory=None):
            # 记录每次调用时的done状态
            self.states_seen.append(len(node.done))

            if len(node.done) == 0:
                return {"type": "CALL", "description": "第一个子任务"}
            elif len(node.done) == 1:
                return {"type": "CALL", "description": "第二个子任务"}
            else:
                return {"type": "RETURN", "description": "任务完成"}

    class InvariantTestThink:
        def __call__(self, node: S, memory=None, tools=None):
            if node.goal == "根任务":
                return {"type": "TODO", "description": "需要执行多个子任务"}
            else:
                return {"type": "RETURN", "description": f"子任务完成：{node.goal}"}

    think_strategy = InvariantTestThink()
    eval_strategy = InvariantTestEval()

    result = start_solve_simple(
        goal="根任务",
        think_llm=think_strategy,
        eval_llm=eval_strategy,
        constraints=Constraints(max_depth=3)
    )

    # 验证强不变式：done状态应该递增
    assert eval_strategy.states_seen == [0, 1, 2]
    print("✓ 强不变式验证成功：done状态按预期递增")


def test_constraints():
    """测试约束机制"""
    print("\n=== 测试约束机制 ===")

    class InfiniteThink:
        def __call__(self, node: S, memory=None, tools=None):
            return {"type": "TODO", "description": "永远制定计划"}

    class InfiniteEval:
        def __call__(self, node: S, memory=None):
            return {"type": "CALL", "description": "永远调用子任务"}

    think_strategy = InfiniteThink()
    eval_strategy = InfiniteEval()

    # 测试深度约束 - 现在返回优雅降级结果而不是抛出异常
    solve_result = start_solve(
        goal="无限递归测试",
        think_llm=think_strategy,
        eval_llm=eval_strategy,
        constraints=Constraints(max_depth=3)
    )
    # 检查是否是优雅降级的结果
    assert solve_result.is_degraded, f"Expected degraded status, got: {solve_result.status}"
    assert "Depth" in str(solve_result.constraint_triggered), f"Expected depth constraint, got: {solve_result.constraint_triggered}"
    print("✓ 深度约束检查成功（优雅降级）")

    # 测试时间约束 - 也是优雅降级
    solve_result = start_solve(
        goal="无限递归测试",
        think_llm=think_strategy,
        eval_llm=eval_strategy,
        constraints=Constraints(max_depth=100, max_time=0.001)  # 更短的时间限制
    )
    # 检查是否是优雅降级的结果（可能是时间或深度约束触发）
    assert solve_result.is_degraded, f"Expected degraded status, got: {solve_result.status}"
    print("✓ 约束检查成功（优雅降级）")


def run_all_tests():
    """运行所有测试"""
    print("开始 Fractal Thinkon 框架测试...\n")

    try:
        test_s_structure()
        test_json_parsing()
        test_simple_execution()
        test_complex_execution()
        test_strong_invariant()
        test_constraints()

        print("\n🎉 所有测试通过！实现符合 thinkon.md 规范")

    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        raise


if __name__ == "__main__":
    run_all_tests()