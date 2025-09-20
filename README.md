# Fractal Thinkon

分形思考（Fractal Thinkon）框架的 Python 核心库实现，基于 [thinkon.md](thinkon.md) 技术规范 v1.0.0。

## 特性

- **最小化递归框架**：通过 Think、Eval 两个算子和状态结构 S 实现完整的递归推理能力
- **严格规范遵循**：完全按照 thinkon.md 规范实现，确保语义一致性和可预测性
- **无外部依赖**：纯 Python 标准库实现，易于集成和部署
- **终止约束机制**：深度、资源、时间三重约束保证系统终止性
- **优雅降级**：约束触发时返回部分结果而非完全失败
- **统一日志系统**：支持调试和监控的完整执行追踪
- **内置示例**：提供规则引擎 Think/Eval 实现用于演示和测试

## 快速开始

### 基本用法

```python
from thinkon_core import start_solve, RuleBasedThink, RuleBasedEval, Constraints

# 创建算子实例
think_strategy = RuleBasedThink()
eval_strategy = RuleBasedEval()

# 执行简单任务
result = start_solve(
    goal="写一篇关于AI的技术文章",
    think_llm=think_strategy,
    eval_llm=eval_strategy,
    constraints=Constraints(max_depth=5)
)

print(result)
```

### 运行演示

```bash
# 运行完整演示程序
python3 examples/demo.py

# 或者作为模块运行
python3 -m examples.demo
```

## 核心概念

### 状态结构 S

```python
from thinkon_core import S

# 创建状态节点
node = S(goal="解决具体问题")
node.todo = "执行计划文本"
node.done = ["步骤1结果", "步骤2结果"]
```

### 算子接口

**Think 算子**：决定计划制定或直接返回
```python
class CustomThink:
    def __call__(self, node, memory=None, tools=None):
        return {
            "type": "TODO",  # 或 "RETURN"
            "description": "自然语言计划或结果",
            "tokens_used": 10  # 可选
        }
```

**Eval 算子**：基于状态决定子任务调用或收束
```python
class CustomEval:
    def __call__(self, node, memory=None):
        return {
            "type": "CALL",  # 或 "RETURN"
            "description": "子目标描述或最终结果",
            "tokens_used": 5   # 可选
        }
```

### 约束配置

```python
from thinkon_core import Constraints

constraints = Constraints(
    max_depth=10,      # 最大递归深度（根节点level=0，触发条件：node.level >= max_depth）
    max_tokens=1000,   # 最大token消耗（全局累计，包括所有递归调用）
    max_time=60.0      # 最大执行时间（秒，从start_solve开始计时）
)
```

## 架构原理

框架实现了规范中定义的核心控制流：

1. **Think阶段**：决定制定计划（TODO）或直接返回结果（RETURN）
2. **Plan_made**：如果Think返回计划，写入 `S.todo`
3. **首启阶段**：Eval基于当前状态决定首个子任务
4. **While循环**：处理连续的子任务调用
5. **强不变式**：每个子任务结果必须先入档到 `S.done`
6. **Return收束**：Eval决定任务完成并返回最终结果

详细技术规范请参考 [thinkon.md](thinkon.md)。

## 文件结构

```
fractal-think/
├── thinkon_core.py      # 核心库实现
├── thinkon_core_old.py  # 旧版本实现（保留）
├── examples/
│   ├── demo.py          # 演示脚本
│   └── __init__.py
├── thinkon.md           # 技术规范文档
└── README.md            # 本文件
```

## 开发指南

### 自定义算子

继承协议或直接实现 `__call__` 方法：

```python
from thinkon_core import ThinkLLM, EvalLLM

class MyThink:
    def __call__(self, node, memory=None, tools=None):
        # 自定义逻辑
        if "简单" in node.goal:
            return {"type": "RETURN", "description": "直接完成"}
        else:
            return {"type": "TODO", "description": "制定详细计划"}

class MyEval:
    def __call__(self, node, memory=None):
        # 基于 node.todo 和 node.done 决策
        if len(node.done) >= 3:
            return {"type": "RETURN", "description": "任务完成"}
        else:
            return {"type": "CALL", "description": f"执行步骤{len(node.done)+1}"}
```

### 错误处理

框架提供完整的错误处理和约束检查：

```python
from thinkon_core import DepthLimitExceeded, ResourceLimitExceeded, TimeLimitExceeded

try:
    result = start_solve(goal, think_llm, eval_llm, constraints)
except (DepthLimitExceeded, ResourceLimitExceeded, TimeLimitExceeded) as e:
    print(f"约束触发，优雅降级：{e}")
```

### JSON序列化

状态结构支持完整的序列化：

```python
# 序列化
state_dict = node.to_dict()

# 反序列化
restored_node = S.from_dict(state_dict, parent=parent_node)
```

## API 参考

### solve vs solve_with_meta

框架提供两个核心API满足不同需求：

#### `solve()` - 向后兼容API

完全兼容 thinkon.md 规范 2.1 节，返回自然语言字符串：

```python
from thinkon_core import solve, S

# 创建状态节点
node = S(goal="解决具体问题")

# 规范兼容的调用方式
result = solve(
    node=node,
    think_llm=my_think_strategy,
    eval_llm=my_eval_strategy,
    constraints=Constraints(max_depth=5),
    logger=my_logger
)

print(f"结果: {result}")  # 纯字符串结果
```

**适用场景**：
- 现有代码迁移
- 只需要结果文本
- 严格遵循 thinkon.md 规范
- 与其他基于规范的实现集成

#### `solve_with_meta()` - 增强API

返回完整的 `SolveResult` 对象，包含执行元信息：

```python
from thinkon_core import solve_with_meta, S

node = S(goal="解决具体问题")

result = solve_with_meta(
    node=node,
    think_llm=my_think_strategy,
    eval_llm=my_eval_strategy,
    constraints=Constraints(max_depth=5),
    logger=my_logger
)

print(f"结果: {result.result}")
print(f"状态: {result.status.value}")
print(f"Token消耗: {result.token_usage.total}")
print(f"执行时间: {result.execution_time:.2f}s")
print(f"最大深度: {result.max_depth_reached}")

# 降级情况的详细信息
if result.is_degraded:
    print(f"约束触发: {result.constraint_triggered}")
    print(f"失败路径: {' -> '.join(result.failure_path)}")
    print(f"失败层级: {result.failure_level}")
    print(f"部分结果: {result.partial_results}")
```

**适用场景**：
- 需要执行统计信息
- 错误诊断和调试
- 性能监控
- 优雅降级处理
- 复杂应用集成

### SolveResult 详解

```python
@dataclass
class SolveResult:
    status: SolveStatus           # 执行状态（COMPLETED/DEGRADED/FAILED）
    result: str                   # 最终结果描述
    token_usage: TokenUsage       # Token消耗统计
    execution_time: float         # 执行时间（秒）
    max_depth_reached: int        # 达到的最大递归深度
    constraint_triggered: str     # 触发的约束类型（如有）
    partial_results: List[str]    # 部分结果列表
    # 降级上下文信息
    failure_path: List[str]       # 失败路径（根到触发节点的目标序列）
    failure_level: int            # 触发约束的具体层级
    failure_node_goal: str        # 触发约束的节点目标
    failure_node_done: List[str]  # 触发约束时该节点已完成的步骤
```

### 状态语义

框架采用纯控制流状态，不判断业务成功与否：

- **COMPLETED**: 控制流正常走到终点（Think直接返回或Eval收束）
- **DEGRADED**: 约束触发，返回部分结果
- **FAILED**: 异常失败（代码错误等）

### 高级使用示例

```python
# 监控和调试模式
result = solve_with_meta(node, think_llm, eval_llm, constraints)

if result.is_completed:
    print(f"✓ 任务完成: {result.result}")
elif result.is_degraded:
    print(f"⚠ 优雅降级: {result.constraint_triggered}")
    print(f"失败路径: {' -> '.join(result.failure_path)}")
    print(f"已完成步骤: {len(result.partial_results)}")
    # 可以基于上下文信息进行重试或恢复
else:
    print(f"✗ 执行失败: {result.result}")

# 性能统计
print(f"Token统计: Think={result.token_usage.think_tokens}, "
      f"Eval={result.token_usage.eval_tokens}, "
      f"总计={result.token_usage.total}")
```

### 约束参数详解

```python
# 约束配置的详细语义
constraints = Constraints(
    max_depth=10,      # 最大递归深度
                       # - 根节点level=0，子节点递增
                       # - 触发条件：node.level >= max_depth
                       # - 推荐值：3-10，取决于问题复杂度

    max_tokens=1000,   # 最大token消耗
                       # - 全局累计，包括所有递归调用的token
                       # - 每次Think/Eval调用后检查
                       # - 算子可通过"tokens_used"字段报告消耗

    max_time=60.0      # 最大执行时间（秒）
                       # - 从start_solve调用开始计时
                       # - 每次约束检查时验证
                       # - 包括Think/Eval/递归调用的总时间
)
```

### 响应格式规范

使用 TypedDict 可获得更好的类型提示：

```python
from thinkon_core import ThinkResponse, EvalResponse

class MyThink:
    def __call__(self, node, memory=None, tools=None) -> ThinkResponse:
        return {
            "type": "TODO",
            "description": "详细计划文本",
            "tokens_used": 15  # 可选字段，用于资源跟踪
        }

class MyEval:
    def __call__(self, node, memory=None) -> EvalResponse:
        return {
            "type": "CALL",
            "description": "子目标描述",
            "tokens_used": 8   # 可选字段
        }
```

### 日志配置

```python
import logging
from thinkon_core import setup_logger

# 基础配置
logger = setup_logger('my_thinkon', logging.DEBUG)

# 避免重复输出
logger = setup_logger('my_thinkon', logging.INFO, propagate=False)

# 使用自定义logger
import logging
custom_logger = logging.getLogger('custom')
# ... 配置handler等 ...

result = start_solve(
    goal="任务",
    think_llm=think_strategy,
    eval_llm=eval_strategy,
    logger=custom_logger
)
```

### 异常处理策略

```python
from thinkon_core import (
    DepthLimitExceeded, ResourceLimitExceeded, TimeLimitExceeded
)

try:
    result = start_solve(goal, think_llm, eval_llm, constraints)
    print(f"任务完成: {result}")
except DepthLimitExceeded as e:
    print(f"递归层数过深: {e}")
except ResourceLimitExceeded as e:
    print(f"资源消耗超限: {e}")
except TimeLimitExceeded as e:
    print(f"执行超时: {e}")
```

**注意**：在正常使用中，`start_solve` 会自动捕获这些异常并返回优雅降级结果，只有在直接调用 `solve` 函数时才需要手动处理异常。

## 许可证

MIT License

---

*基于 thinkon.md 规范实现的分形思考递归推理框架*