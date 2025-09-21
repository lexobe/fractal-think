# Fractal Think

高效的异步分形思考执行框架，支持复杂问题的递归分解和并行处理。

## 特性

- **异步执行引擎**：基于现代异步架构，支持高并发和高效执行
- **智能计划解析**：支持多种计划格式（[]、1.、步骤:、-）自动识别
- **预算约束管理**：深度、Token、时间三重约束保证系统终止性
- **显式状态机**：基于ExecutionFrame的显式栈式状态管理
- **约束终止**：约束违反时立即终止执行并抛出异常，确保资源边界的严格执行
- **向后兼容**：提供同步适配器支持原有API
- **无外部依赖**：纯Python标准库实现，易于集成

## 安装

```bash
git clone https://github.com/your-repo/fractal-think.git
cd fractal-think
```

## 快速开始

### 异步API（推荐）

```python
import asyncio
from src.fractal_think import solve_async, ExecutionBudget

async def my_think(node, memory=None, tools=None):
    """异步Think算子"""
    if len(node.goal) < 20:
        return {"type": "RETURN", "description": f"完成: {node.goal}", "tokens_used": 50}
    else:
        return {
            "type": "TODO",
            "description": "[] 步骤1：分析任务\n[] 步骤2：执行计划\n[] 步骤3：总结结果",
            "tokens_used": 100
        }

async def my_eval(node, memory=None):
    """异步Eval算子"""
    if len(node.done) > 0:
        return {"type": "RETURN", "description": f"任务完成: {node.goal}", "tokens_used": 30}
    else:
        return {"type": "CALL", "description": f"分析: {node.goal}", "tokens_used": 60}

async def main():
    result = await solve_async(
        goal="写一篇关于AI与艺术的短文",
        think_llm=my_think,
        eval_llm=my_eval,
        budget=ExecutionBudget(max_depth=5, max_tokens=2000, max_time=30.0)
    )

    print(f"状态: {result.status}")
    print(f"结果: {result.result}")
    print(f"Token消耗: {result.token_usage.total}")

asyncio.run(main())
```

### 运行示例

```bash
# 运行异步规范示例
python -m src.fractal_think.examples.norm_async

# 运行简单测试
python tests/test_simple.py

# 运行pytest测试（需要先安装依赖）
pip install -r requirements-dev.txt
pytest tests/
```

## 核心概念

### 状态结构 S

```python
from src.fractal_think import S

# 创建状态节点
node = S(goal="解决具体问题")
node.todo = "执行计划文本"
node.done = ["步骤1结果", "步骤2结果"]
print(f"当前层级: {node.level}")
```

### 异步算子接口

**AsyncThinkLLM**：决定计划制定或直接返回
```python
async def my_think(node, memory=None, tools=None):
    """异步Think算子"""
    return {
        "type": "TODO",  # 或 "RETURN"
        "description": "自然语言计划或结果",
        "tokens_used": 10  # 可选
    }
```

**AsyncEvalLLM**：基于状态决定子任务调用或收束
```python
async def my_eval(node, memory=None):
    """异步Eval算子"""
    return {
        "type": "CALL",  # 或 "RETURN"
        "description": "子目标描述或最终结果",
        "tokens_used": 5   # 可选
    }
```

### 预算约束

```python
from src.fractal_think import ExecutionBudget

budget = ExecutionBudget(
    max_depth=10,      # 最大递归深度
    max_tokens=1000,   # 最大token消耗
    max_time=60.0      # 最大执行时间（秒）
)
```


## 架构原理

基于异步状态机的分形执行引擎：

1. **异步执行引擎**：`AsyncExecutionEngine` 管理执行状态机主循环
2. **ExecutionFrame栈**：显式栈管理，每个Frame对应一个S节点的完整执行
3. **状态转换**：THINK → PLANNING → FIRST_EVAL → EVAL → RETURNING/FAILED
4. **智能计划解析**：自动识别多种计划格式并创建子任务
5. **预算管理**：`BudgetManager` 实时检查深度、Token、时间约束
6. **Token追踪**：`UnifiedTokenUsage` 统计Think/Eval调用和消耗

详细技术规范请参考 [thinkon.md](thinkon.md)。

## 项目结构

```
fractal-think/
├── src/fractal_think/          # 核心包
│   ├── __init__.py            # 主要API导出
│   ├── engine.py              # 异步执行引擎
│   ├── types.py               # 核心数据结构
│   ├── frame.py               # 执行帧管理
│   ├── common.py              # 预算和工具组件
│   ├── interfaces.py          # 算子协议定义
│   ├── sync_adapter.py        # 同步兼容层(仅向后兼容)
│   └── examples/              # 示例模块
│       ├── __init__.py
│       ├── specification_operators.py  # 规范版算子
│       ├── mock_operators.py  # 测试用Mock算子
│       └── norm_async.py      # 异步规范示例
├── tests/
│   ├── __init__.py
│   ├── test_async_core.py     # 核心功能测试
│   └── test_simple.py         # 简单功能测试
├── thinkon.md                 # 技术规范文档
└── README.md                  # 本文件
```

## API参考

### 主要函数

#### `solve_async()` - 异步求解

```python
async def solve_async(
    goal: str,
    think_llm: Union[AsyncThinkLLM, ThinkLLM],
    eval_llm: Union[AsyncEvalLLM, EvalLLM],
    budget: Optional[ExecutionBudget] = None,
    logger: Optional[UnifiedLogger] = None,
    memory: Any = None,
    tools: Any = None
) -> SolveResult
```

### 核心数据结构

#### `SolveResult`

```python
@dataclass
class SolveResult:
    status: SolveStatus           # COMPLETED/FAILED
    result: str                   # 最终结果
    token_usage: TokenUsage       # Token消耗统计
    execution_time: float         # 执行时间
    max_depth_reached: int        # 最大深度
    constraint_triggered: Optional[str] = None  # 触发的约束（兼容字段，约束违反时会抛出异常）
```

### 自定义异步算子

```python
from src.fractal_think import AsyncThinkLLM, AsyncEvalLLM

class MyAsyncThink:
    async def __call__(self, node, memory=None, tools=None):
        # 异步自定义逻辑
        if "简单" in node.goal:
            return {"type": "RETURN", "description": "直接完成"}
        else:
            return {"type": "TODO", "description": "[] 分析\n[] 执行\n[] 总结"}

class MyAsyncEval:
    async def __call__(self, node, memory=None):
        # 基于 node.todo 和 node.done 决策
        if len(node.done) >= 3:
            return {"type": "RETURN", "description": "任务完成"}
        else:
            return {"type": "CALL", "description": f"执行步骤{len(node.done)+1}"}
```

### 智能计划解析

框架支持多种计划格式的自动识别：

```python
# 支持的计划格式
plan_formats = """
[] 任务1                    # 规范格式
[] 任务2
1. 步骤一                   # 数字格式
2. 步骤二
- 项目A                    # 列表格式
- 项目B
步骤1：分析                 # 步骤格式
步骤2：执行
"""
```

### 执行状态监控

```python
import asyncio
from src.fractal_think import solve_async, ExecutionBudget, SolveStatus
from src.fractal_think.types import MaxDepthExceeded, ResourceExhausted, ExecutionTimeout, ConstraintViolationError

async def main():
    try:
        result = await solve_async(goal, think_llm, eval_llm, budget)

        # 状态检查
        if result.status == SolveStatus.COMPLETED:
            print(f"✅ 任务完成: {result.result}")
        else:
            print(f"❌ 执行失败: {result.result}")

    except MaxDepthExceeded as e:
        print(f"🚫 深度约束违反: {e}")
    except ResourceExhausted as e:
        print(f"🚫 资源约束违反: {e}")
    except ExecutionTimeout as e:
        print(f"🚫 时间约束违反: {e}")
    except ConstraintViolationError as e:
        print(f"🚫 约束违反: {e}")
        return  # 约束违反时无结果统计

    # 性能统计（仅在成功执行后）
    print(f"📊 执行统计:")
    print(f"  - 时间: {result.execution_time:.2f}s")
    print(f"  - 深度: {result.max_depth_reached}")
    print(f"  - Token: {result.token_usage.total}")
    print(f"  - Think调用: {result.token_usage.think_calls}")
    print(f"  - Eval调用: {result.token_usage.eval_calls}")
```

### 测试和开发

```python
# 使用规范版算子进行开发
from src.fractal_think.examples.specification_operators import (
    SpecificationAIArtThink, SpecificationAIArtEval
)

async def test_specification_example():
    """测试规范示例"""
    think_llm = SpecificationAIArtThink(verbose=True)
    eval_llm = SpecificationAIArtEval(verbose=True)

    result = await solve_async(
        goal='写一篇"AI与艺术"的短文',
        think_llm=think_llm,
        eval_llm=eval_llm,
        budget=ExecutionBudget(max_depth=3, max_tokens=1000)
    )

    assert result.status == SolveStatus.COMPLETED
    assert "AI与艺术" in result.result
    return result

# 运行测试
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_specification_example())
```

## 开发

### 安装开发依赖

```bash
pip install -r requirements-dev.txt
```

### 运行测试

> **⚠️ 重要提醒**: pytest并未预装，使用前必须先安装开发依赖！

**步骤1：安装开发依赖**
```bash
pip install -r requirements-dev.txt
```

**步骤2：选择测试方式**
```bash
# 方式A：使用pytest运行所有测试
pytest tests/

# 方式B：运行特定测试文件
pytest tests/test_simple.py

# 方式C：显示详细输出
pytest tests/test_simple.py -v -s

# 方式D：直接运行测试脚本（无需pytest）
python tests/test_simple.py
```

### 项目结构说明

- `src/fractal_think/` - 核心异步框架
- `src/fractal_think/examples/` - 示例和Mock算子
- `tests/` - 测试套件
- `requirements-dev.txt` - 开发依赖

## 附录

### 同步适配器（向后兼容）

> **注意**: 同步适配器仅用于向后兼容，新项目强烈推荐使用异步API。

```python
from src.fractal_think.sync_adapter import solve_with_async_engine

# 仅用于兼容旧版同步算子
result = solve_with_async_engine(
    goal="任务目标",
    think_llm=legacy_sync_think,  # 遗留同步算子
    eval_llm=legacy_sync_eval,    # 遗留同步算子
    budget=ExecutionBudget()
)
```

## 许可证

MIT License

---

*基于 thinkon.md 规范实现的分形思考递归推理框架*