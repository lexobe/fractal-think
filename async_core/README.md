# Fractal Thinkon 异步执行核心 (Async Core)

## 概述

异步执行核心是 Fractal Thinkon 框架的重大架构升级，从同步递归模型转换为**显式栈式状态机 + 异步事件驱动**的执行模型。这一转换在保持向后兼容性的同时，提供了以下关键能力：

- 🔄 **状态快照与恢复**: 支持任务中断后的精确续跑
- ⚡ **异步并发执行**: 基于 asyncio 的高性能调度
- 📊 **高级约束管理**: 动态预算调整和智能降级
- 🔗 **向后兼容**: 无缝兼容现有同步 API
- 🛡️ **容错性**: 优雅处理约束违规和异常情况

## 架构对比

### 原同步递归模型
```
solve(node) → Think → [RETURN | TODO] → Eval → [CALL子任务 | RETURN] → 递归
```

### 新异步状态机模型
```
ExecutionFrame栈 → AsyncEngine状态机 → Think/Eval异步调度 → 快照/恢复
```

## 核心组件

### 1. 执行帧 (ExecutionFrame)
显式栈管理的核心数据结构：

```python
from async_core.frame import ExecutionFrame, FrameState

# 创建执行帧
frame = ExecutionFrame(
    frame_id="unique-id",
    node=S(goal="任务目标"),
    state=FrameState.CREATED
)

# 子任务管理
frame.add_subtask("子任务1")
frame.add_subtask("子任务2")

# 状态转换
frame.state = FrameState.THINKING
frame.state = FrameState.COMPLETED
```

### 2. 执行快照 (ExecutionSnapshot)
完整的可恢复状态：

```python
from async_core.snapshot import ExecutionSnapshot

# 创建快照
snapshot = ExecutionSnapshot(
    snapshot_id="snapshot-123",
    goal="主任务目标",
    frame_stack=[frame1, frame2],
    budget=ExecutionBudget(),
    global_token_usage=token_usage
)

# 序列化和持久化
json_str = snapshot.to_json()
snapshot.save_to_file("snapshot.json")

# 完整性验证
if snapshot.validate_integrity():
    print("快照状态完整")
```

### 3. 预算管理 (BudgetManager)
协程安全的约束检查：

```python
from async_core.common import ExecutionBudget, BudgetManager

# 创建预算
budget = ExecutionBudget(
    max_depth=10,
    max_tokens=5000,
    max_time=60.0
)

# 约束检查
manager = BudgetManager(budget)
constraint_error = await manager.check_constraints_async(node, current_tokens)
if constraint_error:
    print(f"约束触发: {constraint_error}")
```

### 4. 异步执行引擎 (AsyncExecutionEngine)
核心状态机实现：

```python
from async_core.engine import AsyncExecutionEngine, solve_async

# 直接异步求解
result = await solve_async(
    goal="解决复杂问题",
    think_llm=async_think_llm,
    eval_llm=async_eval_llm,
    budget=ExecutionBudget(max_tokens=10000)
)

# 从快照恢复
engine = AsyncExecutionEngine(context)
result = await engine.resume_from_snapshot(snapshot)
```

## 使用指南

### 基础异步使用

```python
import asyncio
from async_core.engine import solve_async
from async_core.common import ExecutionBudget

async def main():
    # 创建异步Think/Eval算子
    async_think = MyAsyncThinkLLM()
    async_eval = MyAsyncEvalLLM()

    # 异步求解
    result = await solve_async(
        goal="分析市场趋势并制定策略",
        think_llm=async_think,
        eval_llm=async_eval,
        budget=ExecutionBudget(
            max_depth=8,
            max_tokens=15000,
            max_time=120.0
        )
    )

    print(f"结果: {result.result}")
    print(f"状态: {result.status}")
    print(f"Token消耗: {result.token_usage.total}")

# 运行
asyncio.run(main())
```

### 同步兼容使用

```python
from async_core.sync_adapter import LegacyCompatibilityLayer

# 完全兼容原API
result = LegacyCompatibilityLayer.solve(
    goal="兼容性测试",
    think_llm=sync_think_llm,
    eval_llm=sync_eval_llm
)

# 或使用新适配器
from async_core.sync_adapter import solve_with_async_engine

result = solve_with_async_engine(
    goal="使用异步引擎的同步接口",
    think_llm=sync_think_llm,
    eval_llm=sync_eval_llm,
    budget=ExecutionBudget(max_tokens=5000)
)
```

### 快照和恢复

```python
from async_core.recovery import RecoveryManager, RecoveryMode

# 创建恢复管理器
recovery_manager = RecoveryManager("./snapshots")

# 保存快照
snapshot_path = recovery_manager.save_execution_snapshot(snapshot)

# 列出可恢复的快照
snapshots = recovery_manager.list_recoverable_snapshots()
for snap_info in snapshots:
    print(f"快照: {snap_info['snapshot_id']}, 目标: {snap_info['goal']}")

# 从快照恢复
async def recover_execution():
    snapshot = recovery_manager.load_execution_snapshot(snapshot_path)

    # 生成恢复计划
    plan = recovery_manager.create_recovery_plan(snapshot)
    print(f"恢复模式: {plan.mode}")
    print(f"成功率估计: {plan.estimated_success_rate}")

    # 执行恢复
    result = await recovery_manager.execute_recovery(
        snapshot=snapshot,
        think_llm=async_think_llm,
        eval_llm=async_eval_llm,
        plan=plan
    )
    return result
```

### 批量并发处理

```python
from async_core.sync_adapter import AsyncOptimizedLayer

# 批量异步求解
goals = [
    "分析用户需求",
    "设计系统架构",
    "制定实施计划"
]

results = await AsyncOptimizedLayer.solve_batch_async(
    goals=goals,
    think_llm=async_think_llm,
    eval_llm=async_eval_llm,
    budget_per_task=ExecutionBudget(max_tokens=3000),
    max_concurrent=3
)

for i, result in enumerate(results):
    print(f"任务{i+1}: {result.status} - {result.result[:50]}...")
```

### 自定义异步算子

```python
from async_core.interfaces import AsyncThinkLLM, AsyncEvalLLM

class CustomAsyncThinkLLM:
    """自定义异步Think算子"""

    async def __call__(self, node, memory=None, tools=None):
        # 异步处理逻辑
        await asyncio.sleep(0.1)  # 模拟API调用

        if len(node.goal) > 50:
            return {
                "type": "TODO",
                "description": f"分解长任务: {node.goal}",
                "tokens_used": 150
            }
        else:
            return {
                "type": "RETURN",
                "description": f"直接完成: {node.goal}",
                "tokens_used": 80
            }

class CustomAsyncEvalLLM:
    """自定义异步Eval算子"""

    async def __call__(self, node, memory=None):
        # 异步评估逻辑
        await asyncio.sleep(0.05)

        if len(node.done) >= 2:
            return {
                "type": "RETURN",
                "description": "已收集足够信息",
                "tokens_used": 60
            }
        else:
            return {
                "type": "CALL",
                "description": "需要更多信息",
                "tokens_used": 40
            }
```

## 迁移指南

### 从同步到异步的迁移步骤

1. **无需修改现有代码** - 同步API保持完全兼容
2. **可选择性升级** - 渐进式迁移到异步API
3. **性能优化** - 使用异步API获得更好性能

```python
# 原有代码继续工作
from thinkon_core import solve, start_solve
result = solve(goal, think_llm, eval_llm)

# 新异步代码
from async_core.engine import solve_async
result = await solve_async(goal, think_llm, eval_llm)

# 混合使用
from async_core.sync_adapter import solve_with_async_engine
result = solve_with_async_engine(goal, think_llm, eval_llm)  # 同步接口，异步引擎
```

### API映射表

| 原同步API | 新异步API | 兼容适配器 |
|----------|----------|----------|
| `solve()` | `solve_async()` | `LegacyCompatibilityLayer.solve()` |
| `solve_with_meta()` | `solve_async()` | `LegacyCompatibilityLayer.solve_with_meta()` |
| `start_solve()` | `solve_async()` | `LegacyCompatibilityLayer.start_solve()` |
| - | `solve_batch_async()` | - |
| - | `recover_from_file()` | - |

## 高级特性

### 自动快照

```python
from async_core.engine import AsyncExecutionContext

# 配置自动快照
context = AsyncExecutionContext(
    think_llm=async_think,
    eval_llm=async_eval,
    budget_manager=budget_manager,
    token_usage=token_usage,
    logger=logger,
    frame_stack=[],
    auto_snapshot=True,  # 启用自动快照
    snapshot_callback=lambda snapshot: save_snapshot(snapshot)
)

engine = AsyncExecutionEngine(context)
result = await engine.solve_async(goal, budget)
```

### 智能恢复策略

```python
from async_core.recovery import RecoveryStrategy, RecoveryMode

# 自定义恢复分析器
class CustomRecoveryAnalyzer(RecoveryAnalyzer):
    def _select_recovery_mode(self, progress_rate, constraint_triggered,
                             trigger_reason, failed_frames):
        # 自定义恢复模式选择逻辑
        if progress_rate > 0.8:
            return RecoveryMode.CONTINUE
        elif "ResourceLimitExceeded" in (constraint_triggered or ""):
            return RecoveryMode.PARTIAL_RETRY
        else:
            return RecoveryMode.RESTART
```

### 动态预算调整

```python
# 执行中动态调整预算
budget = ExecutionBudget(max_tokens=5000)
budget_manager = BudgetManager(budget)

# 根据执行情况调整
if high_complexity_detected:
    budget.adjust_limits(max_tokens=10000, max_time=120.0)
```

## 性能优化

### 并发最佳实践

```python
# 合理设置并发数
max_concurrent = min(cpu_count(), 5)  # 避免过多并发

# 使用信号量控制资源
semaphore = asyncio.Semaphore(max_concurrent)

async def controlled_solve(goal):
    async with semaphore:
        return await solve_async(goal, think_llm, eval_llm)
```

### 内存管理

```python
# 及时清理完成的帧
engine.cleanup_completed_frames()

# 定期清理旧快照
recovery_manager.cleanup_snapshots(max_age_days=7, max_count=100)
```

## 故障排除

### 常见问题

1. **ImportError**: 确保正确导入异步模块
2. **事件循环错误**: 使用正确的asyncio模式
3. **快照损坏**: 验证快照完整性
4. **内存泄漏**: 及时清理执行栈

### 调试工具

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 快照完整性检查
if not snapshot.validate_integrity():
    print("快照验证失败，详细信息:")
    print(snapshot.get_integrity_report())

# 执行统计
summary = snapshot.get_completion_summary()
print(f"完成帧数: {summary['completed_frames']}/{summary['total_frames']}")
```

## 规范示例（异步版）

本节展示如何使用异步执行核心复现 thinkon.md 规范第 4.1 节的"AI与艺术"短文分形流程。

### 运行规范示例

```bash
# 运行完整的异步规范示例
python3 examples/async_norm_example.py

# 运行单元测试验证流程
python3 test_async_norm.py
```

### 示例特性

- ✅ **完整分形流程**: 根据规范精确实现三段落撰写流程
- ✅ **异步执行**: 算子内部使用 `async def`，包含模拟延时
- ✅ **状态跟踪**: 完整的 todo/done 状态管理和转换
- ✅ **约束管理**: ExecutionBudget 限制和优雅降级
- ✅ **向后兼容**: 同时支持异步和同步接口调用

### 核心算子实现

#### AsyncThinkAIArt 算子

```python
class AsyncThinkAIArt:
    async def __call__(self, node: S, memory=None, tools=None):
        await asyncio.sleep(0.3)  # 模拟异步思考延时

        goal = node.goal

        # 根节点: 返回三段落计划
        if "AI与艺术" in goal and "短文" in goal:
            return {
                "type": "TODO",
                "description": """分成三个段落撰写：
 [] 生成艺术段落
 [] 辅助创作段落
 [] 艺术评论段落""",
                "tokens_used": 150
            }

        # 子节点: 根据具体目标返回段落内容
        elif "生成艺术段落" in goal:
            return {
                "type": "RETURN",
                "description": "AI正在革命性地改变艺术创作领域...",
                "tokens_used": 120
            }
        # ... 其他段落实现
```

#### AsyncEvalAIArt 算子

```python
class AsyncEvalAIArt:
    async def __call__(self, node: S, memory=None):
        await asyncio.sleep(0.2)  # 模拟异步评估延时

        done_count = len(node.done)

        if "AI与艺术" in node.goal:
            if done_count == 0:
                # 首启阶段: 创建第一个子任务
                return {
                    "type": "CALL",
                    "description": "生成艺术段落：含DALL·E例，要求不少于100字",
                    "tokens_used": 70
                }
            elif done_count == 1:
                # 续步阶段: 创建第二个子任务
                return {
                    "type": "CALL",
                    "description": "辅助创作段落：写一个能打动人的事例",
                    "tokens_used": 65
                }
            # ... 其他阶段处理
```

### 执行约束配置

```python
budget = ExecutionBudget(
    max_depth=5,      # 最大递归深度
    max_tokens=2000,  # 最大Token消耗
    max_time=10.0     # 最大执行时间（秒）
)
```

### 预期输出示例

```
🚀 启动异步规范示例：AI与艺术短文分形流程
============================================================
📋 执行约束: max_depth=5, max_tokens=2000, max_time=10.0s

[Think #1] 处理节点: 写一篇"AI与艺术"的短文（800–1200字）
[Think #1] 制定计划: 分成三个段落撰写...
[Eval #1] 评估节点: 写一篇"AI与艺术"的短文（800–1200字）
[Eval #1] 首启阶段 - 创建第一个子任务
[Think #2] 处理节点: 生成艺术段落：含DALL·E例...
[Think #2] 生成艺术段落内容
[Eval #2] 续步阶段 - 创建第二个子任务
...

📊 执行结果分析
============================================================
✅ 执行状态: SolveStatus.COMPLETED
⏱️  执行时间: 2.02s
📊 最大深度: 1
🔢 Token统计: 773
   - Think调用: 4 次，消耗: 480
   - Eval调用: 4 次，消耗: 293

📄 最终结果:
# AI与艺术：创新与思考的交融
[完整的三段落文章内容...]
```

### 验证测试

运行单元测试验证以下关键指标：

- ✅ `SolveResult.status == COMPLETED`
- ✅ `partial_results` 为空（完整完成）
- ✅ 异步执行按既定顺序生成三段落
- ✅ Token统计和算子调用次数正确
- ✅ 约束管理和优雅降级正常

### 与规范的一致性

本示例严格按照 thinkon.md 第 4.1 节规范实现：

1. **状态结构**: 完整的 S{goal, parent, todo, done} 管理
2. **强不变式**: 子层返回后必须先入档到 done
3. **算子语义**: Think 制定计划或返回结果，Eval 决定 CALL 或 RETURN
4. **执行流程**: 首启阶段 → 续步阶段的标准状态转换
5. **终止条件**: 有界递归和优雅降级机制

这个异步版本展示了如何在保持规范完全一致性的同时，利用异步执行核心获得更好的性能和可扩展性。

## 最佳实践

1. **渐进式迁移**: 先使用兼容层，再逐步迁移到纯异步
2. **合理设置预算**: 根据任务复杂度动态调整限制
3. **定期保存快照**: 在关键节点主动触发快照
4. **监控资源消耗**: 跟踪token使用和执行时间
5. **优雅降级**: 合理处理约束触发情况

## 测试

运行完整测试套件：

```bash
python -m pytest test_async_core.py -v
```

运行特定测试：

```bash
python -m pytest test_async_core.py::TestAsyncEngine::test_simple_async_solve -v
```

## 扩展开发

### 自定义状态机

```python
from async_core.frame import FrameState

class CustomFrameState(FrameState):
    CUSTOM_PROCESSING = "custom_processing"

# 在引擎中处理自定义状态
async def _handle_custom_processing_state(self, frame):
    # 自定义状态处理逻辑
    return FrameState.CONTINUING
```

### 插件机制

```python
class ExecutionPlugin:
    async def on_frame_created(self, frame):
        pass

    async def on_frame_completed(self, frame):
        pass

    async def on_constraint_triggered(self, constraint, frame):
        pass
```

这个异步执行核心为 Fractal Thinkon 提供了强大的执行能力，支持复杂任务的可靠执行和恢复。