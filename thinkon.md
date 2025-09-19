# 分形思考（Fractal Thinkon）技术规范

**状态**: 规范草案  
**版本**: 1.0.0
**类型**: 信息规范  
**作者**: Fractal Think Working Group  
**关键字**: 本文档中的关键字"MUST"、"MUST NOT"、"REQUIRED"、"SHALL"、"SHALL NOT"、"SHOULD"、"SHOULD NOT"、"RECOMMENDED"、"MAY"和"OPTIONAL"按照RFC 2119进行解释。

---

## Abstract

本文档定义了分形思考（Fractal Thinkon）框架的技术规范。该框架通过两个核心算子（Think、Eval）和一个状态结构（S），以最小化方式实现递归推理系统。本规范为实现者提供无歧义的技术定义，确保不同实现间的完全兼容性。

核心设计：以最小结构刻画"计划—执行—评估—收束"的递归工作流，强制执行严格的状态管理不变式，保证系统的可预测性和可审计性。

---

## 1. 核心定义

### 1.1 设计原则

分形思考框架基于三个核心组件实现完整的递归推理能力：

* **Think算子**：负责计划制定和工具调用，返回执行结果或自然语言计划
* **Eval算子**：负责状态评估和子任务调度，基于当前上下文决定下一步行动  
* **状态结构S**：提供完整的递归执行上下文和历史追踪能力

### 1.2 状态结构

$$S=\{\ \textbf{goal}:g,\ \textbf{parent}:S_p\ \text{或}\ null,\ \textbf{todo}:T,\ \textbf{done}:D\ \}$$

这四个字段构成了完整的递归执行上下文：

* **goal**：当前层目的，定义本层的求解目标
* **parent**：父节点引用（根节点为null），形成递归调用栈
* **todo**：纯自然语言计划文本，支持活性计划描述
* **done**：完成历史列表$R_{n+1}^{1..k}$，每个条目存储子层返回的结果字符串（上标表示子任务序号）

**架构约束**：`todo`的内容总由Think算子产生，Eval算子只采用不修改。此约束定义了计划制定的数据所有权，确保架构的单一职责原则。

### 1.3 Think算子

$$
\mathrm{Think}_{LLM}(Prompt_t,S_n,M,\mathsf{Tools})
\ \Rightarrow\
\textbf{Return\_up}(R_n)\ \big|\ \textbf{Plan\_made}(T_n)
$$

Think算子在每个新节点创建时激活，作为该层的第一个操作。工具调用权限仅限于Think算子，包括但不限于搜索、代码执行、API调用等外部操作。

**标准输出格式**：
```json
{
    "type": "RETURN" | "TODO",
    "description": "自然语言内容"
}
```

* **type="RETURN"**（Return\_up）：直接给出本层执行结果，终止当前递归分支
* **type="TODO"**（Plan\_made）：制定自然语言计划，写入S_n.todo，激活Eval流程

### 1.4 Eval算子

**状态更新机制**：
$$S_n.done\ \ += x, \quad x\in\{\varnothing,\ R_{n+1}\}$$

当x=∅时（首启阶段），S_n.done保持不变；当x=R_{n+1}时（续步阶段），执行S_n.done.append(R_{n+1})。

**算子定义**：
$$
\mathrm{Eval}_{LLM}(Prompt_e,S_n,M)
\ \Rightarrow\
\textbf{Call}(S_{n+1})\ \big|\ \textbf{Return\_up}(R_n)
$$

Eval算子通过访问S_n.todo和S_n.done的完整状态感知当前阶段，x值通过状态更新机制变相传递给Eval_LLM。

**标准输出格式**：
```json
{
    "type": "RETURN" | "CALL", 
    "description": "自然语言描述"
}
```

* **type="CALL"**（Call）：创建子任务，description包含子目标描述
* **type="RETURN"**（Return\_up）：认定本层完成，description为完成结果

**工作模式**：
* **首启阶段x=∅**：基于S_n.todo/done决定Call(S_{n+1})或Return\_up(R_n)
* **续步阶段x=R_{n+1}**：先更新done，再基于新状态决定后续行动
* **统一原则**：Call包含所有子任务操作（下一项/重做/修复），Return\_up表示层完成

---

## 2. 操作语义

### 2.1 递归控制流

记`Solve(S_n)`为核心求解流程：

```
1. think_result = Think_LLM(P_t, S_n, M, Tools)
2. if think_result.type == "RETURN": return think_result.description
3. S_n.todo = think_result.description  // Plan_made
4. 首启阶段 (x=∅)：eval_result = Eval_LLM(P_e, S_n, M)
5. while eval_result.type == "CALL":
     S_child = create_child_node(eval_result.description)
     R_child = Solve(S_child)  // 递归调用
     S_n.done.append(R_child)  // 强不变式：先入档
     续步阶段 (x=R_child)：eval_result = Eval_LLM(P_e, S_n, M)
6. return eval_result.description  // type == "RETURN"
```

### 2.2 形式化语义规则

**Think操作语义**：

Think返回结果：
$$\frac{\mathrm{Think}_{LLM}(Prompt_t,S_n,M,\mathsf{Tools})=\textbf{Return\_up}(R_n)}{\uparrow R_n}$$

Think返回计划：
$$\frac{\mathrm{Think}_{LLM}(Prompt_t,S_n,M,\mathsf{Tools})=\textbf{Plan\_made}(T_n)}{S_n.\text{todo}\gets T_n}$$

**Eval操作语义**：

首启阶段（无前置返回结果）：
$$\frac{x=\varnothing}
{\mathrm{Eval}_{LLM}(Prompt_e,S_n,M)=\textbf{Call}(S_{n+1})\ \ \text{or}\ \ \textbf{Return\_up}(R_n)}$$

续步阶段（强不变式执行）：
$$\frac{\mathrm{Solve}(S_{n+1})\leadsto \textbf{Return\_up}(R_{n+1})}
{S_n.done{+}{=}R_{n+1};\ \ \ \mathrm{Eval}_{LLM}(Prompt_e,S_n,M):
\ \textbf{Call}(S_{n+1})\ \text{or}\ \textbf{Return\_up}(R_n)}$$

### 2.3 不变式与约束

**强不变式**：任何子层Return\_up(R_{n+1})后，父层必须先将R_{n+1}追加到done，再进行下一步决策。

此不变式确保：执行历史完整记录、审计和回放能力、为失败重试提供完整上下文。数学表示见上述续步阶段语义规则。

### 2.4 终止性保证

**有界递归原理**：
$$\exists (D, R, T) : \text{Solve}(S_n) \text{ 在约束 } (depth \leq D, resources \leq R, time \leq T) \text{ 下终止}$$

递归推理系统必须在有界约束下运行以确保终止性。实现者应提供以下三类约束机制：

**深度约束**：限制递归调用的最大层数，防止无限递归导致的栈溢出

**资源约束**：限制计算资源的最大消耗，确保系统运行的成本可控性，采用LLM的token消耗进行量化

**时间约束**：限制单次执行的最大时长，保证系统的响应性和可用性

**优雅降级原则**：约束违反时系统应执行优雅降级，返回已完成的部分结果而非抛出异常，确保用户获得有价值的输出。

具体约束值和实现策略由部署环境和应用需求确定，超出本规范范围。

---

## 3. 实现规范

### 3.1 JSON通信协议

框架采用统一的JSON格式确保LLM输出的结构化和可解析性：

**设计原则**：
- 最少字段：仅type和description两个必需字段
- 最大自由度：description内容完全由LLM决定
- 类型安全：type字段确保解析的确定性

**解析实现**：
```python
def parse_think_response(response):
    if response["type"] == "RETURN":
        return ReturnAction(response["description"])
    elif response["type"] == "TODO":
        return PlanAction(response["description"])
    else:
        raise ValueError("Invalid Think response type")

def parse_eval_response(response):
    if response["type"] == "RETURN":
        return ReturnAction(response["description"])
    elif response["type"] == "CALL":
        return CallAction(response["description"])
    else:
        raise ValueError("Invalid Eval response type")
```

### 3.2 算子实现要求

**Think算子接口约定**：
- 必须处理Tools参数中的所有工具调用
- 必须基于S_n.goal制定合适计划或直接给出结果
- 可访问Memory进行上下文学习和经验积累

**Eval算子接口约定**：
- 必须基于S_n.todo和S_n.done的完整状态做决策
- 子目标描述必须足够清晰以创建新的S_{n+1}.goal
- 必须能识别失败情况并决定重试、修复或终止策略

**子目标传递机制**：Eval返回的`{"type": "CALL", "description": "子目标描述"}`中，description直接作为子节点的goal，确保语义的清晰传递。

### 3.3 错误处理与失败恢复

**失败检测**：子层可在结果文本中明确表示失败及其原因，通过自然语言描述失败状态和错误信息。

**重做机制**：
- **直接重做**：Eval基于done中的失败结果，重新生成相同类型的子任务
- **修复重做**：先Call一个"补前置条件"的修复子任务，再Call原任务
- **统一表现**：所有重做操作均表现为再次Call，无单独算子

**入档原则**：每一次子层返回（成功或失败）均必须入done，直接添加返回的结果字符串，确保完整的执行审计链。

### 3.4 错误处理规范

**JSON格式错误处理**：
- **解析失败**：当LLM返回无效JSON时，实现MUST记录错误并终止当前分支
- **字段缺失**：当必需字段（type、description）缺失时，实现MUST返回验证错误
- **未知type值**：当type字段包含未定义值时，实现MUST返回类型错误并终止操作
- **格式恢复**：实现MAY尝试有限次数的格式修复，但MUST在连续失败后终止

**错误传播机制**：
- 算子级错误MUST向上传播到调用层
- 错误信息SHOULD包含足够的上下文以支持调试
- 系统级错误MUST触发优雅降级机制

---

## 4. 规范示例

### 4.1 标准执行流程

以下示例展示完整的递归执行流程，演示状态转换和不变式执行：

#### Step 1: 根节点初始化 S₀

```
goal  : 写一篇"AI与艺术"的短文（800–1200字）
parent: null
todo  : （空）
done  : []
```

#### Step 2: Think阶段 - 计划制定

$$\mathrm{Think}_{LLM}(Prompt_t,S_0,M,\mathsf{Tools}) → \textbf{Plan\_made}(T_0) → S_0.todo = T_0$$

```
分成三个段落撰写：
 [] 生成艺术段落  
 [] 辅助创作段落
 [] 艺术评论段落
```

#### Step 3: 首启阶段 - 首个子任务

$$Eval_{LLM}(Prompt_e,S_0,M) → Call(S_1^{(0)})$$

**当前状态**：S₀.done=[]（无前置返回结果，x=∅）

**创建子节点**：S₁⁽⁰⁾
```
goal  : 生成艺术段落：含DALL·E例，要求不少于100字，且能引起兴趣
parent: S₀
todo  : （空）
done  : []
```

#### Step 4: 递归调用

$$\mathrm{Solve}(S_1^{(0)}) → R_1^{(0)}$$

```
生成艺术段落已经写完，含DALL·E例，并符合要求。
```

#### Step 5: 续步阶段 - 处理返回结果

**状态更新**：S₀.done += R₁⁽⁰⁾（强不变式：先入档，x=R₁⁽⁰⁾）

$$Eval_{LLM}(Prompt_e,S_0,M) → Call(S_1^{(1)})$$

**当前状态**：S₀.done=[R₁⁽⁰⁾]（已包含前一个子任务结果）

**创建下一个子节点**：S₁⁽¹⁾
```
goal  : 辅助创作段落：写一个能打动人的事例
parent: S₀
todo  : （空）  
done  : []
```

继续此流程直至所有子任务完成，最终Eval返回完整文档。

---

## 5. 术语对照与定义

### 5.1 术语映射表

本规范中数学符号与JSON格式的对应关系：

| 数学符号 | JSON格式 | 算子 | 含义 |
|---------|----------|------|------|
| Return\_up(R) | "RETURN" | Think/Eval | 返回执行结果，终止当前分支 |
| Plan\_made(T) | "TODO" | Think | 制定自然语言计划 |
| Call(S) | "CALL" | Eval | 创建并调用子任务 |

**注**：本映射表确保跨文档和实现的术语一致性，避免理解歧义。

### 5.2 术语表

* **Call**：由父层Eval启动子层的操作；统一用于"下一项/重做/修复/计划优化"
* **Return\_up**：层结果上返操作；触发处立刻出栈，可携带状态标记
* **Plan\_made**：Think返回的计划状态，将自然语言计划写入当前节点的todo
* **sub-goal**：被Eval从todo中选定并Call的下一层目标，建立父子层的语义关系
* **redo**：失败/不足时的再次Call操作（同一条或修复后重做），无单独算子
* **强不变式**：子层返回后，父层必须先将结果入档到done，再进行下一步决策的架构约束
* **活性计划**：允许在todo中描述动态的、依赖执行结果的计划安排（如"先搜索→再修订计划"）
* **首启阶段**：Eval_LLM第一次被调用时的状态，此时x=∅，S_n.done保持当前状态
* **续步阶段**：处理子任务返回结果的状态，此时x=R_{n+1}，先执行S_n.done.append操作
* **有界递归**：在深度、资源、时间三重约束下执行的递归系统，确保终止性
* **优雅降级**：约束违反时返回部分结果而非完全失败的系统行为

---

## 6. 一句话总结

**Think只"给结果或给计划（自然语言）"；Eval统一在S_n.todo和S_n.done的上下文中做两选一：要么Call下一步（含重做/修复/计划优化），要么认定完成并Return\_up。**

每一次子层返回都必须先入档到done（强不变式），形成完整的、可回放、可审计的分形递归执行链。JSON格式的description字段由prompt决定，保证LLM输出的最大自由度。

---

## 7. 符合性要求

### 7.1 最小合规要求

符合本规范的实现MUST满足以下要求：

**核心算子实现**：
- MUST 实现Think_LLM和Eval_LLM算子的JSON协议
- MUST 支持第5.1节术语映射表中定义的所有JSON格式
- MUST 正确解析和生成标准JSON输出格式

**状态管理**：
- MUST 实现S数据结构的四个字段：goal、parent、todo、done
- MUST 执行强不变式：子层返回后先入档再决策
- MUST 维护递归调用栈的完整性

**终止性保证**：
- MUST 实现第2.4节定义的三类约束检查机制
- MUST 在约束违反时执行优雅降级
- SHOULD 提供可配置的约束参数

### 7.2 扩展点定义

实现者MAY扩展以下接口，但MUST保持核心语义一致性：

**Tools接口扩展**：
- MAY 定义特定领域的工具集
- MUST 确保工具调用仅在Think算子中执行
- SHOULD 提供工具调用的超时和错误处理

**Memory接口扩展**：
- MAY 实现跨层级的上下文存储和检索
- MUST 保证Memory访问不影响强不变式
- SHOULD 支持Memory的可选性（算子可在无Memory情况下运行）

### 7.3 版本与兼容性

**版本号格式**：MAJOR.MINOR.PATCH
- MAJOR：不兼容的API变更
- MINOR：向后兼容的功能增加
- PATCH：向后兼容的错误修复

**兼容性承诺**：
- 同一MAJOR版本内保证向后兼容
- 新版本MUST支持旧版本的核心JSON协议
- 扩展功能的变更仅影响MINOR版本

---

## 安全考虑

实现者应根据第2.4节的终止性保证要求，建立完整的约束检查和监控机制：

**约束实现建议**：
- 递归深度监控：追踪调用栈深度，防止栈溢出
- 资源消耗追踪：监控计算资源使用，支持成本控制
- 时间限制检查：设置执行超时，确保系统响应性

**安全防护措施**：
- 多层防护：软限制（告警）+ 硬限制（强制终止）
- 异常输入防护：防止恶意输入导致的资源耗尽攻击
- 监控与日志：记录约束违反事件，支持系统调优

## 作者地址

[aiden]

---

*本规范为技术标准文档，旨在确保不同实现间的完全兼容性。*
