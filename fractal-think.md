# 分形思考流程 

## 一、理念基础

* **每一个思考 = 一个 Node（分形节点）**。
* Node 是思绪的最小自治单元：它有意图（Intent）、交付物（Deliverable）、度量（Metric）、约束（Constraint）。
* Node 一经创建，就立刻进入执行流程：直做→成功/失败；或拆分递归。
* Node 自治但不孤立：继承上层意图与约束，输出结果与决策，由父 Node 复核并决定是否固化为新约束。

---

## 二、Node 的 Core4 结构

每个 Node 必须以 **Core4** 表述：

### 1. Intent（意图）

* **定义**：思考/行动背后的驱动力，“为什么要做”。
* **扩展说明**：可以是目标、兴趣、好奇心，也可以是为了检验某个假设。
* **本质**：指向性与意义。

---

### 2. Deliverable（交付物）

* **定义**：该思考/任务要产出的**最终表达**，可以是任何可被认可的形态。
* **扩展说明**：

  * 可以是具体产物（代码、数据、论文）。
  * 也可以是抽象表达（一个公式、一句话、一个符号、一个隐喻）。
  * 它是“这次思考的落点”。
  * **本质**：结果的可感知形式。

---

### 3. Metric（度量）

* **定义**：判定 Deliverable 是否达成预期的标准。
* **扩展说明**：

  * 包括量化标准（准确率、性能、时间）。
  * 也包括质性标准（简单、优雅、启发性）。
  * 可以是严格可验证的，也可以是群体共识/审美判断。
* **本质**：结果的判定规则。

---

### 4. Constraint（约束 / 环境准备）

* **定义**：执行时必须满足的边界与所需的环境。
* **扩展说明**：

  * **资源边界**：时间、算力、注意力。
  * **风险边界**：错误率、可靠性阈值。
  * **表达边界**：不得超出一行、一页、一个符号；或必须用指定框架。
  * **环境准备**：上下文、依赖条件（思绪连续性、必要背景）。
  * **预算限制**：尝试上限/时间预算/调用上限；达到上限应早停并返回。
* **本质**：护栏 + 基底，保证思考不会走散。
---

## 三、Node 的四种思考方式

### 1. 思考1：直面处理

* 输入：本 Node 的 Core4。
* 执行：尝试直接完成 Deliverable 并对照 Metric 自评。
* 结果：

  * success → 返回 success
  * failure → 返回 failure
  * Todo → 进入思考4（展开子 Node）

### 2. 思考2：收到 success 的验收和下一步 Core4 内容的更新

* 输入：子 Node 的 success response。
* 执行：

  * Review：

    * 对成果进行评价：评价的结论满足子 Node 和本 Node 的 Metric 与 Constraint，
    * 对实现方式评价：判断其是否合理。
    * 表述兼容性检查：子 Node 的表述/接口是否满足本 Node 期望的格式/模式/版本要求。
  * 抽象（What）：在不改变语义的前提下，将子 Node 的结果提炼为可复用的表述/接口（格式、输入/输出、关键约束、适用范围），并给出一个最小示例；在 Decisions 中记录是否建议固化为新约束。
  * 吸收：从 Decisions 中选择关键选择，必要时固化为新约束。
  
* 选择：

  * 如果接受：

    * Next Todo：根据已有成果，更新下一个 Todo 的 Core4 内容；
    * End：根据已完成的成果，生成 success response 向上
  * 如果不接受（下列二选一）：

    * 重做 → 创建新子 Node（替代路径）。
    * 返回 failure 向上。

### 3. 思考3：收到 failure 的反思

* 输入：子 Node 的 failure response。
* 执行：

  * 进行反思，判断失败是方法性（可换路径）还是结构性（约束/目标冲突）。
  * 定义：
    * 方法性失败（method）：在不改变当前 Intent/Metric/Constraint 的前提下，仍存在可行替代路径（含“信息获取/验证”作为路径）。
    * 结构性失败（structural）：在不改变当前 Intent/Metric/Constraint 的前提下，不存在可行路径；除非放宽/调整约束、目标或度量。
  * 分工：
    * 子 Node 标注：子 Node 在返回 failure 时必须给出 `failure_type`，并在 `root_cause` 与 `next_suggestions` 中说明依据与后续建议。
    * 父 Node 复核：父 Node 在本步骤复核 `failure_type`；如与子 Node 不同，可覆盖分类，并在本 Node 的 Decisions 中记录重分类及理由。
* 选择：

  * response → 无替代可能 → 返回 failure
  * 重试 → 根据失败信息，更新 Core4 表述，创建新子 Node（替代路径）

### 4. 思考4：展开 Todo，分步骤思考

* 输入：一个 Core4 列表（待办子 Node 定义）。
* 执行：顺序执行这些子 Node。
* 过程中：

  * 子 success → 按思考2逻辑处理
  * 子 failure → 按思考3逻辑处理
  * 遵守 Constraint 中的预算限制（尝试/时间/调用）；达到上限即早停并返回 failure。
  * 允许复用既有 Deliverable；但需确保其表述/接口与本 Node 的期望兼容。
* 收敛：

  * 若满足本 Node 的 Metric → 返回 success
  * 否则 → 返回 failure

> **说明**：Todo 是 Node 的内部递归方式，对上不直接暴露。对外只有 success/failure。

---

## 四、Node 的输出（Response）

### ✅ Success Response

```yaml
type: success
node_id: string
result: string | object
self_metric:
  - metric: string
    status: "达成" | "未达成"
    evidence: string
decisions:
  - choice: string
    suggest_promote: bool
residual_risk:
  - risk: string
    mitigation: string
```

> 建议：在 `result` 顶部用简洁文字给出“表述/接口（What）”的抽象（包含格式/模式与一个最小示例），并在 Decisions 中注明是否建议将该表述固化为约束，便于上层理解与重复使用。

### ❌ Failure Response

```yaml
type: failure
node_id: string
failure_type: method | structural
failure_summary:
  - metric: string
    gap: string
attempt_log:
  - method: string
    outcome: string
root_cause: string
what_we_learned:
  - insight: string
next_suggestions:
  - suggestion: string
```

> 说明：`failure_type` 用于标注失败归因层级——method（方法性，可替代路径存在）或 structural（结构性，仅通过调整约束/目标/度量才可能达成）。

---

## 五、运行原则

1. **创建即执行**：Node 从 Core4 出发，直接进入思考1。
2. **出口唯一**：对外只有 success / failure。
3. **最小代价**：在 Constraint 内尽最大可能，优先早停。
4. **透明记录**：关键选择必须自然语言记录在 Decisions 中，可标注是否建议固化。
5. **固化权**：只有父 Node 能将子 Node 的 Decisions 转为新约束。
6. **残余风险**：复杂 Node 必须写 Residual Risk；小 Node 可留空。

---

## 六、一句话总结

**分形思考流程：每个 Node 创建即思考 → 内部可能 Todo 递归 → 对外只返回 success / failure → 父 Node 吸收决策与风险，并固化为新约束 → 全局思绪递归展开而不迷失。**
