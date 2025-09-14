# 分形思元（Fractal Thinkon）文档

## 1. 引言（Introduction）

分形思元（Fractal Thinkon）是一个递归式、层级化的认知框架，用于描述和实现 **目标驱动的分裂式思考与逐层评价机制**。其核心思想是：

* **下级负责生成**（思考与产出候选结果）；
* **上级负责评价**（对下级的结果进行检验、整合与抉择）；
* **刺激递归传递**（目标与已完成的步骤共同构成背景，驱动下一步）。

分形思元强调“父即子之背”（Parent-as-Child-Back）：任何子级的背景信息，必然包含其父级目标及父级已完成的步骤。

---

## 2. 基本定义（Core Definitions）

### 2.1 刺激（Stimulus）
一个刺激帧 $S_n^{(i)}$ 定义为：

$$
S_n^{(i)} = \{ \text{"target"}: g_n^{(i)},\ \text{"background"}: B_n^{(i)} \}
$$

背景 $B_n^{(i)}$ 定义为：

$$
B_n^{(i)} = \text{null}\ \ \text{or}\ \ \{ \text{"up"}: S_{n-1},\ \text{"done"}:d_n^{(i-1)} \}
$$

其中（可扩展）：
* $B_n^{(i)}$ 为null，表示为顶节点
* $S_{n-1}$ 是上一级的刺激
* $n$ 是当前递归的层级
* $g_n^{(i)}$：执行本级的第 $i$ 个目标
* $d_n^{(i-1)}$：本级已完成步骤的情况说明


### 2.2 思考（Thinking）

下级思元在刺激 $S_n$ 和记忆 $M$ 下执行，并有两类返回形式：

$$
Think = \text{LLM}(Prompt_t, S_n, M) \to (o_n, M')
$$

其中 $o_n \in \{\,\text{Return}(r_n),\ \text{Split}(G_n)\,\}$。

含义：

- Return($r_n$)：给出本层的思考结果 $r_n$。
- Split($G_n$)：给出一组派生任务集 $G_n$，用于展开子层。
- $P_{\text{think}}$：思考提示词；
- $M'$：需要记住的内容；

### 2.3 评价（Evaluation）

评价仅在思考返回 Return 时进行，用于决定本层控制流：

$$
Eval = \text{LLM}(Prompt_e, S_{n}^{(i)} ,\ r_n^{(i)},d_n^{(i-1)} ,G_n,\ M\,) \to (a_n, d_n^{(i)}, g_n^{(i)'}, M')
$$

其中：

- $a_n \in \{\text{Think}, \text{Return}(r_n)\}$
  - Think：生成了一个新的刺激，例如：重做、或者下一步
  - Next: 接受前面思考的结果，进行下一个思考
  - Return：将本级的思考完成，将结果返回给上级
- $d_n^{(i)}$：完成$i$步后，对结果的情况说明
- $M'$：需要记住的内容；

### 2.4 Split 的派生

当思考返回 Split 时，父层获得一组需并行/序贯推进的子目标集合（计划），记为派生任务集：

$$
G_n = \{\, g_n^{(1)},\, g_n^{(2)},\, \dots,\, g_n^{(k)} \,\},\quad k\ge 1
$$

父层据此为每个子目标构造对应的子刺激帧：

$$
S_{n+1}^{(j)} = \Big\{\
  \text{"target"}:\, g^{(j)},\
  \ \text{"background"}:\, \{\text{"up"}: S_n,\ \text{"done"}: d_n^{(j)}\}\
\Big\},\quad j\in\{1,\dots,k\}
$$

其中：

- "target": 本子任务的具体目标 $g^{(j)}$。
- "path": 目标路径，定义为 $\operatorname{path}(S_0)=\{g_0\}$，且 $\operatorname{path}(S_{t+1})=\operatorname{path}(S_t)\cup\{g_{t+1}\}$；用于追踪从根到当前子任务的目标累积。若需要保持次序，可将并集视作“追加”。
- "plan": 本轮 Split 的整体计划 $T_n$（对所有 $j$ 恒等），便于子任务“知晓兄弟节点的全貌”。
- "background": 沿袭父即子之背原则，携带父层刺激 $S_n$ 与已完成摘要 $D_n$。

不变式（对所有 $j$）：

- $S_{n+1}^{(j)}.\text{background.up} = S_n$
- $S_{n+1}^{(j)}.\text{plan} = T_n$
- $\operatorname{path}(S_{n+1}^{(j)}) \supset \operatorname{path}(S_n)$ 且仅追加 $g^{(j)}$

示例（JSON 结构）：

```json
// 思考在 S_n 上返回 Split，给出 3 个子目标
Tasks_n /* 对应 T_n */ = ["收集需求", "列出模块", "制定里程碑"]

S_{n+1}^{(2)} = {
  "target": "列出模块",
  "path": ["写项目计划", "列出模块"],
  "plan": ["收集需求", "列出模块", "制定里程碑"],
  "background": { "up": S_n, "done": D_n }
}
```

执行策略由父层决定（并行/串行/带依赖顺序）。父层在各子任务完成后汇总其评价摘要进入新的 $D_{n+1}$，并据此继续派生或收敛。

### 2.5 刺激递归更新

若上级决定继续：

$$
S_{n+1} = \{ \text{"target"}: g_{n+1},\ \text{"background"}:\{\text{"up"}:S_n,\ \text{"done"}:D_n\}\}
$$

---

## 3. 工作流程（Workflow）

1. **接受刺激**：思元接收目标与背景。
2. **思考执行**：
   - 若返回 `Split(T_n)` → 为每个 $g^{(j)}\in T_n$ 构造子刺激 $S_{n+1}^{(j)}$ 并展开子层。
   - 若返回 `Return(r)` → 进入上级评价。
3. **上级评价（仅针对 Return）**：
   - 产生动作 $a\in\{\text{Redo},\ \text{Next},\ \text{Return\_up}\}$，并写入 `done` 摘要。
   - `Redo`：要求子层重做（可调整提示/约束后再次创建子刺激）。
   - `Next`：进入 Todo 中的下一个子任务。
   - `Return_up`：结束本层，将结果与 `done` 交回上层。
4. **收敛与继续**：依据 `done` 的累计与控制流动作，继续派生或上行收敛。
5. **终止条件**：当父层判断目标已充分完成，停止派生并上行。

---

## 4. 核心原则（Principles）

1. **生成-评价分离**：下级只负责生成，不自我评价。
2. **父即子之背**：父层的目标与已完成步骤构成子层的背景。
3. **递归可追溯性**：任意 $S_n$ 可通过 `background.up` 追溯至根刺激 $S_0$。
4. **结果-上下文统一**：刺激既包含目的（target），又包含历史痕迹（done）。

---

## 5. 形式化循环（Formal Loop）

单步过程拆分为思考输出与（可选的）评价决策：

$$
o_n, M_n' = Think(S_n, M_n),\quad o_n\in\{\text{Return}(r_n),\ \text{Split}(T_n)\}
$$

- 若 $o_n=\text{Split}(T_n)$：派生 $\{S_{n+1}^{(j)}\}$ 并递归执行。
- 若 $o_n=\text{Return}(r_n)$：

$$
a_n, \eta_n, M_p' = Eval(\text{Return}(r_n), M_p),\quad a_n\in\{\text{Redo},\ \text{Next},\ \text{Return\_up}\}
$$

据 $a_n$ 控制流进行 Redo/Next/Return\_up 处理并继续递归。

---

## 6. 示例（Example）

### 初始刺激

```json
S0 = {
  "target": "写一篇关于 AI 与艺术的短文",
  "background": null
}
```

### 完成“生成艺术/DALL·E”后

```json
S1 = {
  "target": "辅助创作",
  "background": {
    "up": S0,
    "done": ["生成艺术部分已完成，包含 DALL·E 例子"]
  }
}
```

### 完成“辅助创作”后，派生“艺术评论”

```json
S2 = {
  "target": "艺术评论",
  "background": {
    "up": S0,
    "done": [
      "生成艺术部分已完成，包含 DALL·E 例子",
      "辅助创作部分已完成，并强调人机协作"
    ]
  }
}
```

---

## 7. 总结（Conclusion）

分形思元是一个 **递归目标-背景模型**，其关键特征是：

* **思考与评价分工明确**
* **刺激递归承载目标与完成痕迹**
* **父层目标天然成为子层背景**
* **结果通过 done 机制累积**

这保证了思考过程既能保持分形展开的开放性，又能通过上级评价实现收敛与控制。

---

要不要我帮你把这份文档整理成 **学术论文的 LaTeX 模板**（含公式、定义、示例），可以直接作为正式发表的初稿？
