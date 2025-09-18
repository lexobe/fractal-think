# 分形思考（Fractal Thinkon）正式规范 · 极简递归版

---

## 0. 宗旨

以最小结构刻画“计划—执行—评估—收束”的递归工作流。核心约束：

* **Think** 只返回两类之一：

  1. **Return\_up(Rₙ)**：本层执行结果，包括思考的结果、tool的调用及其结果、完成的代码等等；
  2. **Plan\_ready(Tₙ)**：纯自然语言的表述，以便有更好的扩展性，例如“根据搜索结果决定下一步动作”。
* **Eval** $R_{n+1}$ 则根据 `todo`/`done` , 只输出两类之一：

  1. **$Call(S_{n+1})$**：启动一个**子目标（sub-goal）**；（$R_{n+1} = Null$ 是第一个子目标）
  2. **Return\_up(Rₙ)**：认定本层完成并上返。
* **强不变式**：任何子层 `Return_up(Rₙ₊₁)` 后，**父层必须把 `Rₙ₊₁` 追加到 `done`**，再做裁决。

---

## 1. 数据结构（唯一）

$$
S=\{\ \textbf{goal}:g,\ \textbf{parent}:S_p\ \text{或}\ null,\ \textbf{todo}:T,\ \textbf{done}:D\ \}
$$

* **goal**：当前层目的。
* **parent**：父刺激（根为 null）。
* **todo**：**纯自然语言**计划文本（通常写成带编号的条目 t1/t2/…，但内容可“活性”——允许描述“先做搜索→再修订计划”等）。
* **done**：完成历史（列表）$R_{n+1}^{1..k}$，每个条目直接存储子层返回的结果字符串。

> 约束：`todo` 的**内容**总由 **Think** 产生（本层或经“计划优化”子任务返回的文本），**Eval 只采用,不修改**。

---

## 2. 两个算子（LLM 函数）

### 2.1 Think

$$
\mathrm{Think}_{LLM}(Prompt_t,S_n,M,\mathsf{Tools})
\ \Rightarrow\
\textbf{Return\_up}(R_n)\ \big|\ \textbf{Plan\_ready}(T_n)
$$

**输出格式（JSON Dict）**：
```json
{
    "type": "RETURN" | "TODO",
    "description": "由prompt决定的自然语言内容"
}
```

* **type="TODO"**（对应Plan\_ready）：把description中的**自然语言计划**写入 `S_n.todo`（可含判定标准/依赖/"先做A后改计划"）。
* **type="RETURN"**（对应Return\_up）：直接给出本层结果，description为执行结果。
* 说明：
  * 工具的调用仅在 Think 中实现，包括但不限于 MCP、CLI 等。
  * Think 在每个新节点 $S_n$ 创建时被激活，作为该层的第一个操作。

### 2.2 Eval

**状态更新机制**：
$$
S_n.done\ \  += x,
\quad x\in\{\varnothing,\ R_{n+1}\}
$$

注：当 $x=\varnothing$ 时，模拟一个空的返回事件，此时 $S_n.done$ 保持不变；当 $x=R_{n+1}$ 时，执行 $S_n.done.append(R_{n+1})$。

**Eval算子定义**：
$$
\mathrm{Eval}_{LLM}(Prompt_e,S_n,M)
\ \Rightarrow\
\textbf{Call}(S_{n+1})\ \big|\ \textbf{Return\_up}(R_n)
$$

**输出格式（JSON Dict）**：
```json
{
    "type": "RETURN" | "CALL",
    "description": "文字描述"
}
```

* **type="CALL"**（对应Call）：description包含子目标描述，用于创建 `S_{n+1}`。
* **type="RETURN"**（对应Return\_up）：description为本层完成结果。

**工作机制**：Eval_LLM通过访问 `S_n.todo` 和 `S_n.done` 的内部状态来感知当前阶段，x值通过更新S_n.done变相传递给Eval_LLM。

* **首启阶段 `x=∅`**：首个子目标由用户初始输入的goal分解而来，S_n.done保持当前状态，Eval依据 `S_n.todo/S_n.done` 决定 **Call(S_{n+1})** 或直接 **Return\_up**。
* **续步阶段 `x=R_{n+1}`**：后续子目标都是从todo中拆分出来的，**先**把 `R_{n+1}` 追加到 `S_n.done`，然后Eval依据更新后的状态决定 **Call(S_{n+1})** 或 **Return\_up**。
* **一句话**：依据 `todo/done` 决定**Call 下一项/重做/修复**（都属于 **Call**）或 **Return\_up**
  
> **Redo/反思失败/next**：都统一表现为再次 **Call** 。

---

## 3. 递归控制流（单链，无并发）

记 `Solve(Sₙ)` 为求解流程。

**(1) Think\@n：计划或直接返回**

* `Return_up(Rₙ)` → 本层出栈；
* `Plan_ready(Tₙ)` → `Sₙ.todo ← Tₙ`，转 (2)。

**(2) Eval\@n：首启阶段（x=∅）**

Eval_LLM(Prompt_e,S_n,M) 基于当前 S_n.todo/S_n.done 状态：
* **Call(S_{n+1})**：生成并调用一个子节点；或
* **Return\_up(Rₙ)**：认为已完成。

**(3) Solve@子层**

* 递归运行，直到子层 **Return\_up(Rₙ₊₁)** 上返。

**(4) Eval\@n：续步阶段（x=Rₙ₊₁）**

* 先执行状态更新：`S_n.done += Rₙ₊₁`；
* 然后 Eval_LLM(Prompt_e,S_n,M) 基于更新后的状态决定：
  * **Call(S_{n+1})**：生成并调用下一个子节点；或
  * **Return\_up(Rₙ)**：收束本层。

---

## 4. 操作语义（推导式）

**Think 返回结果**

$$
\frac{\mathrm{Think}(S_n)=\textbf{Return\_up}(R_n)}{\uparrow R_n}
$$

**Think 返回计划（自然语言）**

$$
\frac{\mathrm{Think}(S_n)=\textbf{Plan\_ready}(T_n)}{S_n.\text{todo}\gets T_n}
$$

**Eval 首启阶段（无前置返回结果）**

$$
\frac{x=\varnothing}
{\mathrm{Eval}_{LLM}(Prompt_e,S_n,M)=\textbf{Call}(S_{n+1})\ \ \text{or}\ \ \textbf{Return\_up}(R_n)}
$$

**子层返回→先入档再裁决（续步阶段）**

$$
\frac{\mathrm{Solve}(S_{n+1})\leadsto \textbf{Return\_up}(R_{n+1})}
{S_n.done{+}{=}R_{n+1};\ \ \ \mathrm{Eval}_{LLM}(Prompt_e,S_n,M):
\ \textbf{Call}(S_{n+1})\ \text{or}\ \textbf{Return\_up}(R_n)}
$$

---

## 5. 失败与重做（只有 Call/Return 两态）

* **失败检测**：子层可在结果文本中明确表示失败及其原因。
* **重做机制**：
  - **直接重做**：Eval 基于done中的失败结果，重新生成相同类型的子任务
  - **修复重做**：先 Call 一个"补前置条件"的修复子任务，再 Call 原任务

* **入档原则**：每一次子层返回（成功或失败）均**必须**入 `done`，直接添加返回的结果字符串。

## 6. 错误处理与终止条件

* **LLM 响应错误**：当 Think_LLM 或 Eval_LLM 返回无效格式时，抛出格式错误并终止当前分支。
* **资源限制**：
  - **递归深度限制**：当递归层数超过预设阈值时，强制返回当前状态。
  - **Token 消耗限制**：当 Token 使用量超过预算时，触发提前终止机制。
  - **时间限制**：设置单个 Solve 调用的最大执行时间。
* **死循环检测**：当连续多次 Call 同一子目标且无进展时，触发循环检测机制。
* **优雅降级**：出现不可恢复错误时，返回已完成的部分结果，而非完全失败。

---

## 7. 示例（含活性计划、失败与重做）

#### Step1  S₀（根）

```
goal  : 写一篇“AI 与艺术”的短文（800–1200字）
parent: null
todo  : （空）
done  : []
```

#### Step2 **$Think(S_n) → Plan\_ready(T_0)  → S_n.todo = T_0$**（纯自然语言）

```
分成三个段落撰写：
 [] 生成艺术段落
 [] 辅助创作段落
 [] 艺术评论段落
```

#### Step3 **首启阶段：$Eval_{LLM}(Prompt_e,S_0,M) → Call(S_1^{(0)})$**
**当前状态**：$S_0.done=[]$（无前置返回结果，x=∅）

**由 Eval 创建的子节点 $S_1^{(0)}$**
```
goal  : 生成艺术段落：含 DALL·E 例，要求不少于100，且能引起兴趣；
parent: S₀
todo  : （空）
done  : []
```
#### Step4 **递归调用 Solve($S_1^{(0)}$) 返回的 $R_1^{(0)}$**

``` 
生成艺术段落已经写完，含 DALL·E 例，并符合要求。
```
#### Step5 **续步阶段：处理第一个子任务返回结果**
**状态更新**：$S_0.done += R_1^{(0)}$（强不变式：先入档，x=$R_1^{(0)}$）

**$Eval_{LLM}(Prompt_e,S_0,M) → Call(S_1^{(1)})$**
**当前状态**：$S_0.done=[R_1^{(0)}]$（已包含前一个子任务结果）

**由 Eval 创建的子节点 $S_1^{(1)}$**
```
goal  : 辅助创作段落：写一个能打动人的事例；
parent: S₀
todo  : （空）
done  : []
```

后续步骤略...

---

## 8. 伪代码（顺序单链）

```pseudo
function Solve(S_n):
  think_result = Think_LLM(P_t, S_n, M, Tools)  # 返回JSON dict
  
  if think_result.type == "RETURN":
    return think_result.description              # 直接返回结果
  elif think_result.type == "TODO":
    S_n.todo = think_result.description          # 写入计划

  # 首启阶段：x=∅，S_n.done保持当前状态
  while true:
    # Eval通过访问S_n.todo/S_n.done感知当前阶段
    eval_result = Eval_LLM(P_e, S_n, M)         # 返回JSON dict
    
    if eval_result.type == "RETURN":
      return eval_result.description             # 本层完成
    elif eval_result.type == "CALL":
      # 创建子节点
      S_child = create_child_node()
      S_child.parent = S_n
      S_child.goal = eval_result.description     # 子目标来自description
      S_child.todo = ""
      S_child.done = []
      
      R_child = Solve(S_child)                  # 递归调用

      # 强不变式：先入档再裁决（续步阶段：x=R_child）
      S_n.done.append(R_child) # 状态更新，下次Eval调用将感知到新状态

function create_child_node():
  return new S()  # 创建空的S结构体
```

---

## 9. 术语表

* **Call**：由父层 Eval 启动子层；用于"下一项/重做/修复/计划优化"。
* **Return\_up**：层结果上返；触发处立刻出栈，可携带状态标记。
* **Plan\_ready**：Think 返回的计划状态，将自然语言计划写入当前节点的 todo。
* **sub-goal**：被 Eval 从 `todo` 中选定并 Call 的下一层目标（语义关系）。
* **redo**：失败/不足时的再次 Call（同一条或修复后重做），无单独算子。
* **强不变式**：子层返回后，父层必须先将结果入档到 done，再进行下一步决策。
* **活性计划**：允许在 todo 中描述动态的、依赖执行结果的计划安排。
* **JSON格式**：Think和Eval的标准输出格式，包含type和description两个字段，保证LLM输出的最大自由度。

---

### 一句话总结

**Think 只"给结果或给计划（自然语言）"；Eval 统一在 `S_n.todo` 和 `S_n.done` 的上下文中做两选一：要么 Call 下一步（含重做/修复/计划优化），要么认定完成并 Return\_up。**

每一次子层返回都必须先入档到 `done`（强不变式），形成完整的、可回放、可审计的分形递归执行链。JSON格式的description字段由prompt决定，保证LLM的最大自由度。框架内置错误处理、资源限制和死循环检测机制，确保在复杂任务中的稳定性和可控性。
