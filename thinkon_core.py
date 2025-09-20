"""
分形思考（Fractal Thinkon）核心库实现
基于 thinkon.md 技术规范 v1.0.0

核心组件：
- State S: 字典/数据类封装 goal/parent/todo/done
- Think/Eval 算子接口与解析器
- Solve 控制流实现
- 终止约束机制
- 统一日志系统
- 规则引擎示例
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any, Protocol, Dict, List, Union, Tuple, TypedDict
from abc import ABC, abstractmethod
import json
import logging
import time
from datetime import datetime
from enum import Enum


# =============================================================================
# 1. 配置与日志
# =============================================================================

def setup_logger(name: str = 'thinkon_core', level: int = logging.INFO,
                 custom_logger: Optional[logging.Logger] = None,
                 propagate: bool = False,
                 auto_configure: bool = True) -> Optional[logging.Logger]:
    """配置统一日志系统

    Args:
        name: logger名称
        level: 日志级别
        custom_logger: 自定义logger实例，如果提供则直接返回
        propagate: 是否向父logger传播，False避免重复输出
        auto_configure: 是否自动配置handler，False时仅返回基础logger

    Returns:
        配置好的logger实例，如果custom_logger为None且auto_configure为False则返回None
    """
    if custom_logger:
        return custom_logger

    if not auto_configure:
        # 不自动配置，返回基础logger或None
        return logging.getLogger(name) if name else None

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - [L%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = propagate
    return logger


# =============================================================================
# 2. 状态结构 S
# =============================================================================

@dataclass
class S:
    """
    状态结构 S = {goal, parent, todo, done}

    按照 thinkon.md 规范实现：
    - goal: 当前层目的，定义本层的求解目标
    - parent: 父节点引用（根节点为None），形成递归调用栈
    - todo: 纯自然语言计划文本，支持活性计划描述
    - done: 完成历史列表，每个条目存储子层返回的结果字符串

    架构约束：todo的内容总由Think算子产生，Eval算子只采用不修改
    """
    goal: str
    parent: Optional['S'] = None
    todo: str = ""
    done: List[str] = field(default_factory=list)

    # 调试和监控字段
    level: int = field(init=False)
    timestamp: str = field(init=False)

    def __post_init__(self):
        """计算派生字段"""
        self.level = 0 if self.parent is None else self.parent.level + 1
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，支持JSON序列化"""
        return {
            "goal": self.goal,
            "todo": self.todo,
            "done": self.done.copy(),
            "level": self.level,
            "timestamp": self.timestamp,
            "has_parent": self.parent is not None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], parent: Optional['S'] = None) -> 'S':
        """从字典格式创建实例"""
        instance = cls(
            goal=data["goal"],
            todo=data.get("todo", ""),
            done=data.get("done", []).copy(),
            parent=parent
        )
        return instance


# =============================================================================
# 3. 算子接口与动作类型
# =============================================================================

# TypedDict 定义响应结构
class ThinkResponse(TypedDict, total=False):
    type: str  # "RETURN" | "TODO"
    description: str
    tokens_used: int  # 可选字段

class EvalResponse(TypedDict, total=False):
    type: str  # "RETURN" | "CALL"
    description: str
    tokens_used: int  # 可选字段

@dataclass
class ReturnAction:
    """Return_up(R_n): 返回执行结果，终止当前递归分支"""
    description: str

@dataclass
class PlanAction:
    """Plan_made(T_n): Think返回计划状态，将自然语言计划写入S_n.todo"""
    description: str

@dataclass
class CallAction:
    """Call(S_{n+1}): 创建子任务，description包含子目标描述"""
    description: str


# 算子协议定义
class ThinkLLM(Protocol):
    """
    Think算子接口：Think_LLM(Prompt_t, S_n, M, Tools)

    在每个新节点创建时激活，作为该层的第一个操作。
    工具调用权限仅限于Think算子。

    返回格式：{"type": "RETURN" | "TODO", "description": str}
    可选字段：{"tokens_used": int}
    """
    def __call__(self, node: S, memory: Any = None, tools: Any = None) -> Dict[str, Any]:
        ...

class EvalLLM(Protocol):
    """
    Eval算子接口：Eval_LLM(Prompt_e, S_n, M)

    通过访问S_n.todo和S_n.done的完整状态感知当前阶段。
    首启阶段(x=∅)：基于S_n.todo/done决定Call或Return_up
    续步阶段(x=R_{n+1})：先更新done，再基于新状态决定后续行动

    返回格式：{"type": "RETURN" | "CALL", "description": str}
    可选字段：{"tokens_used": int}
    """
    def __call__(self, node: S, memory: Any = None) -> Dict[str, Any]:
        ...


# =============================================================================
# 4. 响应解析器
# =============================================================================

def parse_think_response(response: Dict[str, Any], context: str = "") -> Union[ReturnAction, PlanAction]:
    """
    解析Think算子响应，严格校验字段

    标准输出格式：
    {
        "type": "RETURN" | "TODO",
        "description": "自然语言内容",
        "tokens_used": int  # 可选
    }
    """
    if not isinstance(response, dict):
        raise ValueError(f"Think response must be dict, got: {type(response)}. Context: {context}")

    # 必需字段检查
    if "type" not in response:
        raise ValueError(f"Think response must contain 'type' field. Context: {context}")

    if "description" not in response:
        raise ValueError(f"Think response must contain 'description' field. Context: {context}")

    response_type = response["type"]
    description = response["description"]

    # 类型检查
    if not isinstance(response_type, str):
        raise ValueError(f"Think response 'type' must be string, got: {type(response_type)}. Context: {context}")

    if not isinstance(description, str):
        raise ValueError(f"Think response 'description' must be string, got: {type(description)}. Context: {context}")

    # 可选字段白名单检查
    allowed_fields = {"type", "description", "tokens_used"}
    extra_fields = set(response.keys()) - allowed_fields
    if extra_fields:
        # 警告但不阻止，保持向后兼容
        import warnings
        warnings.warn(f"Think response contains unexpected fields: {extra_fields}. Context: {context}")

    if response_type == "RETURN":
        return ReturnAction(description)
    elif response_type == "TODO":
        return PlanAction(description)
    else:
        raise ValueError(f"Invalid Think response type: {response_type}. Must be 'RETURN' or 'TODO'. Context: {context}")


def parse_eval_response(response: Dict[str, Any], context: str = "") -> Union[ReturnAction, CallAction]:
    """
    解析Eval算子响应，严格校验字段

    标准输出格式：
    {
        "type": "RETURN" | "CALL",
        "description": "自然语言描述",
        "tokens_used": int  # 可选
    }
    """
    if not isinstance(response, dict):
        raise ValueError(f"Eval response must be dict, got: {type(response)}. Context: {context}")

    # 必需字段检查
    if "type" not in response:
        raise ValueError(f"Eval response must contain 'type' field. Context: {context}")

    if "description" not in response:
        raise ValueError(f"Eval response must contain 'description' field. Context: {context}")

    response_type = response["type"]
    description = response["description"]

    # 类型检查
    if not isinstance(response_type, str):
        raise ValueError(f"Eval response 'type' must be string, got: {type(response_type)}. Context: {context}")

    if not isinstance(description, str):
        raise ValueError(f"Eval response 'description' must be string, got: {type(description)}. Context: {context}")

    # 可选字段白名单检查
    allowed_fields = {"type", "description", "tokens_used"}
    extra_fields = set(response.keys()) - allowed_fields
    if extra_fields:
        # 警告但不阻止，保持向后兼容
        import warnings
        warnings.warn(f"Eval response contains unexpected fields: {extra_fields}. Context: {context}")

    if response_type == "RETURN":
        return ReturnAction(description)
    elif response_type == "CALL":
        return CallAction(description)
    else:
        raise ValueError(f"Invalid Eval response type: {response_type}. Must be 'RETURN' or 'CALL'. Context: {context}")


# =============================================================================
# 5. 终止约束机制
# =============================================================================

@dataclass
class Constraints:
    """终止约束配置"""
    max_depth: int = 10
    max_tokens: int = 10000
    max_time: float = 60.0  # seconds


class SolveStatus(Enum):
    """求解状态枚举 - 纯控制流状态"""
    COMPLETED = "completed"       # 控制流正常走到终点（不判断业务成功与否）
    DEGRADED = "degraded"         # 约束触发，优雅降级
    FAILED = "failed"             # 异常失败


@dataclass
class TokenUsage:
    """Token使用统计"""
    total: int = 0                # 总消耗
    think_calls: int = 0          # Think调用次数
    eval_calls: int = 0           # Eval调用次数
    think_tokens: int = 0         # Think消耗总量
    eval_tokens: int = 0          # Eval消耗总量

    def add_think(self, tokens: int):
        """记录Think调用"""
        self.think_calls += 1
        self.think_tokens += tokens
        self.total += tokens

    def add_eval(self, tokens: int):
        """记录Eval调用"""
        self.eval_calls += 1
        self.eval_tokens += tokens
        self.total += tokens

    def merge(self, other: 'TokenUsage'):
        """合并另一个TokenUsage的统计数据，保持数据一致性"""
        self.think_calls += other.think_calls
        self.eval_calls += other.eval_calls
        self.think_tokens += other.think_tokens
        self.eval_tokens += other.eval_tokens
        self.total += other.total

    def validate_consistency(self) -> bool:
        """验证统计数据的一致性"""
        return self.total == self.think_tokens + self.eval_tokens



@dataclass
class SolveResult:
    """统一的求解结果对象"""
    status: SolveStatus           # 执行状态
    result: str                   # 最终结果描述
    token_usage: TokenUsage       # Token消耗统计
    execution_time: float         # 执行时间（秒）
    max_depth_reached: int        # 达到的最大递归深度
    constraint_triggered: Optional[str] = None  # 触发的约束类型
    partial_results: List[str] = field(default_factory=list)  # 部分结果列表
    # 降级上下文信息
    failure_path: List[str] = field(default_factory=list)  # 失败路径（根到触发节点的目标序列）
    failure_level: Optional[int] = None  # 触发约束的具体层级
    failure_node_goal: Optional[str] = None  # 触发约束的节点目标
    failure_node_done: List[str] = field(default_factory=list)  # 触发约束时该节点已完成的步骤

    @property
    def is_completed(self) -> bool:
        """是否正常完成（控制流走到终点）"""
        return self.status == SolveStatus.COMPLETED

    @property
    def is_degraded(self) -> bool:
        """是否优雅降级"""
        return self.status == SolveStatus.DEGRADED

    @property
    def is_failed(self) -> bool:
        """是否失败"""
        return self.status == SolveStatus.FAILED


def _build_failure_path(node: S) -> List[str]:
    """构建从根节点到当前节点的失败路径"""
    path = []
    current = node
    while current is not None:
        path.append(current.goal)
        current = current.parent
    return list(reversed(path))  # 从根到叶的顺序


def _format_degraded_result(node: S, constraint_info: str, partial_results: List[str]) -> str:
    """格式化降级结果的详细描述"""
    result_parts = [
        f"任务触发约束降级：{constraint_info}",
        f"失败路径：{' -> '.join(_build_failure_path(node))}",
        f"失败层级：Level {node.level}",
        f"节点目标：{node.goal}"
    ]

    if node.todo:
        result_parts.append(f"计划内容：{node.todo[:100]}{'...' if len(node.todo) > 100 else ''}")

    if partial_results:
        result_parts.append(f"已完成步骤（{len(partial_results)}项）：")
        for i, item in enumerate(partial_results[:3], 1):  # 只显示前3项
            result_parts.append(f"  {i}. {item[:50]}{'...' if len(item) > 50 else ''}")
        if len(partial_results) > 3:
            result_parts.append(f"  ... 还有{len(partial_results) - 3}项")
    else:
        result_parts.append("已完成步骤：无")

    return "\n".join(result_parts)


class DepthLimitExceeded(Exception):
    """深度约束违反异常"""
    def __init__(self, message: str, node: S, current_depth: int, max_depth: int):
        super().__init__(message)
        self.node = node
        self.current_depth = current_depth
        self.max_depth = max_depth
        self.failure_path = _build_failure_path(node)

class ResourceLimitExceeded(Exception):
    """资源约束违反异常"""
    def __init__(self, message: str, node: S, current_tokens: int, max_tokens: int):
        super().__init__(message)
        self.node = node
        self.current_tokens = current_tokens
        self.max_tokens = max_tokens
        self.failure_path = _build_failure_path(node)

class TimeLimitExceeded(Exception):
    """时间约束违反异常"""
    def __init__(self, message: str, node: S, elapsed_time: float, max_time: float):
        super().__init__(message)
        self.node = node
        self.elapsed_time = elapsed_time
        self.max_time = max_time
        self.failure_path = _build_failure_path(node)


# =============================================================================
# 6. 核心求解流程 Solve
# =============================================================================

def _solve_internal(node: S, think_llm: ThinkLLM, eval_llm: EvalLLM,
                   memory: Any = None, tools: Any = None,
                   constraints: Optional[Constraints] = None,
                   token_usage: Optional[TokenUsage] = None, start_time: Optional[float] = None,
                   logger: Optional[logging.Logger] = None,
                   _max_depth_tracker: Optional[List[int]] = None) -> SolveResult:
    """
    核心递归求解函数 Solve(S_n)

    按照规范2.1伪代码实现：
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
    """
    # 初始化
    if constraints is None:
        constraints = Constraints()
    if token_usage is None:
        token_usage = TokenUsage()
    if start_time is None:
        start_time = time.time()
    if logger is None:
        logger = setup_logger(auto_configure=False) or logging.getLogger(__name__)
    if _max_depth_tracker is None:
        _max_depth_tracker = [0]

    # 更新最大深度跟踪
    _max_depth_tracker[0] = max(_max_depth_tracker[0], node.level)

    # 检查约束
    try:
        _check_constraints(node, constraints, token_usage.total, start_time, logger)
    except (DepthLimitExceeded, ResourceLimitExceeded, TimeLimitExceeded) as e:
        # 构建降级结果，包含完整上下文
        return SolveResult(
            status=SolveStatus.DEGRADED,
            result=_format_degraded_result(node, str(e), node.done),
            token_usage=token_usage,
            execution_time=time.time() - start_time,
            max_depth_reached=_max_depth_tracker[0],
            constraint_triggered=type(e).__name__,
            partial_results=node.done.copy(),
            failure_path=_build_failure_path(node),
            failure_level=node.level,
            failure_node_goal=node.goal,
            failure_node_done=node.done.copy()
        )

    if logger:
        logger.info(f"[Level {node.level}] Enter Solve: {node.goal}")

    # 1. Think阶段
    think_response = think_llm(node, memory, tools)
    context = f"Level {node.level}, Goal: {node.goal[:30]}..."
    think_action = parse_think_response(think_response, context)

    # 记录token消耗
    think_tokens = think_response.get("tokens_used", 1)  # 默认按步计数
    token_usage.add_think(think_tokens)

    if logger:
        logger.info(f"[Level {node.level}] Think result: {type(think_action).__name__} "
                    f"(+{think_tokens} tokens, total: {token_usage.total})")

    # Think后约束检查
    try:
        _check_constraints(node, constraints, token_usage.total, start_time, logger)
    except (DepthLimitExceeded, ResourceLimitExceeded, TimeLimitExceeded) as e:
        return SolveResult(
            status=SolveStatus.DEGRADED,
            result=_format_degraded_result(node, f"Think后触发约束：{str(e)}", node.done),
            token_usage=token_usage,
            execution_time=time.time() - start_time,
            max_depth_reached=_max_depth_tracker[0],
            constraint_triggered=type(e).__name__,
            partial_results=node.done.copy(),
            failure_path=_build_failure_path(node),
            failure_level=node.level,
            failure_node_goal=node.goal,
            failure_node_done=node.done.copy()
        )

    # 2. 如果Think直接返回结果
    if isinstance(think_action, ReturnAction):
        if logger:
            logger.info(f"[Level {node.level}] Direct return: {think_action.description[:50]}...")
        return SolveResult(
            status=SolveStatus.COMPLETED,
            result=think_action.description,
            token_usage=token_usage,
            execution_time=time.time() - start_time,
            max_depth_reached=_max_depth_tracker[0]
        )

    # 3. Plan_made：设置todo
    if isinstance(think_action, PlanAction):
        node.todo = think_action.description
        if logger:
            logger.debug(f"[Level {node.level}] Plan set: {node.todo[:50]}...")
    else:
        raise TypeError(f"Think must return ReturnAction or PlanAction, got: {type(think_action)}")

    # 4. 首启阶段：x=∅
    eval_response = eval_llm(node, memory)
    eval_action = parse_eval_response(eval_response)
    eval_tokens = eval_response.get("tokens_used", 1)
    token_usage.add_eval(eval_tokens)

    if logger:
        logger.info(f"[Level {node.level}] First Eval: {type(eval_action).__name__} "
                    f"(+{eval_tokens} tokens, total: {token_usage.total})")

    # 首次Eval后约束检查
    try:
        _check_constraints(node, constraints, token_usage.total, start_time, logger)
    except (DepthLimitExceeded, ResourceLimitExceeded, TimeLimitExceeded) as e:
        return SolveResult(
            status=SolveStatus.DEGRADED,
            result=_format_degraded_result(node, f"首次Eval后触发约束：{str(e)}", node.done),
            token_usage=token_usage,
            execution_time=time.time() - start_time,
            max_depth_reached=_max_depth_tracker[0],
            constraint_triggered=type(e).__name__,
            partial_results=node.done.copy(),
            failure_path=_build_failure_path(node),
            failure_level=node.level,
            failure_node_goal=node.goal,
            failure_node_done=node.done.copy()
        )

    # 5. While循环：处理CALL
    call_count = 0
    while isinstance(eval_action, CallAction):
        call_count += 1
        if logger:
            logger.info(f"[Level {node.level}] Call #{call_count}: {eval_action.description[:50]}...")

        # 创建子节点
        child = S(goal=eval_action.description, parent=node)

        # 递归调用 - 捕获异常以保存部分进展
        try:
            child_solve_result = _solve_internal(child, think_llm, eval_llm, memory, tools,
                                                constraints, None, start_time, logger, _max_depth_tracker)
            # 合并子层的完整token统计，保持数据一致性
            token_usage.merge(child_solve_result.token_usage)

            # 强不变式：先入档
            node.done.append(child_solve_result.result)
            if logger:
                logger.debug(f"[Level {node.level}] Child result archived. Done count: {len(node.done)}")

            # 如果子任务触发了约束，传播状态
            if child_solve_result.is_degraded:
                return SolveResult(
                    status=SolveStatus.DEGRADED,
                    result=f"子任务触发约束\n\n子任务的详细信息：\n{child_solve_result.result}",  # 保留子任务的详细信息
                    token_usage=token_usage,
                    execution_time=time.time() - start_time,
                    max_depth_reached=_max_depth_tracker[0],
                    constraint_triggered=child_solve_result.constraint_triggered,
                    partial_results=node.done.copy(),  # 只保留父节点的done，避免重复
                    failure_path=child_solve_result.failure_path,  # 继承子任务的失败路径
                    failure_level=child_solve_result.failure_level,
                    failure_node_goal=child_solve_result.failure_node_goal,
                    failure_node_done=child_solve_result.failure_node_done
                )
        except (DepthLimitExceeded, ResourceLimitExceeded, TimeLimitExceeded) as e:
            # 部分进展保全：生成摘要追加到done
            partial_summary = f"子任务'{child.goal[:30]}...'部分完成（{str(e)}）"
            node.done.append(partial_summary)
            if logger:
                logger.warning(f"[Level {node.level}] Child failed with constraint, partial progress saved: {partial_summary}")
            # 重新抛出异常以继续向上传播
            raise

        # 续步阶段：x=R_child
        eval_response = eval_llm(node, memory)
        eval_action = parse_eval_response(eval_response)
        eval_tokens = eval_response.get("tokens_used", 1)
        token_usage.add_eval(eval_tokens)

        if logger:
            logger.info(f"[Level {node.level}] Continue Eval: {type(eval_action).__name__} "
                        f"(+{eval_tokens} tokens, total: {token_usage.total})")

        # 每次Eval后检查约束
        try:
            _check_constraints(node, constraints, token_usage.total, start_time, logger)
        except (DepthLimitExceeded, ResourceLimitExceeded, TimeLimitExceeded) as e:
            return SolveResult(
                status=SolveStatus.DEGRADED,
                result=_format_degraded_result(node, f"续步Eval后触发约束：{str(e)}", node.done),
                token_usage=token_usage,
                execution_time=time.time() - start_time,
                max_depth_reached=_max_depth_tracker[0],
                constraint_triggered=type(e).__name__,
                partial_results=node.done.copy(),
                failure_path=_build_failure_path(node),
                failure_level=node.level,
                failure_node_goal=node.goal,
                failure_node_done=node.done.copy()
            )

    # 6. Return收束
    if isinstance(eval_action, ReturnAction):
        if logger:
            logger.info(f"[Level {node.level}] Final return after {call_count} calls: "
                        f"{eval_action.description[:50]}...")
        return SolveResult(
            status=SolveStatus.COMPLETED,
            result=eval_action.description,
            token_usage=token_usage,
            execution_time=time.time() - start_time,
            max_depth_reached=_max_depth_tracker[0]
        )
    else:
        raise TypeError(f"Eval must return ReturnAction or CallAction, got: {type(eval_action)}")


def _check_constraints(node: S, constraints: Constraints, current_tokens: int,
                      start_time: float, logger: Optional[logging.Logger]):
    """检查终止约束"""
    # 深度约束
    if node.level >= constraints.max_depth:
        if logger:
            logger.warning(f"Depth limit exceeded: {node.level} >= {constraints.max_depth}")
        raise DepthLimitExceeded(
            f"Depth limit exceeded: {node.level} >= {constraints.max_depth}",
            node, node.level, constraints.max_depth
        )

    # 资源约束
    if current_tokens >= constraints.max_tokens:
        if logger:
            logger.warning(f"Token limit exceeded: {current_tokens} >= {constraints.max_tokens}")
        raise ResourceLimitExceeded(
            f"Token limit exceeded: {current_tokens} >= {constraints.max_tokens}",
            node, current_tokens, constraints.max_tokens
        )

    # 时间约束
    elapsed = time.time() - start_time
    if elapsed >= constraints.max_time:
        if logger:
            logger.warning(f"Time limit exceeded: {elapsed:.2f}s >= {constraints.max_time}s")
        raise TimeLimitExceeded(
            f"Time limit exceeded: {elapsed:.2f}s >= {constraints.max_time}s",
            node, elapsed, constraints.max_time
        )


def solve(node: S, think_llm: ThinkLLM, eval_llm: EvalLLM,
          memory: Any = None, tools: Any = None,
          constraints: Optional[Constraints] = None,
          logger: Optional[logging.Logger] = None) -> str:
    """
    核心递归求解函数 - 向后兼容版本

    按照规范2.1节，返回自然语言字符串结果

    Args:
        node: 当前节点状态
        think_llm: Think算子
        eval_llm: Eval算子
        memory: 记忆系统
        tools: 工具系统
        constraints: 约束配置
        logger: 日志记录器

    Returns:
        str: 自然语言结果字符串

    Raises:
        DepthLimitExceeded: 深度约束违反
        ResourceLimitExceeded: 资源约束违反
        TimeLimitExceeded: 时间约束违反
    """
    result = _solve_internal(node, think_llm, eval_llm, memory, tools, constraints, logger=logger)
    return result.result


def solve_with_meta(node: S, think_llm: ThinkLLM, eval_llm: EvalLLM,
                   memory: Any = None, tools: Any = None,
                   constraints: Optional[Constraints] = None,
                   logger: Optional[logging.Logger] = None) -> SolveResult:
    """
    核心递归求解函数 - 带元信息版本

    返回完整的SolveResult对象，包含执行状态、token统计等元信息

    Args:
        node: 当前节点状态
        think_llm: Think算子
        eval_llm: Eval算子
        memory: 记忆系统
        tools: 工具系统
        constraints: 约束配置
        logger: 日志记录器

    Returns:
        SolveResult: 完整的求解结果对象
    """
    result = _solve_internal(node, think_llm, eval_llm, memory, tools, constraints, logger=logger)

    # 验证TokenUsage数据一致性
    if not result.token_usage.validate_consistency():
        if logger:
            logger.warning(f"TokenUsage inconsistency detected: "
                         f"total={result.token_usage.total}, "
                         f"think_tokens={result.token_usage.think_tokens}, "
                         f"eval_tokens={result.token_usage.eval_tokens}")

    return result


def start_solve(goal: str, think_llm: ThinkLLM, eval_llm: EvalLLM,
                memory: Any = None, tools: Any = None,
                constraints: Optional[Constraints] = None,
                logger: Optional[logging.Logger] = None) -> SolveResult:
    """启动分形思考求解过程的便捷函数

    Returns:
        SolveResult: 统一的结果对象，包含状态、结果、token消耗等信息
    """
    root = S(goal=goal)
    try:
        result = solve_with_meta(root, think_llm, eval_llm, memory, tools, constraints, logger=logger)
        if logger and result.is_completed:
            logger.info(f"Task completed. Total tokens used: {result.token_usage.total}")
        return result
    except Exception as e:
        # 意外异常，返回失败结果，包含完整栈信息
        import traceback
        stack_trace = traceback.format_exc()

        if logger:
            logger.error(f"Unexpected error: {e}")
            logger.error(f"Stack trace:\n{stack_trace}")

        return SolveResult(
            status=SolveStatus.FAILED,
            result=f"任务执行异常：{str(e)}\n\n栈跟踪信息：\n{stack_trace}",
            token_usage=TokenUsage(),
            execution_time=0.0,
            max_depth_reached=0,
            partial_results=getattr(root, 'done', [])
        )


def start_solve_simple(goal: str, think_llm: ThinkLLM, eval_llm: EvalLLM,
                      memory: Any = None, tools: Any = None,
                      constraints: Optional[Constraints] = None,
                      logger: Optional[logging.Logger] = None) -> str:
    """简化版启动函数，仅返回结果字符串（向后兼容）"""
    result = start_solve(goal, think_llm, eval_llm, memory, tools, constraints, logger)
    return result.result


# =============================================================================
# 7. 导出接口
# =============================================================================

__all__ = [
    # 核心类型
    'S', 'Constraints', 'SolveResult', 'SolveStatus', 'TokenUsage',
    'ReturnAction', 'PlanAction', 'CallAction',
    'ThinkLLM', 'EvalLLM',
    'ThinkResponse', 'EvalResponse',  # TypedDict类型

    # 解析器
    'parse_think_response', 'parse_eval_response',

    # 求解函数
    'solve', 'solve_with_meta', 'start_solve', 'start_solve_simple',

    # 异常类型
    'DepthLimitExceeded', 'ResourceLimitExceeded', 'TimeLimitExceeded',

    # 工具函数
    'setup_logger'
]

# Constraints默认值说明
# Constraints(max_depth=10, max_tokens=10000, max_time=60.0)
# SolveResult提供统一的结果接口，支持success/degraded/failed状态区分
