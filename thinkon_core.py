"""
分形思考（Fractal Thinkon）递归框架实现
基于 thinkon.md 正式规范
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any, Protocol, Union, Callable, Dict, List
from abc import ABC, abstractmethod
import json
import logging
from datetime import datetime



# 配置 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [L%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('FractalThinkon')


class S(dict):
    """
    唯一数据结构 S = {goal, parent, todo, done}
    按照 thinkon.md 规范：
    - goal: 当前层目的
    - parent: 父刺激（根为 null）
    - todo: 纯自然语言计划文本
    - done: 完成历史（列表）
    现基于 dict 实现，提供更灵活的数据结构
    """
    
    def __init__(self, goal: str, parent: Optional['S'] = None, 
                 todo: str = "", done: List[str] = None, 
                 level: int = 0, timestamp: str = None):
        super().__init__()
        self['goal'] = goal
        self['parent'] = parent
        self['todo'] = todo
        self['done'] = done or []
        # 保留 level 和 timestamp 用于调试和监控
        self['level'] = level if parent is None else parent['level'] + 1
        self['timestamp'] = timestamp or datetime.now().isoformat()
    
    def __getattr__(self, name: str) -> Any:
        """支持属性访问语法"""
        if name in self:
            return self[name]
        raise AttributeError(f"'S' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """支持属性设置语法"""
        self[name] = value
    
    @property
    def goal(self) -> str:
        return self['goal']
    
    @goal.setter
    def goal(self, value: str):
        self['goal'] = value
    
    @property
    def todo(self) -> str:
        return self['todo']
    
    @todo.setter
    def todo(self, value: str):
        self['todo'] = value
    
    @property
    def done(self) -> List[str]:
        return self['done']
    
    @done.setter
    def done(self, value: List[str]):
        self['done'] = value
    
    @property
    def parent(self) -> Optional['S']:
        return self['parent']
    
    @parent.setter
    def parent(self, value: Optional['S']):
        self['parent'] = value
    
    @property
    def level(self) -> int:
        return self['level']
    
    @level.setter
    def level(self, value: int):
        self['level'] = value
    
    @property
    def timestamp(self) -> str:
        return self['timestamp']
    

    def with_parent(self, parent: 'S') -> 'S':
        """设置父层引用并计算层级"""
        self['parent'] = parent
        self['level'] = parent['level'] + 1 if parent else 0
        return self

    def to_json(self) -> Dict[str, Any]:
        """转换为 JSON 格式"""
        return {
            "goal": self['goal'],
            "todo": self['todo'],
            "done": self['done'],
            "level": self['level'],
            "timestamp": self['timestamp'],
            "has_parent": self['parent'] is not None
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any], parent: Optional['S'] = None) -> 'S':
        """从 JSON 格式创建实例"""
        instance = cls(
            goal=data["goal"],
            todo=data.get("todo", ""),
            done=data.get("done", []),
            level=data.get("level", 0),
            timestamp=data.get("timestamp")
        )
        if parent:
            instance.with_parent(parent)
        return instance


@dataclass
class PlanReady:
    """
    Plan_ready(T_n): Think 返回计划状态
    符合 thinkon.md 规范：把自然语言计划写入 S_n.todo
    """
    plan_text: str      # 纯自然语言计划文本
    reason: str = ""    # 计划理由

    def __str__(self):
        return f"Plan_ready(plan='{self.plan_text[:50]}...')"

    def to_json(self) -> Dict[str, Any]:
        """转换为 JSON 格式"""
        return {
            "action_type": "plan_ready",
            "plan_text": self.plan_text,
            "reason": self.reason,
            "timestamp": datetime.now().isoformat()
        }


@dataclass
class ReturnUp:
    """
    Return_up(R_n): 本层执行结果上返
    符合 thinkon.md 规范：直接给出本层结果，触发处立刻出栈
    """
    result: str     # 本层结果 R_n（自然语言）
    node: S         # 产生结果的节点（用于追踪）
    confidence: float = 1.0  # 结果置信度

    def __str__(self):
        return f"Return_up(result='{self.result}', level={self.node.level})"

    def to_json(self) -> Dict[str, Any]:
        """转换为 JSON 格式"""
        return {
            "action_type": "return_up",
            "result": self.result,
            "node": self.node.to_json(),
            "confidence": self.confidence,
            "timestamp": datetime.now().isoformat()
        }


@dataclass
class Call:
    """
    Call(S_{n+1}): Eval 返回的子目标调用
    符合 thinkon.md 规范：启动一个 sub-goal
    """
    child: S        # 子目标节点 S_{n+1}
    reason: str = ""    # 调用理由

    def __str__(self):
        return f"Call(child='{self.child.goal}', level={self.child.level})"

    def to_json(self) -> Dict[str, Any]:
        """转换为 JSON 格式"""
        return {
            "action_type": "call",
            "child": self.child.to_json(),
            "reason": self.reason,
            "timestamp": datetime.now().isoformat()
        }


# 动作类型联合
ThinkAction = Union[PlanReady, ReturnUp]
EvalAction = Union[Call, ReturnUp]
Action = Union[PlanReady, Call, ReturnUp]


class Think_LLM(Protocol):
    """
    Think 执行算子：在当前层决定下潜或返回
    Think_LLM(Prompt_t, S_n, M, Tools) => Dict{action_type, tokens_used, ...}
    必须返回包含 token 消耗信息的 Dict
    """
    def __call__(self, node: S, memory: Any = None, tools: Any = None) -> Dict[str, Any]:
        """
        Think 决定：下潜到子层 或 返回本层结果
        必须返回 Dict 格式:
        {
            "action_type": "down|return_up",
            "tokens_used": int,  # 必须包含
            "data": {...},
            "reasoning": "..."
        }
        """
        ...


class Eval_LLM(Protocol):
    """
    Eval 评估算子：仅在父层对子层上返结果裁决
    Eval_LLM(Prompt_e, S_n, R_{n+1}, M) => Dict{action_type, tokens_used, evaluation, ...}
    必须返回包含 token 消耗信息的 Dict
    """
    def __call__(self, node: S, child_result: ReturnUp, memory: Any = None) -> Dict[str, Any]:
        """
        Eval 裁决：基于子层返回的结果决定下一步
        必须返回 Dict 格式:
        {
            "action_type": "down|return_up",
            "tokens_used": int,  # 必须包含
            "evaluation": {...},
            "reasoning": "..."
        }
        """
        ...


def parse_think_response(response: Dict[str, Any], node: S) -> tuple[ThinkAction, int]:
    """解析 LLM 返回的 Dict 并提取 token 消耗"""
    if not isinstance(response, dict):
        raise ValueError(f"LLM must return dict format, got: {type(response)}")
    
    # 提取 token 消耗（必须字段）
    tokens_used = response.get("tokens_used")
    if tokens_used is None:
        raise ValueError("LLM response must contain 'tokens_used' field")
    
    action_type = response.get("action_type")
    if action_type == "down":
        data = response.get("data", {})
        child_goal = data.get("goal", "Generated subtask")
        child_context = data.get("context", "")
        reason = response.get("reasoning", "")
        
        child = S(goal=child_goal, context=child_context)
        return Down(child=child, reason=reason), tokens_used
    
    elif action_type == "return_up":
        result = response.get("result", "Task completed")
        confidence = response.get("confidence", 1.0)
        return ReturnUp(result=result, node=node, confidence=confidence), tokens_used
    
    raise ValueError(f"Unsupported action_type: {action_type}")


class TokenLimitExceeded(Exception):
    """当 token 消耗超过限制时抛出的异常"""
    pass


def Solve(node: S, think_llm: Think_LLM, eval_llm: Eval_LLM,
          memory: Any = None, tools: Any = None, 
          max_tokens: int = 10000, current_tokens: int = 0) -> tuple[ReturnUp, int]:
    """
    核心递归求解函数 Solve(S_n)

    按照规范的控制流：
    (1) n 层 - Think 决定：下潜或返回
    (2) 进入 n+1 层 - 递归调用 Solve(S_{n+1}^{(0)})
    (3) 子层返回后 - n 层的 Eval 裁决

    增强功能：
    - Token 消耗跟踪和限制
    - While 循环次数统计
    """

    # 记录入口
    logger.info(f"[Level {node.level}] Enter node: {node.goal} (Token: {current_tokens}/{max_tokens})")
    
    # 检查 token 限制
    if current_tokens >= max_tokens:
        raise TokenLimitExceeded(f"Token consumption exceeds limit: {current_tokens}/{max_tokens}")
    
    # (1) n 层 - Think 决定：下潜或返回
    think_response = think_llm(node, memory, tools)
    action, think_tokens = parse_llm_response(think_response, node)
    current_tokens += think_tokens
    
    logger.info(f"[Level {node.level}] Think decision: {action} (Token: +{think_tokens}, Total: {current_tokens})")
    if think_response.get('reasoning'):
        logger.debug(f"[Level {node.level}] Think reasoning: {think_response['reasoning']}")
    
    # 再次检查 token 限制
    if current_tokens >= max_tokens:
        raise TokenLimitExceeded(f"Token consumption exceeds limit after Think: {current_tokens}/{max_tokens}")

    if isinstance(action, ReturnUp):
        # 本层直接返回，控制权回父层
        logger.info(f"[Level {node.level}] Direct return: {action.result}")
        return action, current_tokens

    if not isinstance(action, Down):
        raise TypeError(f"Think_LLM must return Down or ReturnUp, got: {type(action)}")

    # (2) 进入 n+1 层 - 递归调用 Solve
    child = action.child.with_parent(node)
    logger.info(f"[Level {node.level}] Descend to child (Level {child.level}): {child.goal}")
    child_result, current_tokens = Solve(child, think_llm, eval_llm, memory, tools, max_tokens, current_tokens)
    logger.info(f"[Level {node.level}] Child returned: {child_result.result}")

    # (3) 子层返回后 - n 层的 Eval 裁决
    eval_loop_count = 0
    while True:  # 处理可能的多个子任务
        eval_loop_count += 1
        logger.info(f"[Level {node.level}] Eval loop #{eval_loop_count}")
        
        # 检查 token 限制
        if current_tokens >= max_tokens:
            raise TokenLimitExceeded(f"Token consumption exceeds limit before Eval: {current_tokens}/{max_tokens}")
        
        eval_response = eval_llm(node, child_result, memory)
        decision, eval_tokens = parse_llm_response(eval_response, node)
        current_tokens += eval_tokens
        
        logger.info(f"[Level {node.level}] Eval decision (loop#{eval_loop_count}): {decision} (Token: +{eval_tokens}, Total: {current_tokens})")
        if eval_response.get('reasoning'):
            logger.debug(f"[Level {node.level}] Eval reasoning: {eval_response['reasoning']}")
        
        # 检查 token 限制
        if current_tokens >= max_tokens:
            raise TokenLimitExceeded(f"Token consumption exceeds limit after Eval: {current_tokens}/{max_tokens}")

        if isinstance(decision, ReturnUp):
            # 本层完成，上返
            logger.info(f"[Level {node.level}] Complete return (after {eval_loop_count} loops): {decision.result}")
            return decision, current_tokens

        if not isinstance(decision, Down):
            raise TypeError(f"Eval_LLM must return Down or ReturnUp, got: {type(decision)}")

        # 继续在同一 n 层下降至下一子目标（尾递归）
        child = decision.child.with_parent(node)
        logger.info(f"[Level {node.level}] Continue descend to new child (Level {child.level}): {child.goal}")
        child_result, current_tokens = Solve(child, think_llm, eval_llm, memory, tools, max_tokens, current_tokens)


def start_solve(goal: str, think_llm: Think_LLM, eval_llm: Eval_LLM,
          memory: Any = None, tools: Any = None, 
          max_tokens: int = 10000, current_tokens: int = 0) -> tuple[ReturnUp, int]:
    """启动分形思考求解过程的便捷函数"""
    node = S(goal=goal)
    node.parent = None
    node.level = 0
    node.timestamp = datetime.now().isoformat()
    return Solve(node, think_llm, eval_llm, memory, tools, max_tokens, current_tokens)