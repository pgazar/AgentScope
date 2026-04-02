from pydantic import BaseModel, Field
from typing import Literal


class JudgeConfig(BaseModel):
    backend: Literal["claude", "openai"] = "claude"
    model: str = "claude-sonnet-4-5"
    temperature: float = 0.0
    max_tokens: int = 512


class EvalConfig(BaseModel):
    k: int = 5
    max_steps: int = 10
    ir_green: float = 0.70
    ir_orange: float = 0.40
    agent_green: float = 0.80
    agent_orange: float = 0.60
    geval_green: float = 0.80
    geval_orange: float = 0.60
    hallu_green: float = 0.10
    hallu_orange: float = 0.25
    cost_green: float = 0.020
    cost_orange: float = 0.050
    latency_green: float = 2.0
    latency_orange: float = 5.0
    # Pipeline cost caps — prevents silent cost creep
    max_synth_pairs: int = 50
    max_geval_responses: int = 100
    eval_budget_usd: float = 2.00


class AgentScopeConfig(BaseModel):
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)


def load_config(path: str = "config.yaml") -> AgentScopeConfig:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)
    return AgentScopeConfig(**raw)
