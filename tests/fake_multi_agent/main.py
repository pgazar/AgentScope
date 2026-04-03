"""
Multi-turn fake agent — reference test agent for AgentScope multi-turn evaluation.

Maintains conversation history across calls within the same eval session.
Use with turn_type=multi in the dashboard.

Example eval inputs (one per line):
    Hi! My name is Pegah and I am an AI engineer working on agentic systems.
    What are the best frameworks for building AI agents?
    Can you remind me what I told you about my job at the start?
    Based on my role, which of those frameworks should I prioritize?

Tests: knowledge_retention, turn_relevancy, conversation_completeness
"""
import os
import anthropic

_history: list[dict] = []

SYSTEM = """You are a helpful assistant with perfect memory.
You remember everything the user has told you across the entire conversation.
Always refer back to what the user has shared about themselves when relevant."""

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def run(query: str) -> str:
    """
    Multi-turn entry point. Each call appends to conversation history.
    The sequence of eval_inputs in the dashboard forms the conversation turns.
    """
    global _history
    _history.append({"role": "user", "content": query})

    response = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
        max_tokens=256,
        system=SYSTEM,
        messages=_history,
    )

    answer = response.content[0].text
    _history.append({"role": "assistant", "content": answer})
    return answer
