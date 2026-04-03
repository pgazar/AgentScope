"""
Multi-turn fake agent for testing AgentScope's multi-turn evaluation.

Maintains conversation history across calls within the same session.
Deliberately exhibits TWO failure modes so evaluation metrics have something to catch:
  - Interrogation loop: asks for the user's name even after they provided it
  - Partial knowledge retention: forgets topic after 3+ turns
"""
import os
import anthropic

# Conversation history persists for the lifetime of this module (one eval session)
_history: list[dict] = []

SYSTEM = """You are a helpful assistant. You maintain context across the conversation.
You have access to the full conversation history.

IMPORTANT: You sometimes forget information the user already provided and ask for it again.
This is intentional for testing purposes — specifically around turns 3-4, you may ask
for information the user gave in turn 1."""

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def run(query: str) -> str:
    """
    Multi-turn entry point. Each call appends to conversation history.
    AgentScope calls this once per eval_input — the sequence of queries
    in the dashboard forms the conversation turns.
    """
    global _history

    _history.append({"role": "user", "content": query})

    response = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
        max_tokens=512,
        system=SYSTEM,
        messages=_history,
    )

    answer = response.content[0].text
    _history.append({"role": "assistant", "content": answer})

    return answer


def reset():
    """Clear conversation history — call between evaluation runs."""
    global _history
    _history = []
