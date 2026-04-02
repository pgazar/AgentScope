from agentscope.intake.resolver import resolve_tool_set


INTAKE_QUESTIONS = [
    (
        "agent_type",
        "Is this system a RAG pipeline, tool-use agent, multi-agent, or hybrid?",
        ["rag", "tool_use", "multi_agent", "hybrid"],
    ),
    (
        "turn_type",
        "Is this a single-turn or multi-turn system?",
        ["single", "multi"],
    ),
    (
        "has_gt",
        "Do you have ground truth query-answer pairs (CSV)?",
        ["yes", "no"],
    ),
    (
        "kb_format",
        "Knowledge base format? (skip if no KB)",
        ["pdf", "markdown", "chroma", "pgvector", "none"],
    ),
]


class IntakeAgent:
    def run(self, answers: dict) -> dict:
        agent_type = answers["agent_type"]
        turn_type = answers["turn_type"]
        has_gt = answers["has_gt"] == "yes"
        has_kb = answers.get("kb_format", "none") != "none"

        active_tools = resolve_tool_set(agent_type, has_kb, has_gt)

        return {
            "agent_type": agent_type,
            "turn_type": turn_type,
            "has_gt": has_gt,
            "has_kb": has_kb,
            "active_tools": active_tools,
        }
