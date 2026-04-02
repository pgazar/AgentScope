import os
from agentscope.orchestrator.state import AgentState


def run(state: AgentState) -> AgentState:
    """
    Generates synthetic Q&A pairs from the knowledge base using DeepEval's
    Synthesizer. All generation goes through G-Eval's pipeline — consistent
    with the decision that G-Eval handles all LLM judgment calls.
    """
    from deepeval.synthesizer import Synthesizer
    from deepeval.synthesizer.config import UseCase, SynthesizerConfig

    model_name = state["config"]["judge"]["model"]
    kb_path = state["kb_path"]
    max_pairs = state["config"]["eval"].get("max_synth_pairs", 50)

    synthesizer = Synthesizer(
        model=model_name,
        config=SynthesizerConfig(use_case=UseCase.QA),
    )

    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=[kb_path],
        max_goldens_per_document=max_pairs,
        include_expected_output=True,
    )

    # Hard cap regardless of what the synthesizer returns
    goldens = goldens[:max_pairs]

    run_id = state["run_id"]
    os.makedirs("outputs", exist_ok=True)
    synthesizer.save_as(
        file_type="csv",
        directory="outputs",
        file_name=f"{run_id}_synthetic_gt",
    )

    out_path = f"outputs/{run_id}_synthetic_gt.csv"
    state["gt_path"] = out_path
    state["synth_results"] = {
        "n_pairs": len(goldens),
        "coverage_score": round(len(goldens) / max_pairs, 2),
    }
    return state
