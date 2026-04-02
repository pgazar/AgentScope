import modal

app = modal.App("agentscope-judge")


@app.function(secrets=[modal.Secret.from_name("anthropic-api-key")])
def run_geval_judge(
    criteria:    str,
    input_text:  str,
    output_text: str,
    context:     list[str],
) -> float:
    """
    Serverless G-Eval judge call via Modal.
    Invoked by agentscope/judge/modal_judge.py for cloud-based
    evaluation runs where local inference is too slow or costly.
    """
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    metric = GEval(
        name="remote_judge",
        criteria=criteria,
        model="claude-sonnet-4-5",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
    )
    tc = LLMTestCase(
        input=input_text,
        actual_output=output_text,
        retrieval_context=context,
    )
    metric.measure(tc)
    return metric.score
