import os


def build_model(model_name: str):
    """
    Returns the right model object for DeepEval.
    Claude models must be wrapped in AnthropicModel — DeepEval routes
    bare model strings through OpenAI's client by default.
    """
    if "claude" in model_name.lower():
        from deepeval.models import AnthropicModel
        return AnthropicModel(
            model=model_name,
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
    # OpenAI model names work as plain strings in DeepEval
    return model_name
