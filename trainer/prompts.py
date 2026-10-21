# Modified for DTCA: retain Drift's LLaDA math prompt.

DEFAULT_SYSTEM_PROMPT_LLADA_MATH = (
    """<|startoftext|><|start_header_id|>user<|end_header_id|>"""
    """You need to put your final answer in \\boxed{}. This is the problem:\n"""
    """{{problem}}<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"""
)


def get_default_system_prompt(model_type: str, data_type: str = "math") -> str:
    """Return the math prompt used by both DTCA and RLDF."""
    if model_type != "llada" or data_type != "math":
        raise ValueError("The released experiments use LLaDA on math tasks")
    return DEFAULT_SYSTEM_PROMPT_LLADA_MATH
