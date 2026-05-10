from typing import Any, Dict, Optional, Union

from litellm.integrations.custom_logger import CustomLogger

from agentlightning.llm_proxy import _get_pre_call_data  # type: ignore


class AddLogprobs(CustomLogger):
    """LiteLLM logger hook to request logprobs from vLLM.

    This mutates the outgoing request payload to include `logprobs=1`
    for backends that support logprobs return (e.g., vLLM).
    """

    async def async_pre_call_hook(self, *args: Any, **kwargs: Any) -> Optional[Union[Exception, str, Dict[str, Any]]]:
        """Async pre-call hook to adjust request payload.

        Args:
            args: Positional args from LiteLLM.
            kwargs: Keyword args from LiteLLM.

        Returns:
            Either an updated payload dict or an Exception to short-circuit.
        """
        try:
            data = _get_pre_call_data(args, kwargs)
        except Exception as e:
            return e

        # Ensure logprobs are requested from the backend when supported.
        return {**data, "logprobs": 1}


class AddSamplingParams(CustomLogger):
    """LiteLLM logger hook to enforce unbiased sampling for rollout generation.

    Explicitly sets temperature=1.0 and disables all other decoding filters
    (top_p, top_k, min_p, repetition_penalty, presence_penalty, frequency_penalty)
    so the raw model distribution is preserved. This is critical for the validity
    of importance sampling ratios used during training.
    """

    async def async_pre_call_hook(self, *args: Any, **kwargs: Any) -> Optional[Union[Exception, str, Dict[str, Any]]]:
        """Async pre-call hook to adjust request payload.

        Args:
            args: Positional args from LiteLLM.
            kwargs: Keyword args from LiteLLM.

        Returns:
            Either an updated payload dict or an Exception to short-circuit.
        """
        try:
            data = _get_pre_call_data(args, kwargs)
        except Exception as e:
            return e

        # reference: https://arxiv.org/pdf/2508.03501
        # this means make temperature=1 while disabling all other sampling params.
        return {
            **data,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        }
