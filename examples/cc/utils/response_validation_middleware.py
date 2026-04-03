"""Middleware to validate LLM responses before they reach Claude Code.

Intercepts responses from the vLLM backend and checks for malformed tool calls,
missing tool calls, or invalid tool names. When a check fails, returns a synthetic
response with an observation message instead of forwarding the bad response.
"""

import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Module-level config set before the middleware is added to the app.
_allowed_tools: Optional[Set[str]] = None
_max_step: Optional[int] = None


def set_allowed_tools(tools: Set[str]) -> None:
    """Configure the allowed tool set for response validation."""
    global _allowed_tools
    _allowed_tools = tools


def set_max_step(max_step: int) -> None:
    """Configure the max step for the step warning middleware."""
    global _max_step
    _max_step = max_step


class ResponseValidationMiddleware(BaseHTTPMiddleware):
    """Validate LLM responses and reject malformed or disallowed tool calls.

    This middleware sits inside StreamConversionMiddleware so it operates on
    raw non-streaming JSON responses from vLLM. When a response fails
    validation, a synthetic response containing an error observation is
    returned so Claude Code feeds it back to the LLM on the next turn.
    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.method != "POST":
            return await call_next(request)

        # Only intercept chat completion endpoints.
        path = request.url.path
        if not (path.endswith("/chat/completions") or "/chat/completions?" in path):
            return await call_next(request)

        # Only validate responses for sonnet/opus models (the main agent), skip haiku.
        prev_assistant_had_no_tool_call = False
        try:
            req_body = json.loads(await request.body())
            model: str = req_body.get("model", "")
            if "haiku" in model.lower():
                return await call_next(request)

            # Check whether the last assistant message in history had no tool call.
            # If so, a second consecutive no-tool-call response is an intentional submission.
            messages: List[Dict[str, Any]] = req_body.get("messages", [])
            prev_assistant_had_no_tool_call = self._last_resp_without_tool_call(messages)
        except Exception:
            pass

        response = await call_next(request)

        if not (200 <= response.status_code < 300):
            return response

        # Buffer the response body.
        try:
            if hasattr(response, "body_iterator"):
                chunks: List[bytes] = []
                async for chunk in response.body_iterator:  # type: ignore[union-attr]
                    chunks.append(chunk)  # type: ignore[arg-type]
                body = b"".join(chunks)
            else:
                body = response.body  # type: ignore[union-attr]

            data: Dict[str, Any] = json.loads(body or b"{}")
        except Exception:
            logger.warning("ResponseValidationMiddleware: failed to parse response body, passing through.")
            return Response(content=body, status_code=response.status_code, headers=dict(response.headers))

        choices = data.get("choices")
        if not choices:
            return Response(content=body, status_code=response.status_code, headers=dict(response.headers))

        message: Dict[str, Any] = choices[0].get("message", {}) or {}
        tool_calls: List[Any] = message.get("tool_calls") or []
        content: str = message.get("content") or ""

        observation = self._validate(tool_calls, content, prev_assistant_had_no_tool_call)
        if observation is None:
            # All checks passed — return the original response.
            return Response(content=body, status_code=response.status_code, headers=dict(response.headers))

        logger.info("ResponseValidationMiddleware: intercepted bad response: %s", observation)

        # Build a synthetic response with the observation as content.
        synthetic = dict(data)
        synthetic["choices"] = [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": observation,
                },
                "finish_reason": "stop",
            }
        ]
        synthetic_body = json.dumps(synthetic).encode("utf-8")
        return Response(
            content=synthetic_body,
            status_code=200,
            headers={**dict(response.headers), "content-length": str(len(synthetic_body))},
            media_type="application/json",
        )

    @staticmethod
    def _validate(tool_calls: List[Any], content: str, prev_assistant_had_no_tool_call: bool) -> Optional[str]:
        """Return an observation string if the response is invalid, or ``None`` if it is fine."""
        if not tool_calls and "<tool_call>" in content:
            return (
                "Your tool call has JSON decode error and cannot be parsed. "
                "Please retry to generate a tool call with correct JSON format."
            )

        if not tool_calls and "<tool_call>" not in content:
            if prev_assistant_had_no_tool_call:
                # if the last and the current response both satisfy: not tool_calls and "<tool_call>" not in content
                return None
            return (
                "Your last response does not have a <tool_call>. In the Claude Code setting "
                "if you generate a response without <tool_call>, it means you want to submit "
                "your answer. If you really want to submit your answer, please generate the same "
                "response as your last response again without <tool_call> in your next response. "
                "If you have not finished the task, your next response must contain a <tool_call>."
            )

        allowed = _allowed_tools
        if allowed is not None and tool_calls:
            name = tool_calls[0].get("function", {}).get("name", "")
            if name not in allowed:
                return (
                    f"The tool call used in your response is not in the allowed tool list. "
                    f"Allowed tools are {allowed}. "
                    f"Generate a response with a tool call in the allowed tools."
                )

        return None
    
    @staticmethod
    def _last_resp_without_tool_call(messages: List[Dict[str, Any]]):
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                # find last response
                if not msg.get("tool_calls") and "<tool_call>" not in msg.get("content", ""):
                    return True
                else:
                    return False
        return False


class StepWarningMiddleware(BaseHTTPMiddleware):
    """Append a step-limit warning to user/tool messages in the request.

    Counts non-haiku assistant messages in the conversation history.
    When that count reaches ``max_step - 5``, the warning is appended to
    the last 5 non-assistant messages. Because Claude Code may strip
    appended content between turns, this re-checks and re-appends on
    every request.
    """

    WARNING_TAG = "You are reaching the max step limit of"

    @staticmethod
    def _build_warning(max_step: int, steps_left: int) -> str:
        return (
            f"\n\n<Warning>\n"
            f"You are reaching the max step limit of {max_step}, you have {steps_left} steps left.\n"
            f"Please prepare your submission, especially edit the code if you haven't.\n"
            f"If you want to submit your current changes and terminate the dialogue, just generate a response without tool call. "
            f"If you want to take more actions, you must generate a response with one tool call. "
            f"Note that a response without tool call will directly terminate the dialogue and submit your changes. "
            f"If you want to continue the dialogue, each of your response should have one tool call."
            f"\n</Warning>\n"
        )
    
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.method != "POST":
            return await call_next(request)

        path = request.url.path
        if not (path.endswith("/chat/completions") or "/chat/completions?" in path):
            return await call_next(request)

        max_step = _max_step
        if max_step is None:
            return await call_next(request)

        # Skip haiku requests.
        try:
            req_body = json.loads(await request.body())
            model: str = req_body.get("model", "")
            if "haiku" in model.lower():
                return await call_next(request)
        except Exception:
            return await call_next(request)

        # Count assistant messages (each one represents a step by the main agent).
        messages: List[Dict[str, Any]] = req_body.get("messages", [])
        assistant_count = sum(1 for m in messages if m.get("role") == "assistant")

        if assistant_count < max_step - 5:
            return await call_next(request)

        # Append the warning to the last 5 non-assistant messages that don't already have it.
        appended = 6-(max_step-assistant_count)
        for msg in reversed(messages):
            if appended <= 0:
                break
            if msg.get("role") == "assistant":
                continue
            content = msg.get("content")
            if isinstance(content, str) and self.WARNING_TAG not in content:
                msg["content"] = content + self._build_warning(max_step, 5-appended)
            appended-=1

        logger.info(
            "StepWarningMiddleware: ensured warning in last 5 user/tool messages (assistant_count=%d, max_step=%d)",
            assistant_count,
            max_step,
        )

        # Replace the request body with the modified messages.
        req_body["messages"] = messages
        modified_body = json.dumps(req_body).encode("utf-8")
        request._body = modified_body  # type: ignore[attr-defined]

        return await call_next(request)
