"""Middleware to validate and filter LLM requests and responses for Claude Code.

``ToolSelectionMiddleware`` filters the ``tools`` field in outgoing requests to
retain only allowed tools and injects a synthetic *Submit* tool so the model can
explicitly signal task completion.

``ResponseValidationMiddleware`` intercepts responses from the vLLM backend and
checks for malformed tool calls or missing tool calls. When a check fails it
appends the bad response and a retry prompt to the messages and re-calls the LLM.
If the model calls *Submit*, the tool-call is stripped so Claude Code sees a
plain text response (which it interprets as submission). Only valid responses are
forwarded to Claude Code.
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
    """Configure the allowed tool set for tool selection and response validation."""
    global _allowed_tools
    _allowed_tools = tools


def set_max_step(max_step: int) -> None:
    """Configure the max step for the step warning middleware."""
    global _max_step
    _max_step = max_step


class ToolSelectionMiddleware(BaseHTTPMiddleware):
    """Filter the ``tools`` field in LLM API requests.

    Retains only tools whose function name is in ``_allowed_tools`` and appends
    the synthetic *Submit* tool so the model can explicitly end the dialogue.
    """

    SUBMIT_TOOL_DEFINITION: Dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "Submit",
            "description": "The Submit tool submits your current changes and terminates the dialogue.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.method != "POST":
            return await call_next(request)

        req_body = json.loads(await request.body())
        model: str = req_body.get("model", "")
        if "haiku" in model.lower():
            return await call_next(request)

        tools: Optional[List[Dict[str, Any]]] = req_body.get("tools")
        allowed = _allowed_tools

        if tools is not None and allowed is not None:
            filtered = [t for t in tools if t.get("function", {}).get("name", "") in allowed]
            filtered.append(self.SUBMIT_TOOL_DEFINITION)
            req_body["tools"] = filtered
            logger.debug(
                "ToolSelectionMiddleware: filtered tools from %d to %d (including Submit)",
                len(tools),
                len(filtered),
            )
        else:
            logger.error(f"tool call middleware has error: tools = {tools}; allowed = {allowed}.")

        modified_body = json.dumps(req_body).encode("utf-8")
        request._body = modified_body  # type: ignore[attr-defined]
        return await call_next(request)


class ResponseValidationMiddleware(BaseHTTPMiddleware):
    """Validate LLM responses and retry the LLM call when validation fails.

    This middleware sits inside StreamConversionMiddleware so it operates on
    raw non-streaming JSON responses from vLLM.

    * If the model calls *Submit*, the ``tool_calls`` field is stripped and
      the response is forwarded as plain text — Claude Code treats this as
      submission.
    * If the response has no tool call at all, the model is always asked to
      retry (there is no implicit "consecutive no-tool-call = submission").
    * Malformed tool calls trigger a retry as well.
    """

    _MAX_VALIDATION_RETRIES: int = 3

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.method != "POST":
            return await call_next(request)

        # Only validate responses for sonnet/opus models (the main agent), skip haiku.
        req_body = json.loads(await request.body())
        model: str = req_body.get("model", "")
        if "haiku" in model.lower():
            return await call_next(request)

        messages: List[Dict[str, Any]] = req_body.get("messages", [])

        # First LLM call.
        response = await call_next(request)
        body, data, action = await self._parse_and_validate(response)
        if action is None: # means correct response 
            return Response(content=body, status_code=response.status_code, headers=dict(response.headers))
        if action == "submit":
            body = self._strip_tool_calls(data)
            return Response(content=body, status_code=response.status_code, headers=dict(response.headers))

        # Retry loop: append bad response + retry prompt, re-call LLM.
        rollout_id = request.headers.get("x-rollout-id")
        attempt_id = request.headers.get("x-attempt-id")

        for attempt in range(1, self._MAX_VALIDATION_RETRIES + 1):
            logger.info(
                "ResponseValidationMiddleware: retry %d/%d — %s",
                attempt, self._MAX_VALIDATION_RETRIES, action,
            )

            # Allocate a new sequence_id for this retry so spans don't collide.
            if rollout_id and attempt_id:
                from agentlightning.llm_proxy import get_active_llm_proxy

                store = get_active_llm_proxy().get_store()
                if store is not None:
                    new_seq_id = await store.get_next_span_sequence_id(rollout_id, attempt_id)
                    request.scope["headers"] = [
                        (k, v) for k, v in request.scope["headers"] if k != b"x-sequence-id"
                    ] + [(b"x-sequence-id", str(new_seq_id).encode())]

            # Append the failed assistant message and the retry prompt to messages.
            bad_message: Dict[str, Any] = (
                data["choices"][0].get("message", {}) if data and data.get("choices") else {}
            )
            messages.append(bad_message)
            messages.append({"role": "user", "content": action})

            req_body["messages"] = messages
            modified_body = json.dumps(req_body).encode("utf-8")
            request._body = modified_body  # type: ignore[attr-defined]

            response = await call_next(request)
            body, data, action = await self._parse_and_validate(response)
            if action is None:
                return Response(content=body, status_code=response.status_code, headers=dict(response.headers))
            if action == "submit":
                body = self._strip_tool_calls(data)
                return Response(content=body, status_code=response.status_code, headers=dict(response.headers))

        # Max retries exhausted — return the last response as-is.
        logger.warning("ResponseValidationMiddleware: max retries exhausted, passing through last response.")
        return Response(content=body, status_code=response.status_code, headers=dict(response.headers))

    @staticmethod
    async def _read_body(response: Response) -> bytes:
        """Read the full response body regardless of response type."""
        if hasattr(response, "body_iterator"):
            chunks: List[bytes] = []
            async for chunk in response.body_iterator:  # type: ignore[union-attr]
                if isinstance(chunk, str):
                    chunks.append(chunk.encode())
                else:
                    chunks.append(chunk)
            return b"".join(chunks)
        return response.body  # type: ignore[union-attr]

    async def _parse_and_validate(
        self, response: Response
    ) -> tuple[bytes, Optional[Dict[str, Any]], Optional[str]]:
        """Parse response body and validate.

        Returns ``(body_bytes, parsed_data, action)`` where *action* is:
        * ``None`` — response is valid, forward as-is.
        * ``"submit"`` — model called Submit, strip tool_calls and forward.
        * Any other string — an observation/retry prompt to send back to the model.
        """
        if not (200 <= response.status_code < 300):
            body = await self._read_body(response)
            return body, None, None

        try:
            body = await self._read_body(response)
            data: Dict[str, Any] = json.loads(body or b"{}")
        except Exception:
            logger.warning("ResponseValidationMiddleware: failed to parse response body, passing through.")
            return b"", None, None

        choices = data.get("choices")
        if not choices:
            return body, data, None

        message: Dict[str, Any] = choices[0].get("message", {}) or {}
        tool_calls: List[Any] = message.get("tool_calls") or []
        content: str = message.get("content") or ""

        action = self._validate(tool_calls, content)
        return body, data, action

    @staticmethod
    def _validate(tool_calls: List[Any], content: str) -> Optional[str]:
        """Return an action string describing what to do with this response.

        * ``None`` — valid response with a normal tool call, forward as-is.
        * ``"submit"`` — model called *Submit*, should strip tool_calls.
        * Any other string — a retry observation to append and re-call the LLM.
        """
        # No tool call at all — always retry.
        if len(tool_calls) > 1:
            return (
                "Your response has multiple tool calls. "
                "Please generate a response with exactly one tool call. "
            )

        # Check for Submit tool call.
        if tool_calls:
            name = tool_calls[0].get("function", {}).get("name", "")
            if name == "Submit":
                logger.info("ResponseValidationMiddleware: model called Submit — forwarding as plain text.")
                return "submit"

        # Malformed tool call (model tried but JSON was invalid).
        if not tool_calls and "<tool_call>" in content:
            return (
                "Your tool call has JSON decode error and cannot be parsed. "
                "Please retry to generate a tool call with correct JSON format."
            )

        # No tool call at all — always retry.
        if not tool_calls and ("<tool_call>" not in content):
            return (
                "Your last response does not contain a tool call. "
                "You must either call a tool to continue working, or call the Submit tool to submit your current changes. "
                "Generate a response with exactly one tool call. "
            )

        # Has a tool call — it's valid (tool filtering is handled by ToolSelectionMiddleware).
        return None

    @staticmethod
    def _strip_tool_calls(data: Optional[Dict[str, Any]]) -> bytes:
        """Remove ``tool_calls`` from the first choice's message and re-serialize."""
        if data and data.get("choices"):
            message: Dict[str, Any] = data["choices"][0].get("message", {})
            message.pop("tool_calls", None)
            # Ensure finish_reason reflects a normal stop, not a tool call.
            data["choices"][0]["finish_reason"] = "stop"
        return json.dumps(data).encode("utf-8")


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
            f"If you want to submit your current changes and terminate the dialogue, call the Submit tool. "
            f"If you want to take more actions, you must generate a response with one tool call. "
            f"Note that calling the Submit tool will directly terminate the dialogue and submit your changes. "
            f"If you want to continue the dialogue, each of your responses should have one tool call."
            f"\n</Warning>\n"
        )
    
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.method != "POST":
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
                appended-=1
                continue
            content = msg.get("content")
            if isinstance(content, str) and self.WARNING_TAG not in content:
                msg["content"] = content + self._build_warning(max_step, 5-appended)

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
