from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


@dataclass
class ProviderResult:
    raw_text: str
    model_id: str | None
    request_id: str | None
    usage: dict[str, Any]
    raw_response: dict[str, Any]
    predicted_type_id: int | None = None
    raw_output: str | None = None
    http_status: int | None = None


class GenerationProvider(Protocol):
    provider_name: str
    method: str

    def generate(self, input_text: str) -> ProviderResult: ...


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        category: str,
        status_code: int | None = None,
        raw_output: str | None = None,
        raw_response: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.category = category
        self.status_code = status_code
        self.raw_output = raw_output
        self.raw_response = raw_response or {}


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        env_path = Path(".env")
        if not env_path.exists():
            return
        for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            os.environ.setdefault(name, value)
        return
    load_dotenv(override=False)


def _model_dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        result = dump()
        return result if isinstance(result, dict) else {"value": result}
    return {"value": str(value)}


def _provider_error(exc: Exception) -> ProviderError:
    status_code = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    code = str(getattr(exc, "code", "") or "").lower()
    error_type = str(getattr(exc, "type", "") or "").lower()
    provider_message = ""
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            code = str(error.get("code", code) or "").lower()
            error_type = str(error.get("type", error_type) or "").lower()
            provider_message = str(error.get("message", "") or "").lower()

    # Providers commonly return both temporary throttling and permanent
    # billing/credit failures as HTTP 429. Check terminal quota signals before
    # the generic 429 branch so exhausted credit is never retried.
    terminal_quota_codes = {
        "credit_balance_exhausted",
        "insufficient_quota",
        "billing_hard_limit_reached",
        "billing_not_active",
        "monthly_budget_exceeded",
        "usage_limit_reached",
    }
    terminal_quota_types = {
        "insufficient_quota",
        "billing_error",
    }
    terminal_message_markers = (
        "no credits remaining",
        "credit balance exhausted",
        "add credits to continue",
        "billing hard limit",
        "billing is not active",
    )
    is_terminal_quota = (
        code in terminal_quota_codes
        or error_type in terminal_quota_types
        or any(marker in provider_message for marker in terminal_message_markers)
    )
    if status_code in {401, 403}:
        category, retryable = "authentication", False
    elif is_terminal_quota:
        category, retryable = "quota", False
    elif status_code == 429:
        category, retryable = "rate_limit", True
    elif isinstance(status_code, int) and status_code >= 500:
        category, retryable = "server", True
    elif exc.__class__.__name__ in {"APITimeoutError", "APIConnectionError"}:
        category, retryable = "network", True
    else:
        category, retryable = "provider", False
    # Do not include request headers or any client object that might contain a key.
    return ProviderError(
        f"{exc.__class__.__name__}: {exc}",
        retryable=retryable,
        category=category,
        status_code=status_code,
    )


def parse_type_and_correction(raw_text: str) -> tuple[int, str]:
    """Parse the exact two-field JSON contract used by the VGEC proxy runs."""
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ProviderError(
            f"Model output is not valid JSON: {exc.msg}",
            retryable=False,
            category="invalid_output",
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"type_id", "fixed_sentence"}:
        raise ProviderError(
            "Model JSON must contain exactly type_id and fixed_sentence",
            retryable=False,
            category="invalid_output",
        )
    type_id = payload["type_id"]
    fixed_sentence = payload["fixed_sentence"]
    if isinstance(type_id, bool) or not isinstance(type_id, int) or type_id not in range(5):
        raise ProviderError(
            "Model JSON type_id must be an integer from 0 to 4",
            retryable=False,
            category="invalid_output",
        )
    if not isinstance(fixed_sentence, str) or not fixed_sentence.strip():
        raise ProviderError(
            "Model JSON fixed_sentence must be a non-empty string",
            retryable=False,
            category="invalid_output",
        )
    return type_id, fixed_sentence


WRAPPER_PATTERN = re.compile(
    r"^(?:fixed_sentence|corrected_sentence|output|answer|câu\s+(?:đã\s+)?sửa)\s*:",
    flags=re.IGNORECASE,
)


def parse_correction_only(raw_text: str, finish_reason: str | None = None) -> str:
    """Apply the deliberately minimal correction-only text contract."""
    if finish_reason in {"length", "max_tokens"}:
        raise ProviderError(
            "Model output was truncated by the output-token limit",
            retryable=False,
            category="invalid_output",
            raw_output=raw_text,
        )
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ProviderError(
            "Model returned an empty output",
            retryable=False,
            category="invalid_output",
            raw_output=raw_text if isinstance(raw_text, str) else None,
        )
    prediction = raw_text.strip()
    if prediction.startswith("```") or prediction.endswith("```"):
        raise ProviderError(
            "Model output used a Markdown code fence",
            retryable=False,
            category="invalid_output",
            raw_output=raw_text,
        )
    try:
        wrapped = json.loads(prediction)
    except json.JSONDecodeError:
        wrapped = None
    if isinstance(wrapped, (dict, list)):
        raise ProviderError(
            "Model output used a JSON wrapper",
            retryable=False,
            category="invalid_output",
            raw_output=raw_text,
        )
    if WRAPPER_PATTERN.match(prediction):
        raise ProviderError(
            "Model output used an explicit answer wrapper",
            retryable=False,
            category="invalid_output",
            raw_output=raw_text,
        )
    return prediction


def _chat_completions_url(base_url: str) -> str:
    parsed = urlsplit(base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("API_BASE_URL must be an absolute http(s) URL")
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        endpoint_path = path
    elif path.endswith("/v1"):
        endpoint_path = f"{path}/chat/completions"
    else:
        endpoint_path = f"{path}/v1/chat/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, endpoint_path, "", ""))


def build_type_and_correction_payload(
    *,
    model: str,
    system_prompt: str,
    user_prompt_template: str,
    input_text: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt_template.format(sentence=input_text),
            },
        ],
        "temperature": 0,
        "max_tokens": max_output_tokens,
        "response_format": {"type": "json_object"},
    }


def build_correction_only_payload(
    *,
    model: str,
    system_prompt: str,
    user_prompt_template: str,
    input_text: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    payload = build_type_and_correction_payload(
        model=model,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        input_text=input_text,
        max_output_tokens=max_output_tokens,
    )
    payload.pop("response_format")
    return payload


def _http_provider_error(status_code: int, body: bytes, api_key: str) -> ProviderError:
    text = body.decode("utf-8", errors="replace").replace(api_key, "[REDACTED]")
    message = ""
    code = ""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        message = text[:500].strip()
    else:
        if isinstance(payload, dict):
            error = payload.get("error", payload)
            if isinstance(error, dict):
                message = str(error.get("message", "") or "")[:500]
                code = str(error.get("code", "") or "").lower()
            elif isinstance(error, str):
                message = error[:500]
    lowered = message.lower()
    quota = code in {"insufficient_quota", "credit_balance_exhausted"} or any(
        marker in lowered
        for marker in ("no credits remaining", "credit balance exhausted", "quota")
    )
    if status_code in {401, 403}:
        category, retryable = "authentication", False
    elif quota:
        category, retryable = "quota", False
    elif status_code == 429:
        category, retryable = "rate_limit", True
    elif status_code >= 500:
        category, retryable = "server", True
    else:
        category, retryable = "provider", False
    return ProviderError(
        f"HTTP {status_code}: {message or 'provider request failed'}",
        retryable=retryable,
        category=category,
        status_code=status_code,
    )


class LiteLLMTypeAndCorrectionProvider:
    provider_name = "litellm_openai_compatible"
    method = "zero_shot_type_and_correction"

    def __init__(
        self,
        model: str,
        system_prompt: str,
        user_prompt_template: str,
        max_output_tokens: int,
        timeout: float,
        output_mode: str = "type_and_correction_json",
    ) -> None:
        _load_dotenv_if_available()
        base_url = os.environ.get("API_BASE_URL")
        api_key = os.environ.get("API_KEY")
        missing = [
            name
            for name, value in (("API_BASE_URL", base_url), ("API_KEY", api_key))
            if not value
        ]
        if missing:
            raise ValueError(f"Missing proxy environment variables: {', '.join(missing)}")
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        if output_mode not in {"type_and_correction_json", "correction_only_text"}:
            raise ValueError(f"Unsupported proxy output mode: {output_mode}")
        self.output_mode = output_mode
        self.method = (
            "zero_shot_correction_only"
            if output_mode == "correction_only_text"
            else "zero_shot_type_and_correction"
        )
        self.endpoint = _chat_completions_url(str(base_url))
        self._api_key = str(api_key)

    def configuration(self) -> dict[str, Any]:
        parsed = urlsplit(self.endpoint)
        result = {
            "protocol": "openai_compatible_chat_completions",
            "endpoint": urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", "")),
            "authentication": "x-litellm-api-key",
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
            "output_mode": self.output_mode,
        }
        if self.output_mode == "type_and_correction_json":
            result["response_format"] = {"type": "json_object"}
        return result

    def generate(self, input_text: str) -> ProviderResult:
        payload_builder = (
            build_correction_only_payload
            if self.output_mode == "correction_only_text"
            else build_type_and_correction_payload
        )
        request_payload = payload_builder(
            model=self.model,
            system_prompt=self.system_prompt,
            user_prompt_template=self.user_prompt_template,
            input_text=input_text,
            max_output_tokens=self.max_output_tokens,
        )
        request = Request(
            self.endpoint,
            data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "x-litellm-api-key": self._api_key,
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                http_status = response.status
                response_body = response.read()
        except HTTPError as exc:
            raise _http_provider_error(exc.code, exc.read(), self._api_key) from exc
        except (URLError, TimeoutError) as exc:
            raise ProviderError(
                f"{exc.__class__.__name__}: {exc}",
                retryable=True,
                category="network",
            ) from exc
        try:
            payload = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderError(
                "Proxy response is not valid UTF-8 JSON",
                retryable=False,
                category="invalid_response",
                status_code=http_status,
            ) from exc
        choices = payload.get("choices") if isinstance(payload, dict) else None
        choice = choices[0] if isinstance(choices, list) and choices else None
        message = choice.get("message") if isinstance(choice, dict) else None
        raw_output = message.get("content") if isinstance(message, dict) else None
        if not isinstance(raw_output, str):
            raise ProviderError(
                "Proxy response did not contain choices[0].message.content",
                retryable=False,
                category="invalid_response",
                status_code=http_status,
            )
        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
        if self.output_mode == "correction_only_text":
            predicted_type_id = None
            fixed_sentence = parse_correction_only(raw_output, finish_reason)
        else:
            predicted_type_id, fixed_sentence = parse_type_and_correction(raw_output)
        actual_model = payload.get("model")
        if not isinstance(actual_model, str) or not actual_model:
            actual_model = None
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        request_id = payload.get("id")
        if not isinstance(request_id, str):
            request_id = None
        return ProviderResult(
            raw_text=fixed_sentence,
            model_id=actual_model,
            request_id=request_id,
            usage=usage,
            raw_response={
                "id": request_id,
                "model": actual_model,
                "finish_reason": finish_reason,
                "message_content": raw_output,
                "usage": usage,
                "http_status": http_status,
            },
            predicted_type_id=predicted_type_id,
            raw_output=raw_output,
            http_status=http_status,
        )


class MockProvider:
    """Local deterministic provider used by --dry-run; it never opens a network client."""

    provider_name = "mock"
    method = "dry_run_mock"

    def __init__(
        self,
        selected_model: str,
        prompt: str | None,
        output_mode: str | None = None,
    ) -> None:
        self.selected_model = selected_model
        self.prompt = prompt
        self.output_mode = output_mode

    def generate(self, input_text: str) -> ProviderResult:
        if self.selected_model == "bartpho-pretrained":
            payload: dict[str, Any] = {"input": input_text}
        else:
            payload = {"instructions": self.prompt, "input": input_text}
        return ProviderResult(
            raw_text=input_text,
            model_id=f"mock:{self.selected_model}",
            request_id=None,
            usage={"input_tokens": None, "output_tokens": None, "total_tokens": None},
            raw_response={"mock": True, "request_payload": payload},
            predicted_type_id=(
                0 if self.output_mode == "type_and_correction_json" else None
            ),
        )


class OpenAIResponsesProvider:
    provider_name = "openai"
    method = "zero_shot_prompt"

    def __init__(self, model: str, prompt: str, max_output_tokens: int, timeout: float):
        _load_dotenv_if_available()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for a real OpenAI run")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt before a real API run") from exc
        self.model = model
        self.prompt = prompt
        self.max_output_tokens = max_output_tokens
        self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)

    def generate(self, input_text: str) -> ProviderResult:
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=self.prompt,
                input=input_text,
                max_output_tokens=self.max_output_tokens,
            )
        except Exception as exc:  # SDK exception hierarchy varies by version.
            raise _provider_error(exc) from exc
        raw_text = response.output_text
        if not isinstance(raw_text, str):
            raise ProviderError(
                "OpenAI response did not contain string output_text",
                retryable=False,
                category="invalid_response",
            )
        return ProviderResult(
            raw_text=raw_text,
            model_id=str(getattr(response, "model", self.model)),
            request_id=(
                getattr(response, "_request_id", None)
                or getattr(response, "request_id", None)
                or getattr(response, "id", None)
            ),
            usage=_model_dump(getattr(response, "usage", None)),
            raw_response={
                "id": getattr(response, "id", None),
                "model": getattr(response, "model", self.model),
                "status": getattr(response, "status", None),
                "output_text": raw_text,
                "usage": _model_dump(getattr(response, "usage", None)),
            },
        )


class QwenCompatibleProvider:
    provider_name = "qwen_openai_compatible"
    method = "zero_shot_prompt"

    def __init__(self, prompt: str, max_output_tokens: int, timeout: float):
        _load_dotenv_if_available()
        model = os.environ.get("QWEN_MODEL")
        base_url = os.environ.get("QWEN_BASE_URL")
        api_key = os.environ.get("QWEN_API_KEY")
        missing = [
            name
            for name, value in (
                ("QWEN_MODEL", model),
                ("QWEN_BASE_URL", base_url),
                ("QWEN_API_KEY", api_key),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing Qwen environment variables: {', '.join(missing)}")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt before a real API run") from exc
        self.model = str(model)
        self.prompt = prompt
        self.max_output_tokens = max_output_tokens
        self.client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0
        )

    def generate(self, input_text: str) -> ProviderResult:
        # Most Qwen providers expose OpenAI-compatible Chat Completions. We pass no
        # temperature/top_p because support differs by endpoint.
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": input_text},
                ],
                max_tokens=self.max_output_tokens,
            )
        except Exception as exc:  # SDK exception hierarchy varies by version.
            raise _provider_error(exc) from exc
        choice = response.choices[0] if response.choices else None
        raw_text = None if choice is None else choice.message.content
        if not isinstance(raw_text, str):
            raise ProviderError(
                "Qwen response did not contain a string message",
                retryable=False,
                category="invalid_response",
            )
        actual_model = str(getattr(response, "model", self.model))
        return ProviderResult(
            raw_text=raw_text,
            model_id=actual_model,
            request_id=(
                getattr(response, "_request_id", None)
                or getattr(response, "request_id", None)
                or getattr(response, "id", None)
            ),
            usage=_model_dump(getattr(response, "usage", None)),
            raw_response={
                "id": getattr(response, "id", None),
                "model": actual_model,
                "finish_reason": getattr(choice, "finish_reason", None),
                "message_content": raw_text,
                "usage": _model_dump(getattr(response, "usage", None)),
            },
        )


def build_api_provider(
    model: str,
    prompt: str,
    max_output_tokens: int,
    timeout: float,
    user_prompt_template: str | None = None,
    output_mode: str = "type_and_correction_json",
) -> GenerationProvider:
    if model in {"azure-4.1", "azure-4o"}:
        if user_prompt_template is None:
            raise ValueError("Proxy models require a user prompt template")
        return LiteLLMTypeAndCorrectionProvider(
            model,
            prompt,
            user_prompt_template,
            max_output_tokens,
            timeout,
            output_mode,
        )
    if model in {"gpt-4.1", "gpt-4o"}:
        return OpenAIResponsesProvider(model, prompt, max_output_tokens, timeout)
    if model == "qwen":
        return QwenCompatibleProvider(prompt, max_output_tokens, timeout)
    raise ValueError(f"Unsupported API model: {model}")
