"""
model_provider_router.py

Replaceable AI-reply provider layer for the Ascendant Path agent model.

Existing agents can keep building OpenAI Responses-style request dictionaries,
then call:

    resp = create_response_for_stage("pm.generate", req)

The router reads provider_config.py and dispatches to:
- OpenAI Responses API
- local Ollama
- GPTWeb/browser automation, protected by a global single-thread lock

It returns a minimal response-like object with:
- output_text
- id
- provider
- profile_name
- model
- usage

This keeps agent code provider-agnostic.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Set

from provider_config import get_stage_profile, get_provider_profile, ROUTING_CONSTRAINTS
from trace_utils import trace_event
from response_parser import extract_json_text, strip_gptweb_guard, loads_json_lenient

try:
    from context_compressor import compress_request_for_stage, emergency_deterministic_compress_request_for_stage
except Exception:
    compress_request_for_stage = None  # type: ignore
    emergency_deterministic_compress_request_for_stage = None  # type: ignore

# Browser/GPTWeb automation controls a real browser session. It must be serialized.
#
# Two layers are used:
#   1. _BROWSER_PROVIDER_LOCK protects threads inside this Python process.
#   2. a lock file protects multiple Python processes using the same browser session.
#
# This deliberately does NOT try to parallelize browser calls with multiple tabs.
# Parallel browser automation is fragile because tabs share login state, rate limits,
# selectors, and conversation context. OpenAI/Ollama should be used for parallel stages.
_BROWSER_PROVIDER_LOCK = threading.Lock()
_BROWSER_LOCK_OWNER = None


def _browser_lock_timeout_sec() -> float:
    return float(os.getenv("GPTWEB_LOCK_TIMEOUT_SEC", "900"))


def _browser_lock_stale_sec() -> float:
    return float(os.getenv("GPTWEB_LOCK_STALE_SEC", "3600"))


def _browser_lock_path() -> Path:
    raw = os.getenv("GPTWEB_LOCK_PATH", "")
    if raw.strip():
        return Path(raw).expanduser()
    return Path("outputs") / "GPTWeb_output" / "browser_provider.lock"


def _try_acquire_file_lock(lock_path: Path, payload: Mapping[str, Any]) -> bool:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    except OSError:
        # If the filesystem lock cannot be created, fall back to thread lock only.
        return True

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(dict(payload), ensure_ascii=False, indent=2))
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
    return True


def _release_file_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass


def _remove_stale_file_lock(lock_path: Path) -> None:
    try:
        if not lock_path.exists():
            return
        age = time.time() - lock_path.stat().st_mtime
        if age >= _browser_lock_stale_sec():
            lock_path.unlink(missing_ok=True)
    except Exception:
        pass


@contextmanager
def _browser_provider_lock(stage_key: str, profile_name: str) -> Iterator[None]:
    """Serialize GPTWeb/browser calls across threads and best-effort across processes."""
    global _BROWSER_LOCK_OWNER

    if not ROUTING_CONSTRAINTS.get("browser_single_thread_lock_required", True):
        yield
        return

    timeout = _browser_lock_timeout_sec()
    start = time.time()
    owner = {
        "pid": os.getpid(),
        "stage_key": stage_key,
        "profile_name": profile_name,
        "created_at_epoch": start,
    }

    # First protect threads in the current Python process.
    acquired_thread_lock = _BROWSER_PROVIDER_LOCK.acquire(timeout=timeout)
    if not acquired_thread_lock:
        raise TimeoutError(
            f"Timed out after {timeout:.0f}s waiting for browser provider thread lock. "
            f"Current owner: {_BROWSER_LOCK_OWNER!r}"
        )

    lock_path = _browser_lock_path()
    acquired_file_lock = False
    try:
        _BROWSER_LOCK_OWNER = owner

        # Then protect other Python processes using the same attached browser.
        while True:
            _remove_stale_file_lock(lock_path)
            acquired_file_lock = _try_acquire_file_lock(lock_path, owner)
            if acquired_file_lock:
                break
            if time.time() - start >= timeout:
                raise TimeoutError(
                    f"Timed out after {timeout:.0f}s waiting for browser provider file lock: {lock_path}"
                )
            time.sleep(0.5)

        yield
    finally:
        if acquired_file_lock:
            _release_file_lock(lock_path)
        _BROWSER_LOCK_OWNER = None
        _BROWSER_PROVIDER_LOCK.release()


@dataclass
class RoutedModelResponse:
    output_text: str
    id: str
    provider: str
    profile_name: str
    model: Optional[str]
    usage: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None


def _sha_id(prefix: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{int(time.time())}-{digest}"


def _input_to_text(value: Any) -> str:
    """Flatten Responses API `input` into a text prompt for non-OpenAI providers."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, Mapping):
                role = item.get("role", "user")
                content = item.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(str(x.get("text", x)) if isinstance(x, Mapping) else str(x) for x in content)
                parts.append(f"[{role}]\n{content}")
            else:
                parts.append(str(item))
        return "\n\n".join(parts)
    return json.dumps(value, ensure_ascii=False, indent=2)


def _schema_instruction_from_req(req: Mapping[str, Any]) -> str:
    fmt = ((req.get("text") or {}).get("format") or {}) if isinstance(req.get("text"), Mapping) else {}
    if fmt.get("type") != "json_schema":
        return ""
    schema = fmt.get("schema")
    name = fmt.get("name", "output")
    return (
        f"\n\nOUTPUT FORMAT REQUIREMENT:\n"
        f"Return ONLY valid JSON matching the schema named {name}. "
        f"No markdown, no commentary.\nSCHEMA:\n"
        + json.dumps(schema, ensure_ascii=False)
    )


def _req_to_prompt(req: Mapping[str, Any]) -> str:
    instructions = req.get("instructions", "")
    input_text = _input_to_text(req.get("input", ""))
    schema_hint = _schema_instruction_from_req(req)
    if instructions:
        return f"SYSTEM / INSTRUCTIONS:\n{instructions}\n\nUSER INPUT:\n{input_text}{schema_hint}"
    return f"USER INPUT:\n{input_text}{schema_hint}"


def _estimate_tokens(text: str) -> int:
    # Lightweight estimate only. Good enough for local/provider accounting.
    return max(1, len(text) // 4)



def _safe_model_dump(value: Any) -> Any:
    """Best-effort conversion of SDK objects into JSON-safe dictionaries."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _safe_model_dump(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_model_dump(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _safe_model_dump(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _safe_model_dump(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _safe_model_dump(value.__dict__)
        except Exception:
            pass
    return repr(value)


def _extract_cached_tokens(usage: Mapping[str, Any]) -> Optional[int]:
    """Extract cached input token count from common OpenAI usage shapes."""
    details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details")
    if isinstance(details, Mapping):
        value = details.get("cached_tokens")
        if isinstance(value, int):
            return value
    value = usage.get("cached_tokens")
    if isinstance(value, int):
        return value
    return None


def _apply_openai_prompt_cache_policy(req: Dict[str, Any], profile: Mapping[str, Any]) -> Dict[str, Any]:
    """Attach OpenAI prompt-cache routing hints when configured."""
    if not profile.get("prompt_cache_enabled"):
        return req

    updated = dict(req)
    cache_key = str(profile.get("prompt_cache_key") or "").strip()
    if cache_key:
        updated["prompt_cache_key"] = cache_key

    retention = str(profile.get("prompt_cache_retention") or "").strip()
    if retention:
        updated["prompt_cache_retention"] = retention

    return updated


def _remove_openai_prompt_cache_policy(req: Dict[str, Any]) -> Dict[str, Any]:
    """Remove cache-control parameters for SDK compatibility retry."""
    updated = dict(req)
    updated.pop("prompt_cache_key", None)
    updated.pop("prompt_cache_retention", None)
    return updated


def _openai_create(req: Dict[str, Any], profile: Mapping[str, Any]) -> RoutedModelResponse:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        # Keep compatibility with the original project convention, but do not
        # require this file to exist. Non-OpenAI providers should work without it.
        key_path = os.path.join(os.getcwd(), "openai_api_key")
        if os.path.exists(key_path):
            os.environ["OPENAI_API_KEY"] = open(key_path, "r", encoding="utf-8").read().strip()

    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError(
            "OpenAI provider selected but OPENAI_API_KEY is not set. "
            "Set OPENAI_API_KEY or choose an Ollama/browser provider in provider_config.py."
        )

    client = OpenAI()

    req = dict(req)
    profile_model = profile.get("model")
    if isinstance(profile_model, str) and profile_model and not profile_model.startswith("SET_"):
        req["model"] = profile_model

    if profile.get("supports_tools") is False:
        req.pop("tools", None)

    cache_policy_requested = bool(profile.get("prompt_cache_enabled"))
    req_with_cache = _apply_openai_prompt_cache_policy(req, profile)

    try:
        raw = client.responses.create(**req_with_cache)
        cache_policy_applied = cache_policy_requested
        used_req = req_with_cache
    except TypeError as exc:
        # Some installed SDK versions can lag behind API parameters. Retry without
        # prompt-cache controls so caching support never breaks normal execution.
        if "prompt_cache" not in str(exc):
            raise
        trace_event(
            "prompt_cache_policy_retry_without_parameters",
            stage_key=profile.get("stage_key"),
            profile_name=profile.get("profile_name"),
            model=req.get("model"),
            error=str(exc),
        )
        used_req = _remove_openai_prompt_cache_policy(req_with_cache)
        raw = client.responses.create(**used_req)
        cache_policy_applied = False

    output_text = getattr(raw, "output_text", "") or ""
    usage_obj = getattr(raw, "usage", None)
    usage = _safe_model_dump(usage_obj) if usage_obj is not None else {}
    if not isinstance(usage, dict):
        usage = {"raw_usage": usage}
    usage["prompt_cache"] = {
        "requested": cache_policy_requested,
        "applied": cache_policy_applied,
        "prompt_cache_key": used_req.get("prompt_cache_key"),
        "prompt_cache_retention": used_req.get("prompt_cache_retention"),
        "cached_tokens": _extract_cached_tokens(usage),
    }

    return RoutedModelResponse(
        output_text=output_text,
        id=getattr(raw, "id", _sha_id("openai", output_text)),
        provider="openai",
        profile_name=str(profile.get("profile_name")),
        model=used_req.get("model"),
        usage=usage,
        raw=raw,
    )


def _ollama_create(req: Dict[str, Any], profile: Mapping[str, Any]) -> RoutedModelResponse:
    host = str(profile.get("host") or "http://localhost:11434").rstrip("/")
    model = str(profile.get("model") or "llama3.2:1b")
    prompt = _req_to_prompt(req)
    max_tokens = int(profile.get("max_output_tokens") or 1200)
    temperature = profile.get("temperature")

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    if temperature is not None:
        payload["options"]["temperature"] = temperature

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{host}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(os.getenv("OLLAMA_TIMEOUT_SEC", "180"))) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama provider call failed. Is Ollama running at {host}? {exc}") from exc

    output_text = ((raw.get("message") or {}).get("content") or "").strip()
    return RoutedModelResponse(
        output_text=output_text,
        id=_sha_id("ollama", prompt + output_text),
        provider="ollama",
        profile_name=str(profile.get("profile_name")),
        model=model,
        usage={
            "input_tokens": _estimate_tokens(prompt),
            "output_tokens": _estimate_tokens(output_text),
            "source": "estimated",
        },
        raw=raw,
    )


def _browser_create(req: Dict[str, Any], profile: Mapping[str, Any]) -> RoutedModelResponse:
    try:
        from gpt_web_collector import GPTWeb
    except Exception as exc:
        raise RuntimeError(
            "Browser provider requires gpt_web_collector.py in the project root."
        ) from exc

    base_prompt = _req_to_prompt(req)
    if profile.get("append_text_only_suffix"):
        suffix = ROUTING_CONSTRAINTS.get("browser_text_only_suffix", "")
        if suffix and suffix not in base_prompt:
            base_prompt = base_prompt.rstrip() + "\n\n" + str(suffix)

    url = str(profile.get("url") or os.getenv("GPTWEB_URL", "")).strip()
    if not url:
        raise RuntimeError("Browser provider selected but GPTWEB_URL/profile url is empty.")

    connect_cdp_url = str(profile.get("connect_cdp_url") or os.getenv("GPTWEB_CDP_URL", "")).strip()
    request_id = f"GPTWEB_REQ_{uuid.uuid4().hex[:16]}"
    prefix = f"[{request_id}]"
    prompt = (
        base_prompt.rstrip()
        + "\n\nGPTWEB REQUEST-ID GUARD:\n"
        + f"You MUST start your reply with exactly this prefix on the first line: {prefix}\n"
        + "After that prefix, provide the requested answer. If JSON was requested, put the JSON immediately after the prefix line."
    )

    def _strip_guard(raw_reply: Any) -> str:
        text = str(raw_reply or "").strip()

        # Normal case: assistant reply begins with the current request guard.
        if text.startswith(prefix):
            return text[len(prefix):].lstrip()

        lines = text.splitlines()
        if lines and lines[0].strip() == prefix:
            return "\n".join(lines[1:]).strip()

        # Browser selectors can occasionally capture a larger block containing
        # both the user prompt and the assistant response. In that case the
        # current guard can appear later in the extracted text. Use the last
        # occurrence of the exact current prefix; do not accept any other guard.
        idx = text.rfind(prefix)
        if idx >= 0:
            return text[idx + len(prefix):].lstrip("\r\n \t")

        raise RuntimeError(
            f"GPTWeb response id mismatch for {request_id}. The browser reply did not contain the current guard {prefix!r}; refusing possible stale output."
        )

    reply = None
    last_error: Optional[Exception] = None
    with _browser_provider_lock(
        stage_key=str(profile.get("stage_key") or "unknown_stage"),
        profile_name=str(profile.get("profile_name") or "unknown_profile"),
    ):
        for attempt in range(1, int(os.getenv("GPTWEB_REQUEST_ID_MAX_ATTEMPTS", "1")) + 1):
            try:
                reply = GPTWeb(
                    prompt if attempt == 1 else prompt + f"\n\nPrevious attempt failed request-id validation. Begin with {prefix} exactly.",
                    url=url,
                    headless=False if connect_cdp_url else True,
                    connect_cdp_url=connect_cdp_url,
                )
                output_text = _strip_guard(reply)
                break
            except Exception as exc:
                last_error = exc
                trace_event(
                    "browser_request_id_validation_failed",
                    stage_key=profile.get("stage_key"),
                    profile_name=profile.get("profile_name"),
                    request_id=request_id,
                    attempt=attempt,
                    error=str(exc),
                )
        else:
            raise RuntimeError(f"Browser provider failed request-id validation after retries: {last_error}")

    return RoutedModelResponse(
        output_text=output_text,
        id=_sha_id("browser", prompt + output_text),
        provider="browser",
        profile_name=str(profile.get("profile_name")),
        model=str(profile.get("model") or "private_website"),
        usage={
            "input_tokens": _estimate_tokens(prompt),
            "output_tokens": _estimate_tokens(output_text),
            "source": "estimated",
            "gptweb_request_id": request_id,
            "gptweb_prefix_validated": True,
        },
        raw=reply,
    )





def _schema_type_name(schema_type: Any) -> str:
    if isinstance(schema_type, list):
        return "|".join(str(x) for x in schema_type)
    return str(schema_type or "any")


def _value_matches_json_type(value: Any, expected: Any) -> bool:
    """Small JSON-schema type checker used before accepting provider output.

    This is intentionally not a full jsonschema implementation. It validates the
    failure classes that crash agents most often: wrong top-level type, missing
    required fields, bad nested object/array types, enum/const violations, and
    unexpected keys when additionalProperties is false.
    """
    if isinstance(expected, list):
        return any(_value_matches_json_type(value, x) for x in expected)
    if expected in (None, "any"):
        return True
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _json_schema_errors(value: Any, schema: Any, *, path: str = "$", max_errors: int = 12) -> List[str]:
    """Best-effort, dependency-free schema validation for model JSON outputs."""
    if not isinstance(schema, Mapping):
        return []

    errors: List[str] = []

    def add(msg: str) -> None:
        if len(errors) < max_errors:
            errors.append(msg)

    if "const" in schema and value != schema.get("const"):
        add(f"{path}: expected const {schema.get('const')!r}")
        return errors
    if "enum" in schema and isinstance(schema.get("enum"), list) and value not in schema.get("enum", []):
        add(f"{path}: value {value!r} not in enum")
        return errors

    if "anyOf" in schema and isinstance(schema.get("anyOf"), list):
        if not any(not _json_schema_errors(value, sub, path=path, max_errors=1) for sub in schema["anyOf"]):
            add(f"{path}: value does not match anyOf")
        return errors
    if "oneOf" in schema and isinstance(schema.get("oneOf"), list):
        matches = sum(1 for sub in schema["oneOf"] if not _json_schema_errors(value, sub, path=path, max_errors=1))
        if matches != 1:
            add(f"{path}: value matches {matches} oneOf schemas")
        return errors

    expected_type = schema.get("type")
    if expected_type is not None and not _value_matches_json_type(value, expected_type):
        add(f"{path}: expected {_schema_type_name(expected_type)}, got {type(value).__name__}")
        return errors

    if isinstance(value, dict):
        props = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        for key in required:
            if key not in value:
                add(f"{path}.{key}: missing required field")
        if schema.get("additionalProperties") is False and props:
            for key in value:
                if key not in props:
                    add(f"{path}.{key}: unexpected field")
        for key, subschema in props.items():
            if key in value:
                errors.extend(_json_schema_errors(value[key], subschema, path=f"{path}.{key}", max_errors=max_errors - len(errors)))
                if len(errors) >= max_errors:
                    break

    if isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema.get("minItems") or 0):
            add(f"{path}: expected at least {schema.get('minItems')} items")
        if "maxItems" in schema and len(value) > int(schema.get("maxItems") or len(value)):
            add(f"{path}: expected at most {schema.get('maxItems')} items")
        items_schema = schema.get("items")
        if isinstance(items_schema, Mapping):
            for idx, item in enumerate(value[:25]):
                errors.extend(_json_schema_errors(item, items_schema, path=f"{path}[{idx}]", max_errors=max_errors - len(errors)))
                if len(errors) >= max_errors:
                    break

    return errors[:max_errors]




def _coerce_object_list_for_schema(value: Any, schema: Optional[Mapping[str, Any]]) -> Any:
    """Recover safe list-wrapped object payloads before schema validation."""
    if not isinstance(schema, Mapping) or schema.get("type") != "object" or not isinstance(value, list):
        return value
    if len(value) == 1 and isinstance(value[0], dict):
        return dict(value[0])
    if value and all(isinstance(item, dict) for item in value):
        merged: Dict[str, Any] = {}
        for item in value:
            for key, item_value in item.items():
                if key in merged and merged[key] != item_value:
                    return value
                merged[key] = item_value
        if merged:
            return merged
    return value


def _schema_from_req(req: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    fmt = ((req.get("text") or {}).get("format") or {}) if isinstance(req.get("text"), Mapping) else {}
    schema = fmt.get("schema") if isinstance(fmt, Mapping) else None
    return schema if isinstance(schema, Mapping) else None


def _fallback_json_response(
    *,
    response: RoutedModelResponse,
    req: Dict[str, Any],
    profile: Mapping[str, Any],
    attempted_profiles: Set[str],
    reason: str,
    validation_errors: Optional[List[str]] = None,
) -> RoutedModelResponse:
    fallback_name = str(profile.get("fallback_profile") or "").strip()
    if (not fallback_name or fallback_name in attempted_profiles) and profile.get("provider") != "openai":
        fallback_name = "openai_fast_json"

    if not fallback_name or fallback_name in attempted_profiles:
        detail = f" Validation errors: {validation_errors}" if validation_errors else ""
        raise RuntimeError(
            f"Provider {profile.get('profile_name')!r} returned unusable JSON output for strict JSON stage "
            f"{profile.get('stage_key')!r} ({reason}), and no usable fallback profile is available.{detail}\n\nRAW:\n{response.output_text}"
        )

    trace_event(
        "provider_json_fallback",
        stage_key=profile.get("stage_key"),
        from_profile=profile.get("profile_name"),
        from_provider=response.provider,
        fallback_profile=fallback_name,
        reason=reason,
        validation_errors=validation_errors or [],
    )

    fallback_profile = get_provider_profile(fallback_name)
    fallback_profile.update({
        "stage_key": profile.get("stage_key"),
        "requires_strict_json": True,
        "criticality": profile.get("criticality"),
    })
    attempted_profiles.add(fallback_name)
    fallback_response = _dispatch_provider(req, fallback_profile)
    fallback_response = _normalize_or_fallback(
        response=fallback_response,
        req=req,
        profile=fallback_profile,
        attempted_profiles=attempted_profiles,
    )
    fallback_response.usage = dict(fallback_response.usage or {})
    fallback_response.usage["fallback_from_profile"] = profile.get("profile_name")
    return fallback_response


def _requires_json(req: Mapping[str, Any], profile: Mapping[str, Any]) -> bool:
    fmt = ((req.get("text") or {}).get("format") or {}) if isinstance(req.get("text"), Mapping) else {}
    return bool(profile.get("requires_strict_json") or fmt.get("type") == "json_schema")


def _strip_code_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = value.strip("`").strip()
        if value.lower().startswith("json"):
            value = value[4:].strip()
    return value


def _extract_json_candidate(text: str) -> Optional[str]:
    """Return canonical JSON text if possible; tolerate code fences, GPTWeb guards, and prose."""
    return extract_json_text(text)


def _dispatch_provider(req: Dict[str, Any], profile: Mapping[str, Any]) -> RoutedModelResponse:
    provider = profile.get("provider")
    if provider == "deterministic":
        raise RuntimeError(f"Stage {profile.get('stage_key')!r} is deterministic and must not call an LLM provider.")
    if provider == "openai":
        return _openai_create(req, profile)
    if provider == "ollama":
        return _ollama_create(req, profile)
    if provider == "browser":
        return _browser_create(req, profile)
    raise RuntimeError(f"Unsupported provider {provider!r} for stage {profile.get('stage_key')!r}.")


def _normalize_or_fallback(
    *,
    response: RoutedModelResponse,
    req: Dict[str, Any],
    profile: Mapping[str, Any],
    attempted_profiles: Set[str],
) -> RoutedModelResponse:
    """For strict JSON stages, parse/extract JSON, validate schema shape, or fall back.

    The old router accepted any valid JSON candidate. That let outputs like
    ``[81]`` pass router validation for PM, then crash at the agent parser. This
    function now validates the candidate against the request's JSON schema before
    accepting it.
    """
    if not _requires_json(req, profile):
        return response

    candidate = _extract_json_candidate(response.output_text)
    if candidate is None:
        return _fallback_json_response(
            response=response,
            req=req,
            profile=profile,
            attempted_profiles=attempted_profiles,
            reason="non_json_output_for_strict_json_stage",
        )

    try:
        parsed = loads_json_lenient(candidate)
    except Exception as exc:
        return _fallback_json_response(
            response=response,
            req=req,
            profile=profile,
            attempted_profiles=attempted_profiles,
            reason=f"extracted_json_failed_to_parse:{exc.__class__.__name__}",
        )

    schema = _schema_from_req(req)
    parsed = _coerce_object_list_for_schema(parsed, schema)
    validation_errors = _json_schema_errors(parsed, schema) if schema else []
    if validation_errors:
        return _fallback_json_response(
            response=response,
            req=req,
            profile=profile,
            attempted_profiles=attempted_profiles,
            reason="json_schema_shape_mismatch",
            validation_errors=validation_errors,
        )

    response.output_text = json.dumps(parsed, ensure_ascii=False)
    response.usage = dict(response.usage or {})
    response.usage["json_validation"] = "parsed_extracted_and_schema_validated" if schema else "parsed_or_extracted"
    return response


def create_response_for_stage(
    stage_key: str,
    req: Dict[str, Any],
    *,
    run_mode: Optional[str] = None,
    model_override: Optional[str] = None,
) -> RoutedModelResponse:
    """Create a model response for a workflow stage using provider_config.py."""
    profile = get_stage_profile(stage_key, run_mode=run_mode)
    req = dict(req)
    if model_override and os.getenv("ASCENDANT_RESPECT_AGENT_MODEL", "0") == "1":
        req["model"] = model_override

    trace_event(
        "provider_call_start",
        stage_key=profile.get("stage_key") or stage_key,
        provider=profile.get("provider"),
        profile_name=profile.get("profile_name"),
        model=profile.get("model"),
        run_mode=profile.get("run_mode"),
        requires_strict_json=profile.get("requires_strict_json"),
        max_input_tokens=profile.get("max_input_tokens"),
        max_output_tokens=profile.get("max_output_tokens"),
        prompt_cache_enabled=profile.get("prompt_cache_enabled"),
        prompt_cache_key=profile.get("prompt_cache_key"),
        prompt_cache_retention=profile.get("prompt_cache_retention"),
    )

    compression_meta = {"skipped_reason": "compressor_unavailable"}
    if compress_request_for_stage is not None:
        try:
            compression_result = compress_request_for_stage(
                stage_key=str(profile.get("stage_key") or stage_key),
                req=req,
                profile=profile,
            )
            req = compression_result.request
            compression_meta = dict(compression_result.metadata or {})
            compression_meta["applied"] = bool(compression_result.applied)
            if compression_meta.get("warning") and os.getenv("CONTEXT_COMPRESSOR_WARN_STDERR", "1").strip().lower() not in {"0", "false", "no", "off"}:
                print(
                    f"[context_compressor warning] {compression_meta.get('warning')} "
                    f"stage={compression_meta.get('stage_key', stage_key)} "
                    f"reason={compression_meta.get('skipped_reason', 'unknown')}",
                    file=sys.stderr,
                )
        except Exception as exc:
            # Compression must never become a workflow blocker. It also should
            # not silently pass a giant expensive prompt through when the
            # compressor crashes. Use router-level deterministic fallback if
            # available; only then continue unchanged as a last resort.
            if emergency_deterministic_compress_request_for_stage is not None:
                try:
                    fallback_result = emergency_deterministic_compress_request_for_stage(
                        stage_key=str(profile.get("stage_key") or stage_key),
                        req=req,
                        profile=profile,
                        reason=str(exc),
                    )
                    req = fallback_result.request
                    compression_meta = dict(fallback_result.metadata or {})
                    compression_meta["applied"] = bool(fallback_result.applied)
                except Exception as fallback_exc:
                    compression_meta = {
                        "applied": False,
                        "error": f"compressor={exc}; router_fallback={fallback_exc}",
                        "warning": "Context compressor crashed and router fallback failed. Original input was sent unchanged.",
                    }
            else:
                compression_meta = {
                    "applied": False,
                    "error": str(exc),
                    "warning": "Context compressor crashed. Original input was sent unchanged.",
                }
            if compression_meta.get("warning") and os.getenv("CONTEXT_COMPRESSOR_WARN_STDERR", "1").strip().lower() not in {"0", "false", "no", "off"}:
                print(f"[context_compressor warning] {compression_meta['warning']} error={compression_meta.get('error')}", file=sys.stderr)

    trace_event(
        "context_compression_result",
        stage_key=profile.get("stage_key") or stage_key,
        provider=profile.get("provider"),
        profile_name=profile.get("profile_name"),
        applied=bool(compression_meta.get("applied")),
        skipped_reason=compression_meta.get("skipped_reason"),
        warning=compression_meta.get("warning"),
        original_estimated_tokens=compression_meta.get("original_estimated_tokens"),
        compressed_estimated_tokens=compression_meta.get("compressed_estimated_tokens"),
        target_tokens=compression_meta.get("target_tokens"),
        method=compression_meta.get("method"),
        ollama_model=compression_meta.get("ollama_model"),
        error=compression_meta.get("error") or compression_meta.get("ollama_error"),
    )

    attempted_profiles: Set[str] = {str(profile.get("profile_name"))}
    try:
        response = _dispatch_provider(req, profile)
        response = _normalize_or_fallback(
            response=response,
            req=req,
            profile=profile,
            attempted_profiles=attempted_profiles,
        )
    except Exception as exc:
        trace_event(
            "provider_call_error",
            stage_key=profile.get("stage_key") or stage_key,
            provider=profile.get("provider"),
            profile_name=profile.get("profile_name"),
            model=profile.get("model"),
            error_type=exc.__class__.__name__,
            error=str(exc),
        )
        raise

    response.usage = dict(response.usage or {})
    response.usage["context_compressor"] = compression_meta
    trace_event(
        "provider_call_end",
        stage_key=profile.get("stage_key") or stage_key,
        provider=response.provider,
        profile_name=response.profile_name,
        model=response.model,
        response_id=response.id,
        prompt_cache=(response.usage or {}).get("prompt_cache"),
        usage=response.usage,
    )
    return response
