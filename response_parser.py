"""
response_parser.py

Shared helpers for parsing model responses that should contain JSON.

This is intentionally tolerant because browser/GPTWeb providers may return:
- a request-id guard line such as [GPTWEB_REQ_abcd1234]
- JSON inside a code fence
- a small amount of wrapper/prose around an otherwise-valid JSON object

Provider-level request-id validation still happens in model_provider_router.py.
This module only cleans already-returned text before agent-level json.loads.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional


_GPTWEB_GUARD_RE = re.compile(r"\[GPTWEB_REQ_[A-Za-z0-9_-]+\]")


def strip_gptweb_guard(text: str) -> str:
    """Remove GPTWeb request-id guard markers from returned text.

    If the marker appears more than once, keep content after the last marker.
    That handles overly-broad browser selectors that accidentally include the
    user prompt plus the assistant response in the same extracted text.
    """
    value = str(text or "").strip()
    matches = list(_GPTWEB_GUARD_RE.finditer(value))
    if not matches:
        return value
    return value[matches[-1].end():].lstrip("\r\n \t")


def strip_code_fence(text: str) -> str:
    value = str(text or "").strip()
    if not value.startswith("```"):
        return value

    # Handle fenced JSON or plain fenced content.
    lines = value.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```"):
        if lines[-1].strip().startswith("```"):
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        if lines and lines[0].strip().lower() == "json":
            lines = lines[1:]
        return "\n".join(lines).strip()

    return value.strip("`").strip()


def escape_control_chars_inside_json_strings(text: str) -> str:
    """Escape raw control characters that appear inside JSON strings.

    Browser/plain-text model output sometimes contains literal newlines inside
    string values instead of JSON-escaped \n. Standard json.loads rejects that.
    This repair preserves structure while converting those raw characters to
    JSON escapes only when the scanner is inside a quoted string.
    """
    out = []
    in_string = False
    escape = False
    for ch in str(text or ""):
        if in_string:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            if ord(ch) < 32:
                out.append(f"\\u{ord(ch):04x}")
                continue
            out.append(ch)
            continue

        out.append(ch)
        if ch == '"':
            in_string = True
            escape = False

    return "".join(out)


def loads_json_lenient(text: str) -> Any:
    """json.loads with one safe repair pass for raw control chars in strings."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = escape_control_chars_inside_json_strings(text)
        return json.loads(repaired)


def extract_json_text(raw: Any) -> Optional[str]:
    """Return canonical JSON text if a JSON object/array can be found.

    Recovery must preserve the outermost JSON structure. A common model error is
    prose followed by ``[{...}, {...}]``. The previous object-first scan could
    grab only the first ``{...}``, silently discarding later list objects. This
    scanner now evaluates both object and array candidates and returns the
    earliest valid top-level candidate, preferring the longer candidate when two
    candidates start at the same character.
    """
    text = strip_code_fence(strip_gptweb_guard(str(raw or ""))).strip()
    if not text:
        return None

    try:
        parsed = loads_json_lenient(text)
        return json.dumps(parsed, ensure_ascii=False)
    except Exception:
        pass

    candidates = []
    for opening, closing in (("{", "}"), ("[", "]")):
        start = text.find(opening)
        while start >= 0:
            depth = 0
            in_string = False
            escape = False
            for idx in range(start, len(text)):
                ch = text[idx]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == opening:
                    depth += 1
                elif ch == closing:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:idx + 1]
                        try:
                            parsed = loads_json_lenient(candidate)
                            candidates.append((start, -(idx + 1 - start), parsed))
                        except Exception:
                            pass
                        break
            start = text.find(opening, start + 1)

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    return json.dumps(candidates[0][2], ensure_ascii=False)


def parse_json_response_text(raw: Any) -> Any:
    """Parse a model response that should contain JSON.

    Raises ValueError with a short preview when no JSON can be recovered.
    """
    candidate = extract_json_text(raw)
    if candidate is None:
        preview = str(raw or "").strip().replace("\n", " ")[:500]
        raise ValueError(f"No valid JSON object/array found in model response. Preview: {preview}")
    return loads_json_lenient(candidate)


def _coerce_list_to_dict(parsed: list) -> Optional[dict]:
    """Recover common model mistakes where an object is wrapped as a list.

    Some providers/models occasionally return one of these even when a strict
    object schema was requested:
    - [{...}]
    - [{"key_a": ...}, {"key_b": ...}]

    The second form is only safe to recover when each item is a dict and keys
    do not conflict. Conflicting values remain an error so malformed multi-item
    payloads do not get silently accepted.
    """
    if len(parsed) == 1 and isinstance(parsed[0], dict):
        return dict(parsed[0])

    if parsed and all(isinstance(item, dict) for item in parsed):
        merged: dict = {}
        for item in parsed:
            for key, value in item.items():
                if key in merged and merged[key] != value:
                    return None
                merged[key] = value
        if merged:
            return merged

    return None


def parse_json_response_dict(raw: Any) -> dict:
    parsed = parse_json_response_text(raw)
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        recovered = _coerce_list_to_dict(parsed)
        if recovered is not None:
            return recovered
    preview = str(raw or "").strip().replace("\n", " ")[:500]
    raise ValueError(
        f"Expected JSON object from model response, got {type(parsed).__name__}. "
        f"Preview: {preview}"
    )
