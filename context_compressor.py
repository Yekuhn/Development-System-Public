"""
context_compressor.py

Cost-control context compressor for the Ascendant Path / VeRealm agent model.

Strategy:
- Do not compress intake or small/comfortably-budgeted prompts.
- For large expensive prompts, deterministically reduce first using free local
  Python logic/libraries, then ask cheap local Ollama to polish/compress.
- If Ollama fails, keep using the deterministic reduced context. Never fall
  back to sending the full original huge prompt just because Ollama timed out.

Optional free dependencies:
- tiktoken: better token counting.
- rapidfuzz: stronger near-duplicate removal.
The module works without them via stdlib fallbacks.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # Optional. Do not make the whole workflow depend on this package.
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore

try:  # Optional. Do not make the whole workflow depend on this package.
    from rapidfuzz import fuzz as rapidfuzz_fuzz  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    rapidfuzz_fuzz = None  # type: ignore


@dataclass
class CompressionResult:
    request: Dict[str, Any]
    applied: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


_TOKEN_ENCODER = None


def _get_encoder():
    global _TOKEN_ENCODER
    if _TOKEN_ENCODER is not None:
        return _TOKEN_ENCODER
    if tiktoken is None:
        return None
    enc_name = os.getenv("CONTEXT_COMPRESSOR_TIKTOKEN_ENCODING", "cl100k_base")
    try:
        _TOKEN_ENCODER = tiktoken.get_encoding(enc_name)
    except Exception:
        try:
            _TOKEN_ENCODER = tiktoken.encoding_for_model(os.getenv("CONTEXT_COMPRESSOR_TIKTOKEN_MODEL", "gpt-4o"))
        except Exception:
            _TOKEN_ENCODER = None
    return _TOKEN_ENCODER


def estimate_tokens(text: str) -> int:
    """Local token estimate. Uses tiktoken when present, len//4 fallback otherwise."""
    if not text:
        return 0
    enc = _get_encoder()
    if enc is not None:
        try:
            return max(1, len(enc.encode(text)))
        except Exception:
            pass
    return max(1, len(text) // 4)


def _input_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, Mapping):
                role = item.get("role", "user")
                content = item.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(
                        str(x.get("text", x)) if isinstance(x, Mapping) else str(x)
                        for x in content
                    )
                parts.append(f"[{role}]\n{content}")
            else:
                parts.append(str(item))
        return "\n\n".join(parts)
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _near_duplicate(a: str, b: str, *, threshold: int = 94) -> bool:
    a1 = _normalize_ws(a)
    b1 = _normalize_ws(b)
    if not a1 or not b1:
        return False
    if a1 == b1:
        return True
    # Cheap substring catch for repeated JSON/log lines.
    if len(a1) > 60 and (a1 in b1 or b1 in a1):
        return True
    if rapidfuzz_fuzz is not None:
        try:
            return int(rapidfuzz_fuzz.ratio(a1, b1)) >= threshold
        except Exception:
            return False
    return False


def _uniq_keep_order(items: Sequence[str], *, fuzzy: bool = False, max_compare: int = 80) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        raw = x.rstrip()
        key = _normalize_ws(raw)
        if not key or key in seen:
            continue
        if fuzzy:
            recent = out[-max_compare:]
            if any(_near_duplicate(raw, y) for y in recent):
                continue
        seen.add(key)
        out.append(raw)
    return out


def _clip_to_tokens(text: str, target_tokens: int) -> str:
    if estimate_tokens(text) <= target_tokens:
        return text
    enc = _get_encoder()
    if enc is not None:
        try:
            toks = enc.encode(text)
            return enc.decode(toks[: max(1, target_tokens)])
        except Exception:
            pass
    return text[: max(1, target_tokens * 4)]


def _json_compact(obj: Any, *, max_depth: int = 5, max_list: int = 16, max_str: int = 240) -> Any:
    """Deterministically compact JSON-like data while preserving key facts."""
    important_keys = {
        "run_id", "task_id", "active_task_id", "agent", "stage", "mode", "brief", "summary",
        "objective", "acceptance_criteria", "files_expected", "files", "paths", "dependencies",
        "error", "errors", "issue", "issues", "qa", "blocker", "block_reason", "block_type",
        "required_action", "question", "options", "decision", "status", "notes", "constraints",
        "resource", "resources", "artifact_paths", "pointers", "node_status", "current_node",
    }

    def rec(v: Any, depth: int) -> Any:
        if depth > max_depth:
            return "...[depth limit]"
        if isinstance(v, Mapping):
            keys = list(v.keys())
            ordered = [k for k in keys if str(k) in important_keys] + [k for k in keys if str(k) not in important_keys]
            out: Dict[str, Any] = {}
            for k in ordered[:60]:
                out[str(k)] = rec(v[k], depth + 1)
            if len(keys) > 60:
                out["__truncated_keys__"] = len(keys) - 60
            return out
        if isinstance(v, list):
            if len(v) <= max_list:
                return [rec(x, depth + 1) for x in v]
            head = [rec(x, depth + 1) for x in v[: max_list // 2]]
            tail = [rec(x, depth + 1) for x in v[-max(1, max_list // 2):]]
            return head + [f"...[{len(v) - len(head) - len(tail)} items omitted]..."] + tail
        if isinstance(v, str):
            if len(v) <= max_str or any(token in v.lower() for token in ("traceback", "error", "block", "t00", ".py", ".ts", ".tsx", ".gitkeep")):
                return v if len(v) <= max_str * 3 else (v[: max_str * 2] + " ... " + v[-max_str:])
            return v[:max_str] + f"...[{len(v) - max_str} chars omitted]"
        return v

    return rec(obj, 0)


class ContextCompressor:
    """Preprocesses intended AI input into a shorter temporary input."""

    IMPORTANT_KEYWORDS = (
        "task_id", "work_item", "summary", "brief", "user", "requirement",
        "constraint", "non-goal", "scope_in", "scope_out", "acceptance",
        "verification", "files_expected", "dependencies", "interface",
        "risk", "qa", "issue", "blocker", "error", "traceback",
        "required_action", "rerun", "retry", "directive", "human", "decision",
        "resource", "artifact", "path", "file", "directory", "staged_write",
        "write_report", "violation", "gitkeep", ".gitkeep", "placeholder",
        "current_node", "node_status", "blocked", "block_reason", "block_payload",
    )

    PATH_RE = re.compile(r"(?<![\w.-])(?:[A-Za-z0-9_.-]+/)+(?:[A-Za-z0-9_.-]+(?:\.[A-Za-z0-9_.-]+)?|\.gitkeep)/?")
    ID_RE = re.compile(r"\b(T\d+|eng_\d+|attempt_\d+|run_id|request_id|response_id|current_node|block_reason)\b")

    def __init__(self) -> None:
        self.enabled = os.getenv("CONTEXT_COMPRESSOR_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
        self.min_input_tokens = int(os.getenv("CONTEXT_COMPRESSOR_MIN_INPUT_TOKENS", "4000"))
        self.force_input_tokens = int(os.getenv("CONTEXT_COMPRESSOR_FORCE_INPUT_TOKENS", "12000"))
        self.default_target_tokens = int(os.getenv("CONTEXT_COMPRESSOR_TARGET_INPUT_TOKENS", "2600"))
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self.ollama_model = os.getenv("CONTEXT_COMPRESSOR_OLLAMA_MODEL", os.getenv("OLLAMA_SMALL_MODEL", "llama3.2:1b"))
        raw_timeout = float(os.getenv("CONTEXT_COMPRESSOR_TIMEOUT_SEC", "30"))
        timeout_cap = float(os.getenv("CONTEXT_COMPRESSOR_MAX_TIMEOUT_SEC", "30"))
        self.timeout_sec = max(5.0, min(raw_timeout, timeout_cap))
        self.use_ollama = os.getenv("CONTEXT_COMPRESSOR_USE_OLLAMA", "1").strip().lower() not in {"0", "false", "no", "off"}
        self.skip_stages = {
            item.strip()
            for item in os.getenv(
                "CONTEXT_COMPRESSOR_SKIP_STAGES",
                "intake.generate,intake.revise_after_user_feedback",
            ).split(",")
            if item.strip()
        }
        self.skip_stage_prefixes = tuple(
            item.strip()
            for item in os.getenv("CONTEXT_COMPRESSOR_SKIP_STAGE_PREFIXES", "").split(",")
            if item.strip()
        )
        self.provider_budget_margin = float(os.getenv("CONTEXT_COMPRESSOR_PROVIDER_BUDGET_MARGIN", "0.85"))
        # Keep local prompts small enough for llama3.2:1b. This is the fix for
        # previous engineer.fix_after_qa timeouts on 20k-40k token raw prompts.
        self.local_input_tokens = int(os.getenv("CONTEXT_COMPRESSOR_LOCAL_INPUT_TOKENS", "3200"))
        self.local_retry_input_tokens = int(os.getenv("CONTEXT_COMPRESSOR_LOCAL_RETRY_INPUT_TOKENS", "2200"))
        self.fallback_tokens = int(os.getenv("CONTEXT_COMPRESSOR_DETERMINISTIC_FALLBACK_TOKENS", "4200"))
        self.ollama_num_ctx = int(os.getenv("CONTEXT_COMPRESSOR_OLLAMA_NUM_CTX", "8192"))
        self.ollama_output_tokens = int(os.getenv("CONTEXT_COMPRESSOR_OLLAMA_OUTPUT_TOKENS", "900"))
        self.use_rapidfuzz = os.getenv("CONTEXT_COMPRESSOR_USE_RAPIDFUZZ", "1").strip().lower() not in {"0", "false", "no", "off"}

    def should_skip(self, *, stage_key: str, profile: Mapping[str, Any]) -> Tuple[bool, str]:
        if not self.enabled:
            return True, "disabled"
        if profile.get("provider") == "deterministic":
            return True, "deterministic_stage"
        if stage_key.startswith("context.") or stage_key.startswith("token."):
            return True, "context_or_token_stage"
        # QA is a gate, not a creative drafting stage. Compressing QA input can
        # remove staged_write_report/files_written/candidate_workspace_dir and
        # cause false blocks. Keep QA evidence lossless unless explicitly forced.
        if stage_key.startswith("qa.") and os.getenv("ASCENDANT_COMPRESS_QA", "0").strip().lower() not in {"1", "true", "yes", "on"}:
            return True, "qa_evidence_lossless"
        if stage_key in self.skip_stages:
            return True, "stage_excluded"
        if self.skip_stage_prefixes and any(stage_key.startswith(prefix) for prefix in self.skip_stage_prefixes):
            return True, "stage_prefix_excluded"
        return False, ""

    def target_tokens_for_profile(self, profile: Mapping[str, Any]) -> int:
        max_input = int(profile.get("max_input_tokens") or 0)
        if max_input <= 0:
            return self.default_target_tokens
        return max(1500, min(self.default_target_tokens, int(max_input * 0.45)))

    def compress_request(
        self,
        *,
        stage_key: str,
        req: Mapping[str, Any],
        profile: Mapping[str, Any],
    ) -> CompressionResult:
        new_req: Dict[str, Any] = deepcopy(dict(req))
        skip, reason = self.should_skip(stage_key=stage_key, profile=profile)
        if skip:
            return CompressionResult(new_req, False, {"skipped_reason": reason})

        original_input = new_req.get("input", "")
        input_text = _input_to_text(original_input)
        original_tokens = estimate_tokens(input_text)
        max_input_tokens = int(profile.get("max_input_tokens") or 0)

        if original_tokens < self.min_input_tokens:
            return CompressionResult(
                new_req,
                False,
                {
                    "skipped_reason": "below_threshold",
                    "original_estimated_tokens": original_tokens,
                    "threshold_tokens": self.min_input_tokens,
                    "token_counter": "tiktoken" if _get_encoder() is not None else "chars_div_4",
                },
            )

        if (
            max_input_tokens > 0
            and original_tokens <= int(max_input_tokens * self.provider_budget_margin)
            and original_tokens < self.force_input_tokens
        ):
            return CompressionResult(
                new_req,
                False,
                {
                    "skipped_reason": "within_provider_budget",
                    "original_estimated_tokens": original_tokens,
                    "max_input_tokens": max_input_tokens,
                    "provider_budget_margin": self.provider_budget_margin,
                    "force_input_tokens": self.force_input_tokens,
                    "token_counter": "tiktoken" if _get_encoder() is not None else "chars_div_4",
                },
            )

        target_tokens = self.target_tokens_for_profile(profile)
        started = time.time()
        ollama_error = ""

        local_input = self._deterministic_shorten(
            input_text,
            stage_key=stage_key,
            target_tokens=max(self.local_input_tokens, target_tokens + 600),
            mode="ollama_precompact",
        )
        local_input_tokens = estimate_tokens(local_input)

        retry_input = self._deterministic_shorten(
            input_text,
            stage_key=stage_key,
            target_tokens=self.local_retry_input_tokens,
            mode="ollama_retry_precompact",
        )

        compressed = ""
        if self.use_ollama:
            for candidate in (local_input, retry_input):
                try:
                    compressed = self._compress_with_ollama(
                        text=candidate,
                        stage_key=stage_key,
                        target_tokens=target_tokens,
                    ).strip()
                    if compressed:
                        break
                except Exception as exc:
                    ollama_error = str(exc)
                    compressed = ""
        else:
            ollama_error = "ollama_disabled"

        if compressed:
            compressed_tokens = estimate_tokens(compressed)
            if compressed_tokens < int(original_tokens * 0.95):
                return self._result_with_input(
                    new_req,
                    stage_key=stage_key,
                    compressed=compressed,
                    method="ollama_after_deterministic_precompact",
                    original_tokens=original_tokens,
                    target_tokens=target_tokens,
                    started=started,
                    extra={
                        "ollama_model": self.ollama_model,
                        "local_precompact_estimated_tokens": local_input_tokens,
                        "token_counter": "tiktoken" if _get_encoder() is not None else "chars_div_4",
                    },
                )

        fallback = self._deterministic_shorten(
            input_text,
            stage_key=stage_key,
            target_tokens=max(1500, min(self.fallback_tokens, target_tokens + 1600)),
            mode="deterministic_fallback_after_ollama_failure",
        )
        fallback_tokens = estimate_tokens(fallback)
        if fallback and fallback_tokens < int(original_tokens * 0.95):
            return self._result_with_input(
                new_req,
                stage_key=stage_key,
                compressed=fallback,
                method="deterministic_fallback_after_ollama_failed",
                original_tokens=original_tokens,
                target_tokens=target_tokens,
                started=started,
                extra={
                    "ollama_model": self.ollama_model if self.use_ollama else None,
                    "ollama_error": ollama_error or "empty compression result",
                    "local_precompact_estimated_tokens": local_input_tokens,
                    "fallback_estimated_tokens": fallback_tokens,
                    "notice": "Ollama compression failed, so deterministic compression was applied instead of sending the full original input.",
                    "token_counter": "tiktoken" if _get_encoder() is not None else "chars_div_4",
                },
            )

        # Last-resort safety: force a clipped deterministic context rather than
        # allowing a huge full prompt to leak through silently.
        emergency = _clip_to_tokens(self._emergency_pack(input_text, stage_key=stage_key), max(1200, target_tokens))
        if emergency:
            return self._result_with_input(
                new_req,
                stage_key=stage_key,
                compressed=emergency,
                method="emergency_deterministic_clip",
                original_tokens=original_tokens,
                target_tokens=target_tokens,
                started=started,
                extra={
                    "ollama_model": self.ollama_model if self.use_ollama else None,
                    "ollama_error": ollama_error or "empty compression result",
                    "notice": "Emergency deterministic clipping was applied; the full original input was not sent.",
                },
            )

        return CompressionResult(
            new_req,
            False,
            {
                "stage_key": stage_key,
                "skipped_reason": "compression_failed_no_safe_fallback",
                "warning": "Context compression failed and no deterministic fallback could be built. Original input was sent unchanged.",
                "original_estimated_tokens": original_tokens,
                "target_tokens": target_tokens,
                "duration_sec": round(time.time() - started, 3),
                "ollama_model": self.ollama_model,
                "ollama_error": ollama_error or "empty compression result",
            },
        )

    def _result_with_input(
        self,
        req: Dict[str, Any],
        *,
        stage_key: str,
        compressed: str,
        method: str,
        original_tokens: int,
        target_tokens: int,
        started: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> CompressionResult:
        compressed_tokens = estimate_tokens(compressed)
        temporary_input = (
            "[TEMPORARY COMPRESSED INPUT]\n"
            f"Target stage: {stage_key}\n"
            f"Compression method: {method}\n"
            f"Original estimated input tokens: {original_tokens}\n"
            f"Compressed estimated input tokens: {compressed_tokens}\n"
            "Instruction: Use this compressed input as the working context for the requested task. "
            "It is a cost-control summary and does not replace original artifacts on disk. "
            "Preserve task IDs, file paths, scope, QA blockers, and acceptance criteria exactly. "
            "When a needed detail is absent from this compressed context, use available artifacts/files on disk rather than inventing.\n\n"
            "--- COMPRESSED CONTEXT ---\n"
            f"{compressed}"
        )
        new_req = deepcopy(req)
        new_req["input"] = [{"role": "user", "content": temporary_input}]
        meta = {
            "stage_key": stage_key,
            "method": method,
            "original_estimated_tokens": original_tokens,
            "compressed_estimated_tokens": estimate_tokens(temporary_input),
            "target_tokens": target_tokens,
            "duration_sec": round(time.time() - started, 3),
        }
        if extra:
            meta.update(extra)
        return CompressionResult(new_req, True, meta)

    def _score_line(self, line: str, idx: int, total_lines: int, *, stage_key: str) -> int:
        low = line.lower()
        score = 0
        for kw in self.IMPORTANT_KEYWORDS:
            if kw in low:
                score += 4
        if self.ID_RE.search(line):
            score += 6
        if self.PATH_RE.search(line):
            score += 5
        if re.search(r"\.(py|ts|tsx|js|jsx|json|md|txt|css|html|yml|yaml|gitkeep)\b", line):
            score += 3
        if any(x in low for x in ("traceback", "runtimeerror", "failed", "missing", "invalid", "blocked", "violation")):
            score += 7
        if any(x in low for x in ("acceptance", "verify", "verification", "must", "required", "expected")):
            score += 4
        if stage_key.startswith("engineer") and any(x in low for x in ("files_expected", "staged_write", "write_report", "qa", "fix_after_qa")):
            score += 6
        # Recency matters: last 25% of the prompt often has newest QA/failure.
        if idx > total_lines * 0.75:
            score += 2
        if len(line) > 900:
            score -= 2
        return score

    def _extract_json_blocks(self, text: str, *, target_chars: int) -> str:
        snippets: List[str] = []
        # Conservative extraction of balanced-ish JSON objects by line ranges is hard;
        # instead capture highly relevant JSON lines and compact parseable whole input.
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                obj = json.loads(stripped)
                compact = json.dumps(_json_compact(obj), ensure_ascii=False, indent=2)
                return compact[:target_chars]
            except Exception:
                pass
        for line in text.splitlines():
            low = line.lower()
            if any(k in low for k in ("\"task_id\"", "\"files_expected\"", "\"block_reason\"", "\"required_action\"", "\"error\"", "\"status\"", "\"question\"")):
                snippets.append(line)
                if len("\n".join(snippets)) >= target_chars:
                    break
        return "\n".join(_uniq_keep_order(snippets, fuzzy=False))[:target_chars]

    def _deterministic_shorten(self, text: str, *, stage_key: str, target_tokens: int, mode: str) -> str:
        target_tokens = max(800, int(target_tokens))
        target_chars = max(2500, target_tokens * 4)
        if estimate_tokens(text) <= target_tokens:
            return text

        lines = text.splitlines()
        total_lines = len(lines)
        scored: List[Tuple[int, int, str]] = []
        for i, line in enumerate(lines):
            score = self._score_line(line, i, total_lines, stage_key=stage_key)
            if score > 0:
                scored.append((score, i, line))

        head_chars = int(target_chars * 0.16)
        tail_chars = int(target_chars * 0.24)
        json_chars = int(target_chars * 0.18)
        path_chars = int(target_chars * 0.10)
        mid_chars = max(1000, target_chars - head_chars - tail_chars - json_chars - path_chars - 1800)

        head = text[:head_chars]
        tail = text[-tail_chars:] if tail_chars > 0 else ""

        json_snips = self._extract_json_blocks(text, target_chars=json_chars)

        paths = _uniq_keep_order(self.PATH_RE.findall(text), fuzzy=False)
        path_snips = "\n".join(paths)[:path_chars]

        scored.sort(key=lambda x: (-x[0], x[1]))
        snippet_lines: List[str] = []
        used = set()
        for _score, idx, _line in scored:
            for j in range(max(0, idx - 2), min(len(lines), idx + 3)):
                if j not in used:
                    used.add(j)
                    snippet_lines.append(lines[j])
            if len("\n".join(snippet_lines)) >= mid_chars:
                break

        snippets = "\n".join(_uniq_keep_order(snippet_lines, fuzzy=bool(self.use_rapidfuzz and rapidfuzz_fuzz is not None)))[:mid_chars]

        packed = (
            "[DETERMINISTIC CONTEXT COMPACTION]\n"
            f"stage={stage_key}\nmode={mode}\n"
            f"token_counter={'tiktoken' if _get_encoder() is not None else 'chars_div_4'}\n"
            f"rapidfuzz={'available' if rapidfuzz_fuzz is not None else 'unavailable'}\n"
            "Policy: keep current task, acceptance criteria, files, QA blockers, human decisions, exact errors, and recent context. Drop duplicated/stale logs.\n\n"
            "--- BEGINNING / CONTRACT ---\n"
            f"{head}\n\n"
            "--- COMPACT JSON / KEY FIELDS ---\n"
            f"{json_snips}\n\n"
            "--- FILE PATHS / DIRECTORIES MENTIONED ---\n"
            f"{path_snips}\n\n"
            "--- HIGH-SIGNAL TASK / QA / ERROR LINES ---\n"
            f"{snippets}\n\n"
            "--- RECENT TAIL ---\n"
            f"{tail}"
        )
        return _clip_to_tokens(packed, target_tokens)

    def _emergency_pack(self, text: str, *, stage_key: str) -> str:
        lines = text.splitlines()
        relevant = []
        for i, line in enumerate(lines):
            if self._score_line(line, i, len(lines), stage_key=stage_key) >= 5:
                relevant.append(line)
        relevant = _uniq_keep_order(relevant, fuzzy=False)
        return (
            "[EMERGENCY DETERMINISTIC CONTEXT]\n"
            f"stage={stage_key}\n"
            "The normal compressor failed. This emergency pack preserves only high-signal lines and the recent tail.\n\n"
            "--- HIGH SIGNAL ---\n"
            + "\n".join(relevant[:400])
            + "\n\n--- TAIL ---\n"
            + text[-5000:]
        )

    def _compress_with_ollama(self, *, text: str, stage_key: str, target_tokens: int) -> str:
        # Keep llama3.2:1b cheap and fast. Asking it for 2k+ tokens caused
        # repeated timeouts in engineer.fix_after_qa. The deterministic fallback
        # remains the safety net if this shorter local pass still fails.
        max_output_tokens = max(256, min(self.ollama_output_tokens, target_tokens))
        prompt = (
            "You are a cheap local context compressor for an agentic software workflow.\n"
            "Your only job is to shorten and reorganize the input for the next AI agent.\n"
            "Do not ask questions. Do not route. Do not decide whether the workflow is blocked.\n"
            "Preserve hard facts, user requirements, constraints, non-goals, file paths, task IDs, acceptance criteria, resource decisions, exact errors, and verification requirements.\n"
            "For engineering stages, preserve files_expected, actual file paths, QA issues, required_action, retry directives, and staged-write evidence.\n"
            "Remove repetition, stale commentary, redundant logs, and duplicated instructions.\n"
            "Return plain text only.\n\n"
            f"Target stage: {stage_key}\n"
            f"Approximate target length: under {target_tokens} tokens.\n\n"
            "INPUT TO COMPRESS:\n"
            f"{text}"
        )
        # Clip prompt to fit the small local model. Do not let optional tiktoken
        # overshoot cause another timeout.
        prompt_budget = max(1200, min(self.local_input_tokens + 900, self.ollama_num_ctx - max_output_tokens - 512))
        prompt = _clip_to_tokens(prompt, prompt_budget)
        input_est = estimate_tokens(prompt)
        num_ctx = max(4096, min(32768, max(self.ollama_num_ctx, input_est + max_output_tokens + 512)))
        payload: Dict[str, Any] = {
            "model": self.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": max_output_tokens,
                "temperature": 0.0,
                "num_ctx": num_ctx,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.ollama_host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Ollama compression failed at {self.ollama_host}: {exc}") from exc
        return ((raw.get("message") or {}).get("content") or "").strip()


_DEFAULT_COMPRESSOR: Optional[ContextCompressor] = None


def get_default_compressor() -> ContextCompressor:
    global _DEFAULT_COMPRESSOR
    if _DEFAULT_COMPRESSOR is None:
        _DEFAULT_COMPRESSOR = ContextCompressor()
    return _DEFAULT_COMPRESSOR


def compress_request_for_stage(
    *,
    stage_key: str,
    req: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> CompressionResult:
    return get_default_compressor().compress_request(stage_key=stage_key, req=req, profile=profile)


def emergency_deterministic_compress_request_for_stage(
    *,
    stage_key: str,
    req: Mapping[str, Any],
    profile: Mapping[str, Any],
    reason: str = "compressor_exception",
) -> CompressionResult:
    """Router-level safety net if ContextCompressor itself throws."""
    compressor = ContextCompressor()
    new_req: Dict[str, Any] = deepcopy(dict(req))
    original_input = new_req.get("input", "")
    text = _input_to_text(original_input)
    original_tokens = estimate_tokens(text)
    target = compressor.target_tokens_for_profile(profile)
    started = time.time()
    compacted = compressor._deterministic_shorten(
        text,
        stage_key=stage_key,
        target_tokens=max(1500, min(compressor.fallback_tokens, target + 1600)),
        mode=f"emergency_after_{reason}",
    )
    if compacted and estimate_tokens(compacted) < int(original_tokens * 0.98):
        return compressor._result_with_input(
            new_req,
            stage_key=stage_key,
            compressed=compacted,
            method="router_emergency_deterministic_fallback",
            original_tokens=original_tokens,
            target_tokens=target,
            started=started,
            extra={"error": reason, "notice": "ContextCompressor raised, so router applied deterministic fallback."},
        )
    return CompressionResult(new_req, False, {
        "applied": False,
        "error": reason,
        "warning": "Context compressor crashed and router fallback could not reduce input. Original input was sent unchanged.",
        "original_estimated_tokens": original_tokens,
    })
