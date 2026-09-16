"""
preflight.py

Early deterministic environment checks for the Ascendant Path workflow. Preflight
should fail fast on missing provider/runtime prerequisites instead of letting the
pipeline reach an agent call and crash late.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _tcp_connectable(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except Exception:
        return False


def _http_get_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 500
    except Exception:
        return False


def _provider_needs_openai(profile: Dict[str, Any]) -> bool:
    return str(profile.get("provider", "")).lower() == "openai"


def _provider_needs_gptweb(profile: Dict[str, Any]) -> bool:
    return str(profile.get("provider", "")).lower() in {"gptweb", "browser", "chatgpt_web"}


def _provider_needs_ollama(profile: Dict[str, Any]) -> bool:
    return str(profile.get("provider", "")).lower() == "ollama"


def run_preflight(*, output_dir: str | Path = ".", strict: bool = False) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, severity: str, message: str, details: Dict[str, Any] | None = None) -> None:
        checks.append({"name": name, "ok": bool(ok), "severity": severity, "message": message, "details": details or {}})

    # Python module checks. OpenAI is only an error if an OpenAI-backed stage is selected.
    for mod in ["openai", "pydantic"]:
        present = importlib.util.find_spec(mod) is not None
        add(f"python_module:{mod}", present, "info" if mod == "openai" else "warning", f"Python module {mod} {'is available' if present else 'is not installed'}.")

    # Config resolution checks.
    stage_profiles: Dict[str, Any] = {}
    try:
        from provider_config import STAGE_REGISTRY, get_stage_profile  # type: ignore
        for stage_key in sorted(STAGE_REGISTRY.keys()):
            try:
                stage_profiles[stage_key] = get_stage_profile(stage_key)
            except Exception as exc:
                add(f"provider_stage:{stage_key}", False, "error", f"Stage profile does not resolve: {exc}")
        add("provider_config_import", True, "error", "provider_config imported successfully.")
    except Exception as exc:
        add("provider_config_import", False, "error", f"provider_config import failed: {exc}")

    providers = {str(p.get("provider", "")).lower() for p in stage_profiles.values() if isinstance(p, dict)}
    if "openai" in providers:
        key_present = bool(os.getenv("OPENAI_API_KEY", "").strip()) or Path("openai_api_key").exists()
        add("OPENAI_API_KEY", key_present, "error", "OpenAI key present for OpenAI-backed stages." if key_present else "OpenAI-backed stages selected but OPENAI_API_KEY/openai_api_key is missing.")
    if "ollama" in providers or os.getenv("CONTEXT_COMPRESSOR_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}:
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        add("ollama_reachable", _http_get_ok(host + "/api/tags"), "warning", f"Ollama endpoint checked at {host}.", {"url": host + "/api/tags"})
    if providers.intersection({"gptweb", "browser", "chatgpt_web"}):
        gptweb_url = os.getenv("GPTWEB_URL", "").strip()
        add("GPTWEB_URL", bool(gptweb_url), "error", "GPTWEB_URL present for GPTWeb-backed stages." if gptweb_url else "GPTWeb-backed stages selected but GPTWEB_URL is missing.")
        cdp_port = int(os.getenv("CHROME_CDP_PORT", os.getenv("GPTWEB_CDP_PORT", "9222")))
        add("chrome_cdp_reachable", _tcp_connectable("127.0.0.1", cdp_port), "warning", f"Chrome CDP checked on 127.0.0.1:{cdp_port}.")

    initial_path = Path("initial_input.json")
    if initial_path.exists():
        try:
            obj = json.loads(initial_path.read_text(encoding="utf-8"))
            add("initial_input_json", isinstance(obj, dict), "warning", "initial_input.json is valid JSON dict." if isinstance(obj, dict) else "initial_input.json exists but is not a JSON object.")
        except Exception as exc:
            add("initial_input_json", False, "warning", f"initial_input.json is invalid JSON: {exc}")
    else:
        add("initial_input_json", True, "info", "initial_input.json not present; supervised intake can create it.")

    out = Path(output_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".preflight_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)  # type: ignore[arg-type]
        add("output_dir_writable", True, "error", f"Output directory is writable: {out}")
    except Exception as exc:
        add("output_dir_writable", False, "error", f"Output directory is not writable: {exc}")

    errors = [c for c in checks if c["severity"] == "error" and not c["ok"]]
    warnings = [c for c in checks if c["severity"] == "warning" and not c["ok"]]
    ok = not errors and (not strict or not warnings)
    report = {
        "schema_version": "preflight.v1",
        "ok": ok,
        "strict": bool(strict),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "checks": checks,
    }
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "preflight_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Ascendant Path preflight checks.")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    result = run_preflight(output_dir=args.output_dir, strict=args.strict)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.get("ok") else 1)
