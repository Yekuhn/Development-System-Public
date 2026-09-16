"""
project_executor.py

Deterministic execution gate for generated projects.

The executor runs local, bounded checks against outputs/<run_id>/workspace before
final handoff. It is intentionally conservative: it records exact command logs,
never claims success without evidence, and supports optional custom commands.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CommandSpec:
    name: str
    cmd: List[str]
    cwd: Path
    required: bool = True
    timeout_sec: int = 120


@dataclass
class CommandResult:
    name: str
    cmd: List[str]
    cwd: str
    required: bool
    skipped: bool = False
    skip_reason: str = ""
    returncode: Optional[int] = None
    duration_sec: float = 0.0
    stdout: str = ""
    stderr: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cmd": self.cmd,
            "cwd": self.cwd,
            "required": self.required,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "returncode": self.returncode,
            "duration_sec": round(self.duration_sec, 3),
            "stdout": self.stdout[-20000:],
            "stderr": self.stderr[-20000:],
            "passed": (self.skipped and not self.required) or (self.returncode == 0),
        }


def _which(name: str) -> bool:
    return shutil.which(name) is not None


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            obj = json.loads(path.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else None
    except Exception:
        return None
    return None


def _run_command(spec: CommandSpec) -> CommandResult:
    started = time.time()
    try:
        proc = subprocess.run(
            spec.cmd,
            cwd=str(spec.cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=spec.timeout_sec,
            check=False,
        )
        return CommandResult(
            name=spec.name,
            cmd=spec.cmd,
            cwd=str(spec.cwd),
            required=spec.required,
            returncode=proc.returncode,
            duration_sec=time.time() - started,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            name=spec.name,
            cmd=spec.cmd,
            cwd=str(spec.cwd),
            required=spec.required,
            returncode=124,
            duration_sec=time.time() - started,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=((exc.stderr or "") if isinstance(exc.stderr, str) else "") + f"\nTIMEOUT after {spec.timeout_sec}s",
        )
    except Exception as exc:
        return CommandResult(
            name=spec.name,
            cmd=spec.cmd,
            cwd=str(spec.cwd),
            required=spec.required,
            returncode=125,
            duration_sec=time.time() - started,
            stderr=f"EXECUTOR_ERROR: {exc}",
        )


def _discover_python_checks(workspace: Path) -> List[CommandSpec]:
    py_files = [p for p in workspace.rglob("*.py") if "__pycache__" not in p.parts]
    if not py_files:
        return []
    specs = [CommandSpec(name="python_compileall", cmd=[sys.executable, "-m", "compileall", "-q", "."], cwd=workspace, required=True, timeout_sec=120)]
    tests = [p for p in workspace.rglob("test_*.py") if "__pycache__" not in p.parts] + [p for p in workspace.rglob("*_test.py") if "__pycache__" not in p.parts]
    if tests:
        if _which("pytest"):
            specs.append(CommandSpec(name="pytest", cmd=[sys.executable, "-m", "pytest", "-q"], cwd=workspace, required=True, timeout_sec=240))
        else:
            specs.append(CommandSpec(name="pytest", cmd=[sys.executable, "-m", "pytest", "-q"], cwd=workspace, required=False, timeout_sec=1))
    return specs


def _discover_node_checks(workspace: Path) -> List[CommandSpec]:
    package_json = workspace / "package.json"
    if not package_json.exists():
        return []
    specs: List[CommandSpec] = []
    npm_available = _which("npm")
    if not npm_available:
        return [CommandSpec(name="npm_available", cmd=["npm", "--version"], cwd=workspace, required=True, timeout_sec=1)]
    allow_install = os.getenv("ASCENDANT_EXECUTOR_ALLOW_INSTALL", "0").strip().lower() in {"1", "true", "yes", "on"}
    if allow_install and not (workspace / "node_modules").exists():
        install_cmd = ["npm", "ci"] if (workspace / "package-lock.json").exists() else ["npm", "install"]
        specs.append(CommandSpec(name="npm_install", cmd=install_cmd, cwd=workspace, required=True, timeout_sec=int(os.getenv("ASCENDANT_EXECUTOR_INSTALL_TIMEOUT_SEC", "600"))))
    pkg = _load_json(package_json) or {}
    scripts = pkg.get("scripts") if isinstance(pkg.get("scripts"), dict) else {}
    if "build" in scripts:
        if (workspace / "node_modules").exists() or allow_install:
            specs.append(CommandSpec(name="npm_run_build", cmd=["npm", "run", "build"], cwd=workspace, required=True, timeout_sec=300))
        else:
            specs.append(CommandSpec(name="npm_run_build", cmd=["npm", "run", "build"], cwd=workspace, required=False, timeout_sec=1))
    if "test" in scripts and os.getenv("ASCENDANT_EXECUTOR_RUN_NPM_TEST", "0").strip().lower() in {"1", "true", "yes", "on"}:
        specs.append(CommandSpec(name="npm_test", cmd=["npm", "test", "--", "--runInBand"], cwd=workspace, required=True, timeout_sec=300))
    return specs


def _load_custom_commands(workspace: Path) -> List[CommandSpec]:
    raw = os.getenv("ASCENDANT_EXECUTOR_COMMANDS", "").strip()
    config_path = workspace / ".ascendant_executor.json"
    obj: Any = None
    if raw:
        try:
            obj = json.loads(raw)
        except Exception:
            obj = None
    elif config_path.exists():
        try:
            obj = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            obj = None
    if isinstance(obj, dict):
        commands = obj.get("commands")
    else:
        commands = obj
    if not isinstance(commands, list):
        return []
    specs: List[CommandSpec] = []
    for idx, item in enumerate(commands):
        if isinstance(item, str):
            specs.append(CommandSpec(name=f"custom_{idx+1}", cmd=item.split(), cwd=workspace, required=True, timeout_sec=300))
        elif isinstance(item, dict):
            cmd = item.get("cmd")
            if isinstance(cmd, str):
                cmd = cmd.split()
            if not isinstance(cmd, list) or not cmd:
                continue
            cwd = workspace / str(item.get("cwd", "."))
            specs.append(CommandSpec(
                name=str(item.get("name") or f"custom_{idx+1}"),
                cmd=[str(x) for x in cmd],
                cwd=cwd,
                required=bool(item.get("required", True)),
                timeout_sec=int(item.get("timeout_sec", 300)),
            ))
    return specs


def _skip_result(spec: CommandSpec, reason: str) -> CommandResult:
    return CommandResult(name=spec.name, cmd=spec.cmd, cwd=str(spec.cwd), required=spec.required, skipped=True, skip_reason=reason)


def run_project_executor(*, workspace_dir: str | Path, output_dir: str | Path) -> Dict[str, Any]:
    workspace = Path(workspace_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if os.getenv("ASCENDANT_EXECUTOR_ENABLED", "1").strip().lower() in {"0", "false", "no", "off"}:
        report = {
            "schema_version": "project_executor.v1",
            "status": "skipped",
            "ok": True,
            "reason": "ASCENDANT_EXECUTOR_ENABLED=0",
            "workspace_dir": str(workspace),
            "commands": [],
        }
        (output / "executor_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report

    commands: List[CommandSpec] = []
    commands.extend(_discover_python_checks(workspace))
    commands.extend(_discover_node_checks(workspace))
    commands.extend(_load_custom_commands(workspace))

    results: List[CommandResult] = []
    for spec in commands:
        if not spec.cwd.exists():
            results.append(_skip_result(spec, "cwd_does_not_exist"))
            continue
        # Special optional skip cases produce a clear warning, not fake pass.
        if spec.name == "pytest" and not _which("pytest"):
            results.append(_skip_result(spec, "pytest_not_available"))
            continue
        if spec.name == "npm_run_build" and not (workspace / "node_modules").exists() and os.getenv("ASCENDANT_EXECUTOR_ALLOW_INSTALL", "0").strip().lower() not in {"1", "true", "yes", "on"}:
            results.append(_skip_result(spec, "node_modules_missing_and_install_disabled"))
            continue
        results.append(_run_command(spec))

    result_dicts = [r.as_dict() for r in results]
    required_failures = [r for r in result_dicts if r.get("required") and not r.get("passed")]
    required_skips = [r for r in result_dicts if r.get("required") and r.get("skipped")]
    optional_skips = [r for r in result_dicts if (not r.get("required")) and r.get("skipped")]

    if not workspace.exists():
        status = "failed"
        ok = False
        summary = "workspace_dir does not exist."
    elif required_failures:
        status = "failed"
        ok = False
        summary = f"{len(required_failures)} required executor command(s) failed."
    elif not result_dicts:
        status = "warning"
        ok = True
        summary = "No deterministic project checks were discovered. Add .ascendant_executor.json for stronger verification."
    elif required_skips or optional_skips:
        status = "warning"
        ok = True
        summary = "Executor completed with skipped checks; review report before relying on final output."
    else:
        status = "passed"
        ok = True
        summary = "All discovered deterministic executor checks passed."

    report = {
        "schema_version": "project_executor.v1",
        "status": status,
        "ok": ok,
        "summary": summary,
        "workspace_dir": str(workspace),
        "output_dir": str(output),
        "duration_sec": round(time.time() - started, 3),
        "commands": result_dicts,
        "counts": {
            "commands": len(result_dicts),
            "required_failures": len(required_failures),
            "required_skips": len(required_skips),
            "optional_skips": len(optional_skips),
        },
        "policy": {
            "install_allowed": os.getenv("ASCENDANT_EXECUTOR_ALLOW_INSTALL", "0").strip().lower() in {"1", "true", "yes", "on"},
            "npm_test_enabled": os.getenv("ASCENDANT_EXECUTOR_RUN_NPM_TEST", "0").strip().lower() in {"1", "true", "yes", "on"},
        },
    }
    report_path = output / "executor_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_lines = [
        "# Project Executor Report",
        "",
        f"Status: **{status}**",
        "",
        summary,
        "",
        "## Commands",
    ]
    for r in result_dicts:
        md_lines.extend([
            "",
            f"### {r['name']}",
            f"- Required: {r['required']}",
            f"- Skipped: {r['skipped']} {('(' + r['skip_reason'] + ')') if r.get('skip_reason') else ''}",
            f"- Return code: {r['returncode']}",
            f"- Command: `{' '.join(r['cmd'])}`",
        ])
    (output / "EXECUTOR_REPORT.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run deterministic checks against a generated workspace.")
    parser.add_argument("workspace_dir", nargs="?", default="workspace")
    parser.add_argument("--output-dir", default="executor")
    args = parser.parse_args()
    result = run_project_executor(workspace_dir=args.workspace_dir, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.get("ok") else 1)
