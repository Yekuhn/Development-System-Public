"""
provider_config.py

Ascendant Path / VeRealm agent-model provider routing configuration.

This file declares which AI-reply method each AI-using workflow stage should use.
It is configuration only: it should not call OpenAI, Ollama, GPTWeb, Playwright,
or any other provider directly.

Provider options:
- openai: OpenAI API through OpenAI Responses API.
- ollama: local Ollama server, usually http://localhost:11434.
- browser: GPTWeb/browser automation against a private AI website.
- deterministic: no LLM call; use Python logic only.

Only two run modes are intentionally supported:

1. power
   - All substantive AI agents use OpenAI API.
   - Deterministic stages remain deterministic.
   - Context compression/indexing/log compression use local Ollama in both modes.

2. development
   - Implementation agents use OpenAI API:
       UX / Designer, Engineering Lead, Engineer, QA.
   - Operational implementation documentation also uses OpenAI:
       Engineering Lead runbook.
   - High-level, administrative, and management agents use GPTWeb/browser:
       Intake, PM, Coordinator handoff/gates, docs, non-critical summaries.
   - Coordinator workflow-decision mode uses OpenAI because it may be invoked
     from parallel worker-loop exceptions.
   - Context compression/indexing/log compression use local Ollama.

Concurrency rule:
GPTWeb/browser is a single-threaded, lock-protected, text-first provider. It must
not be used by any stage that may run concurrently, especially Engineer and QA
worker-loop stages.

Design rule:
Agents request a stage. The Provider Router reads this config and selects the
backend. Individual agents should not hardcode OpenAI/Ollama/GPTWeb decisions.

Important:
Browser/GPTWeb stages that require JSON must be wrapped by the provider router
with JSON extraction, validation, and fallback. Browser output is text-first.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, Mapping, Optional


# ============================================================
# Run modes
# ============================================================

SUPPORTED_RUN_MODES = {"power", "development"}

RUN_MODE_ALIASES = {
    "power": "power",
    "power_mode": "power",
    "Power Mode": "power",
    "POWER": "power",
    "quality": "power",
    "all_openai": "power",

    "development": "development",
    "development_mode": "development",
    "Development Mode": "development",
    "dev": "development",
    "DEV": "development",
    "debug": "development",
    "budget": "development",
    "balanced": "development",
    "local_first": "development",
    "browser_assisted": "development",
}


def normalize_run_mode(run_mode: Optional[str] = None) -> str:
    raw = run_mode or os.getenv("ASCENDANT_RUN_MODE", "development")
    raw = str(raw).strip()
    normalized = RUN_MODE_ALIASES.get(raw, RUN_MODE_ALIASES.get(raw.lower(), raw.lower()))
    if normalized not in SUPPORTED_RUN_MODES:
        raise ValueError(
            f"Unsupported ASCENDANT_RUN_MODE={raw!r}. "
            f"Use one of: {sorted(SUPPORTED_RUN_MODES)}"
        )
    return normalized


DEFAULT_RUN_MODE = normalize_run_mode(os.getenv("ASCENDANT_RUN_MODE", "development"))


# ============================================================
# OpenAI prompt caching policy
# ============================================================

# Prompt caching is automatic when OpenAI requests share exact long prefixes.
# These optional settings add cache routing hints and retention policy for OpenAI-backed stages.
# Default is OFF because SDK/API support for explicit prompt-cache request fields can vary.
# The request content still controls the actual cacheable prefix: stable agent
# instructions and schemas should stay before dynamic run/task context.
OPENAI_PROMPT_CACHE_RETENTION = os.getenv("OPENAI_PROMPT_CACHE_RETENTION", "in-memory")
OPENAI_PROMPT_CACHE_ENABLED = os.getenv("OPENAI_PROMPT_CACHE_ENABLED", "0").strip().lower() not in {"0", "false", "no", "off"}


# ============================================================
# Provider profiles
# ============================================================

PROVIDER_PROFILES: Dict[str, Dict[str, Any]] = {
    "deterministic": {
        "provider": "deterministic",
        "model": None,
        "strict_json": True,
        "supports_parallel": True,
        "usage_source": "none",
        "max_input_tokens": 0,
        "max_output_tokens": 0,
        "temperature": None,
        "fallback_profile": None,
        "notes": "Use Python logic only. No model call should happen.",
    },

    # --------------------------
    # OpenAI profiles
    # --------------------------
    "openai_high_reasoning": {
        "provider": "openai",
        "model": os.getenv("OPENAI_HIGH_REASONING_MODEL", "gpt-5"),
        "strict_json": True,
        "supports_parallel": True,
        "usage_source": "provider_reported",
        "max_input_tokens": int(os.getenv("OPENAI_HIGH_MAX_INPUT", "30000")),
        "max_output_tokens": int(os.getenv("OPENAI_HIGH_MAX_OUTPUT", "6000")),
        "temperature": 0.2,
        "prompt_cache_enabled": OPENAI_PROMPT_CACHE_ENABLED,
        "prompt_cache_retention": OPENAI_PROMPT_CACHE_RETENTION,
        "fallback_profile": "openai_standard_json",
        "notes": "Architecture, hard reasoning, conflict resolution, and final high-value decisions.",
    },
    "openai_standard_json": {
        "provider": "openai",
        "model": os.getenv("OPENAI_STANDARD_MODEL", "gpt-5"),
        "strict_json": True,
        "supports_parallel": True,
        "usage_source": "provider_reported",
        "max_input_tokens": int(os.getenv("OPENAI_STANDARD_MAX_INPUT", "16000")),
        "max_output_tokens": int(os.getenv("OPENAI_STANDARD_MAX_OUTPUT", "3500")),
        "temperature": 0.2,
        "prompt_cache_enabled": OPENAI_PROMPT_CACHE_ENABLED,
        "prompt_cache_retention": OPENAI_PROMPT_CACHE_RETENTION,
        "fallback_profile": "openai_fast_json",
        "notes": "Default for reliable structured outputs, ordinary planning, QA, and gates.",
    },
    "openai_fast_json": {
        "provider": "openai",
        "model": os.getenv("OPENAI_FAST_MODEL", "gpt-4.1-mini"),
        "strict_json": True,
        "supports_parallel": True,
        "usage_source": "provider_reported",
        "max_input_tokens": int(os.getenv("OPENAI_FAST_MAX_INPUT", "8000")),
        "max_output_tokens": int(os.getenv("OPENAI_FAST_MAX_OUTPUT", "1800")),
        "temperature": 0.1,
        "prompt_cache_enabled": OPENAI_PROMPT_CACHE_ENABLED,
        "prompt_cache_retention": OPENAI_PROMPT_CACHE_RETENTION,
        "fallback_profile": "openai_standard_json",
        "notes": "Cheaper schema-bound OpenAI profile.",
    },
    "openai_text_standard": {
        "provider": "openai",
        "model": os.getenv("OPENAI_TEXT_MODEL", "gpt-5"),
        "strict_json": False,
        "supports_parallel": True,
        "usage_source": "provider_reported",
        "max_input_tokens": int(os.getenv("OPENAI_TEXT_MAX_INPUT", "12000")),
        "max_output_tokens": int(os.getenv("OPENAI_TEXT_MAX_OUTPUT", "2500")),
        "temperature": 0.3,
        "prompt_cache_enabled": OPENAI_PROMPT_CACHE_ENABLED,
        "prompt_cache_retention": OPENAI_PROMPT_CACHE_RETENTION,
        "fallback_profile": "openai_standard_json",
        "notes": "OpenAI profile for normal plain-text generation.",
    },
    "openai_codex_engineer_high": {
        "provider": "openai",
        "model": os.getenv("OPENAI_ENGINEER_CODEX_MODEL", os.getenv("OPENAI_CODEX_MODEL", "gpt-5.3-codex")),
        "strict_json": True,
        "supports_parallel": True,
        "usage_source": "provider_reported",
        "max_input_tokens": int(os.getenv("OPENAI_CODEX_HIGH_MAX_INPUT", "30000")),
        "max_output_tokens": int(os.getenv("OPENAI_CODEX_HIGH_MAX_OUTPUT", "6000")),
        "temperature": 0.2,
        "prompt_cache_enabled": OPENAI_PROMPT_CACHE_ENABLED,
        "prompt_cache_retention": OPENAI_PROMPT_CACHE_RETENTION,
        "fallback_profile": "openai_high_reasoning",
        "notes": "Codex-specialized OpenAI profile for complex Engineer WorkItems.",
    },
    "openai_codex_engineer_standard": {
        "provider": "openai",
        "model": os.getenv("OPENAI_ENGINEER_CODEX_MODEL", os.getenv("OPENAI_CODEX_MODEL", "gpt-5.3-codex")),
        "strict_json": True,
        "supports_parallel": True,
        "usage_source": "provider_reported",
        "max_input_tokens": int(os.getenv("OPENAI_CODEX_STANDARD_MAX_INPUT", "16000")),
        "max_output_tokens": int(os.getenv("OPENAI_CODEX_STANDARD_MAX_OUTPUT", "3500")),
        "temperature": 0.2,
        "prompt_cache_enabled": OPENAI_PROMPT_CACHE_ENABLED,
        "prompt_cache_retention": OPENAI_PROMPT_CACHE_RETENTION,
        "fallback_profile": "openai_standard_json",
        "notes": "Codex-specialized OpenAI profile for ordinary Engineer WorkItems.",
    },
    "openai_codex_engineer_fast": {
        "provider": "openai",
        "model": os.getenv("OPENAI_ENGINEER_CODEX_MODEL", os.getenv("OPENAI_CODEX_MODEL", "gpt-5.3-codex")),
        "strict_json": True,
        "supports_parallel": True,
        "usage_source": "provider_reported",
        "max_input_tokens": int(os.getenv("OPENAI_CODEX_FAST_MAX_INPUT", "8000")),
        "max_output_tokens": int(os.getenv("OPENAI_CODEX_FAST_MAX_OUTPUT", "1800")),
        "temperature": 0.1,
        "prompt_cache_enabled": OPENAI_PROMPT_CACHE_ENABLED,
        "prompt_cache_retention": OPENAI_PROMPT_CACHE_RETENTION,
        "fallback_profile": "openai_fast_json",
        "notes": "Codex-specialized OpenAI profile for low-risk/simple Engineer WorkItems.",
    },

    # --------------------------
    # Ollama local profiles
    # --------------------------
    "ollama_local_small": {
        "provider": "ollama",
        "model": os.getenv("OLLAMA_SMALL_MODEL", "llama3.2:1b"),
        "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "strict_json": False,
        "supports_parallel": False,
        "usage_source": "local_estimated",
        "max_input_tokens": int(os.getenv("OLLAMA_SMALL_MAX_INPUT", "6000")),
        "max_output_tokens": int(os.getenv("OLLAMA_SMALL_MAX_OUTPUT", "1200")),
        "temperature": 0.1,
        "fallback_profile": "openai_fast_json",
        "json_validation_required": True,
        "notes": "Cheap local summarization, tagging, relevance checks, and compression tests.",
    },
    "ollama_local_medium": {
        "provider": "ollama",
        "model": os.getenv("OLLAMA_MEDIUM_MODEL", "llama3.2:3b"),
        "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "strict_json": False,
        "supports_parallel": False,
        "usage_source": "local_estimated",
        "max_input_tokens": int(os.getenv("OLLAMA_MEDIUM_MAX_INPUT", "10000")),
        "max_output_tokens": int(os.getenv("OLLAMA_MEDIUM_MAX_OUTPUT", "2000")),
        "temperature": 0.15,
        "fallback_profile": "openai_standard_json",
        "json_validation_required": True,
        "notes": "Use only if local hardware can handle it.",
    },

    # --------------------------
    # GPTWeb/browser profiles
    # --------------------------
    "browser_manager_text": {
        "provider": "browser",
        "model": os.getenv("GPTWEB_MODEL_LABEL", "private_website"),
        "url": os.getenv("GPTWEB_URL", ""),
        "connect_cdp_url": os.getenv("GPTWEB_CDP_URL", "http://localhost:9222"),
        "strict_json": False,
        "supports_parallel": False,
        "usage_source": "local_estimated",
        "max_input_tokens": int(os.getenv("GPTWEB_MANAGER_MAX_INPUT", "12000")),
        "max_output_tokens": int(os.getenv("GPTWEB_MANAGER_MAX_OUTPUT", "3000")),
        "temperature": None,
        "fallback_profile": "openai_standard_json",
        "append_text_only_suffix": True,
        "json_validation_required": True,
        "notes": "Development-mode provider for high-level/admin/management agents. JSON must be validated by router.",
    },
    "browser_low_risk_text": {
        "provider": "browser",
        "model": os.getenv("GPTWEB_MODEL_LABEL", "private_website"),
        "url": os.getenv("GPTWEB_URL", ""),
        "connect_cdp_url": os.getenv("GPTWEB_CDP_URL", "http://localhost:9222"),
        "strict_json": False,
        "supports_parallel": False,
        "usage_source": "local_estimated",
        "max_input_tokens": int(os.getenv("GPTWEB_LOW_RISK_MAX_INPUT", "8000")),
        "max_output_tokens": int(os.getenv("GPTWEB_LOW_RISK_MAX_OUTPUT", "1800")),
        "temperature": None,
        "fallback_profile": "ollama_local_small",
        "append_text_only_suffix": True,
        "json_validation_required": False,
        "notes": "Low-risk text drafts, summaries, and wording.",
    },
}


# ============================================================
# Stage-level cache-key mapping
# ============================================================

def _cache_key_for_stage(stage_key: str) -> str:
    """Return a stable cache routing key for OpenAI-backed calls."""
    family = stage_key.split(".", 1)[0].replace("_", "-")
    if stage_key.startswith("eng_lead."):
        family = "engineering-lead"
    elif stage_key.startswith("qa."):
        family = "qa"
    elif stage_key.startswith("coordinator."):
        family = "coordinator"
    elif stage_key.startswith("docs."):
        family = "docs"
    return f"ascendant-{family}"


# ============================================================
# Stage registry
# ============================================================

STAGE_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Intake / administrative front door
    "intake.generate": {
        "current_agent_file": "intake_agent.py",
        "default_profile": "openai_fast_json",
        "allowed_profiles": ["openai_standard_json", "openai_fast_json", "browser_manager_text"],
        "requires_strict_json": True,
        "criticality": "high",
        "stage_family": "administrative",
        "description": "Convert vague user request into structured initial_input or clarification questions.",
        "cost_strategy": "Compact prompt; strict JSON. Development mode can use GPTWeb with validation/fallback.",
    },
    "intake.revise_after_user_feedback": {
        "current_agent_file": "intake_agent.py",
        "default_profile": "openai_fast_json",
        "allowed_profiles": ["openai_standard_json", "openai_fast_json", "browser_manager_text"],
        "requires_strict_json": True,
        "criticality": "high",
        "stage_family": "administrative",
        "description": "Revise intake after user feedback.",
        "cost_strategy": "Send prior draft plus new feedback only.",
    },

    # Product management / high-level product definition
    "pm.generate": {
        "current_agent_file": "pm_agent.py",
        "default_profile": "openai_high_reasoning",
        "allowed_profiles": ["openai_high_reasoning", "openai_standard_json", "openai_fast_json", "browser_manager_text"],
        "requires_strict_json": True,
        "criticality": "high",
        "stage_family": "management",
        "description": "Generate PM handoff for design and engineering.",
        "cost_strategy": "Approved initial_input only plus relevant asset summary. Development mode can use GPTWeb with validation/fallback.",
    },
    "pm.review": {
        "current_agent_file": "pm_agent.py",
        "default_profile": "openai_standard_json",
        "allowed_profiles": ["openai_standard_json", "openai_fast_json", "browser_manager_text"],
        "requires_strict_json": True,
        "criticality": "medium",
        "stage_family": "management",
        "description": "Product review of target artifact.",
        "cost_strategy": "Target artifact summary plus review criteria.",
    },

    # UX / Designer as implementation-facing agent
    "ux.generate": {
        "current_agent_file": "ux_agent.py",
        "default_profile": "openai_high_reasoning",
        "allowed_profiles": ["openai_high_reasoning", "openai_standard_json", "openai_fast_json"],
        "requires_strict_json": True,
        "criticality": "high",
        "stage_family": "implementation",
        "description": "Implementation-ready UX/UI/design-system spec.",
        "cost_strategy": "OpenAI in both modes because Designer/UX is implementation-facing.",
    },
    "ux.revise_after_resource_decision": {
        "current_agent_file": "ux_agent.py",
        "default_profile": "openai_standard_json",
        "allowed_profiles": ["openai_standard_json", "openai_fast_json"],
        "requires_strict_json": True,
        "criticality": "high",
        "stage_family": "implementation",
        "description": "Revise UX after resource decision.",
        "cost_strategy": "OpenAI in both modes because UX changes affect implementation contracts.",
    },
    "ux.light_copy_or_microcopy": {
        "current_agent_file": "ux_agent.py",
        "default_profile": "browser_low_risk_text",
        "allowed_profiles": ["browser_low_risk_text", "ollama_local_small", "openai_text_standard"],
        "requires_strict_json": False,
        "criticality": "low",
        "stage_family": "low_risk_text",
        "description": "Non-critical UI wording.",
        "cost_strategy": "Browser/local acceptable; this is not a gate.",
    },

    # Engineering Lead / implementation architecture
    "eng_lead.generate_plan": {
        "current_agent_file": "eng_lead_agent.py",
        "default_profile": "openai_high_reasoning",
        "allowed_profiles": ["openai_high_reasoning", "openai_standard_json"],
        "requires_strict_json": True,
        "criticality": "mission_critical",
        "stage_family": "implementation",
        "description": "Architecture plan, execution plan, and role-agnostic WorkItems.",
        "cost_strategy": "OpenAI in both modes. This is the implementation task graph source.",
    },
    "eng_lead.review": {
        "current_agent_file": "eng_lead_agent.py",
        "default_profile": "openai_standard_json",
        "allowed_profiles": ["openai_standard_json", "openai_fast_json"],
        "requires_strict_json": True,
        "criticality": "high",
        "stage_family": "implementation",
        "description": "Engineering feasibility/architecture review.",
        "cost_strategy": "OpenAI in both modes for architecture correctness.",
    },
    "eng_lead.validate_task_graph": {
        "current_agent_file": "eng_lead_agent.py / future graph validator",
        "default_profile": "deterministic",
        "allowed_profiles": ["deterministic", "openai_fast_json", "openai_standard_json"],
        "requires_strict_json": True,
        "criticality": "high",
        "stage_family": "implementation_control",
        "description": "Validate WorkItems for dependency and scope issues.",
        "cost_strategy": "Deterministic first; OpenAI only for ambiguity.",
    },
    "eng_lead.runbook": {
        "current_agent_file": "eng_lead_agent.py",
        "default_profile": "openai_standard_json",
        "allowed_profiles": ["openai_standard_json", "openai_text_standard"],
        "requires_strict_json": True,
        "criticality": "medium",
        "stage_family": "administrative",
        "description": "Human run instructions after engineering execution.",
        "cost_strategy": "OpenAI in both modes. Runbook is operational documentation and should not use GPTWeb by default.",
    },

    # Deterministic orchestration
    "orchestrator.ingest_plan": {
        "current_agent_file": "eng_orchestrator_agent.py",
        "default_profile": "deterministic",
        "allowed_profiles": ["deterministic"],
        "requires_strict_json": True,
        "criticality": "mission_critical",
        "stage_family": "deterministic_control",
        "description": "Ingest WorkItems into DAG/task queue.",
        "cost_strategy": "No LLM.",
    },
    "orchestrator.claim_next": {
        "current_agent_file": "eng_orchestrator_agent.py",
        "default_profile": "deterministic",
        "allowed_profiles": ["deterministic"],
        "requires_strict_json": True,
        "criticality": "mission_critical",
        "stage_family": "deterministic_control",
        "description": "Claim available task.",
        "cost_strategy": "No LLM.",
    },
    "orchestrator.submit_result": {
        "current_agent_file": "eng_orchestrator_agent.py",
        "default_profile": "deterministic",
        "allowed_profiles": ["deterministic"],
        "requires_strict_json": True,
        "criticality": "mission_critical",
        "stage_family": "deterministic_control",
        "description": "Update queue state after result.",
        "cost_strategy": "No LLM.",
    },

    # Engineer / implementation agents
    "engineer.execute_detailed": {
        "current_agent_file": "eng_agent.py",
        "default_profile": "openai_codex_engineer_high",
        "allowed_profiles": ["openai_codex_engineer_high", "openai_codex_engineer_standard", "openai_high_reasoning", "openai_standard_json"],
        "requires_strict_json": True,
        "criticality": "high",
        "stage_family": "implementation",
        "description": "Complex WorkItem implementation planning/output.",
        "cost_strategy": "OpenAI in both modes. Task-specific context pack only.",
    },
    "engineer.execute_standard": {
        "current_agent_file": "eng_agent.py",
        "default_profile": "openai_codex_engineer_standard",
        "allowed_profiles": ["openai_codex_engineer_high", "openai_codex_engineer_standard", "openai_codex_engineer_fast", "openai_high_reasoning", "openai_standard_json", "openai_fast_json"],
        "requires_strict_json": True,
        "criticality": "medium",
        "stage_family": "implementation",
        "description": "Ordinary WorkItem implementation output.",
        "cost_strategy": "OpenAI in both modes. Compiled task pack only.",
    },
    "engineer.execute_simple": {
        "current_agent_file": "eng_agent.py",
        "default_profile": "openai_codex_engineer_fast",
        "allowed_profiles": ["openai_codex_engineer_fast", "openai_codex_engineer_standard", "openai_fast_json", "openai_standard_json", "ollama_local_small", "browser_low_risk_text"],
        "requires_strict_json": False,
        "criticality": "low",
        "stage_family": "implementation_low_risk",
        "description": "Low-risk helper code or draft notes.",
        "cost_strategy": "Power uses OpenAI. Development may use OpenAI unless explicitly routed lower.",
    },
    "engineer.fix_after_qa": {
        "current_agent_file": "eng_agent.py",
        "default_profile": "openai_codex_engineer_standard",
        "allowed_profiles": ["openai_codex_engineer_standard", "openai_codex_engineer_high", "openai_standard_json", "openai_high_reasoning"],
        "requires_strict_json": True,
        "criticality": "high",
        "stage_family": "implementation",
        "description": "Revise work after QA issues.",
        "cost_strategy": "OpenAI in both modes.",
    },
    "engineer.interface_summary": {
        "current_agent_file": "eng_agent.py / future context processor",
        "default_profile": "ollama_local_small",
        "allowed_profiles": ["ollama_local_small", "browser_low_risk_text", "openai_fast_json"],
        "requires_strict_json": False,
        "criticality": "low",
        "stage_family": "context_compression",
        "description": "Summarize handoff interfaces for downstream tasks.",
        "cost_strategy": "Local Ollama preferred in both modes.",
    },

    # QA as implementation gate
    "qa.review_work_result": {
        "current_agent_file": "qa_agent.py",
        "default_profile": "openai_standard_json",
        "allowed_profiles": ["openai_standard_json", "openai_fast_json"],
        "requires_strict_json": True,
        "criticality": "high",
        "stage_family": "implementation",
        "description": "Review Engineer WorkResult and decide queue update.",
        "cost_strategy": "OpenAI in both modes.",
    },
    "qa.light_review": {
        "current_agent_file": "qa_agent.py",
        "default_profile": "openai_fast_json",
        "allowed_profiles": ["openai_fast_json", "ollama_local_small"],
        "requires_strict_json": True,
        "criticality": "medium",
        "stage_family": "implementation",
        "description": "Cheap review for low-risk tasks.",
        "cost_strategy": "OpenAI by default. Ollama only advisory if explicitly routed later.",
    },
    "qa.final_build_evidence_review": {
        "current_agent_file": "qa_agent.py / future release gate",
        "default_profile": "openai_standard_json",
        "allowed_profiles": ["openai_standard_json", "openai_high_reasoning"],
        "requires_strict_json": True,
        "criticality": "mission_critical",
        "stage_family": "implementation",
        "description": "Review final build/test evidence.",
        "cost_strategy": "OpenAI in both modes.",
    },

    # Coordinator / management
    "coordinator.gate_decision": {
        "current_agent_file": "coordinator_agent.py",
        "default_profile": "openai_high_reasoning",
        "allowed_profiles": ["openai_high_reasoning", "openai_standard_json", "browser_manager_text"],
        "requires_strict_json": True,
        "criticality": "mission_critical",
        "stage_family": "management",
        "description": "Proceed/revise/block/human-input decision.",
        "cost_strategy": "Power uses OpenAI. Development uses GPTWeb with validation/fallback.",
    },
    "coordinator.workflow_decision": {
        "current_agent_file": "coordinator_agent.py",
        "default_profile": "openai_standard_json",
        "allowed_profiles": ["openai_high_reasoning", "openai_standard_json"],
        "requires_strict_json": True,
        "criticality": "mission_critical",
        "stage_family": "management",
        "description": "Abnormal-state workflow decision: continue, rerun, ask user, block, or finish.",
        "cost_strategy": "OpenAI in both modes because this may be invoked from parallel worker-loop exceptions.",
    },

    "coordinator.conflict_resolution": {
        "current_agent_file": "coordinator_agent.py",
        "default_profile": "openai_high_reasoning",
        "allowed_profiles": ["openai_high_reasoning", "openai_standard_json", "browser_manager_text"],
        "requires_strict_json": True,
        "criticality": "mission_critical",
        "stage_family": "management",
        "description": "Resolve conflicts between agents.",
        "cost_strategy": "Power uses OpenAI. Development can use GPTWeb, but router must fall back if JSON or quality fails.",
    },
    "coordinator.final_handoff": {
        "current_agent_file": "coordinator_agent.py",
        "default_profile": "openai_standard_json",
        "allowed_profiles": ["openai_standard_json", "openai_fast_json", "openai_text_standard", "browser_manager_text"],
        "requires_strict_json": True,
        "criticality": "medium",
        "stage_family": "management",
        "description": "Final execution summary, risks, blockers, and next actions.",
        "cost_strategy": "Development mode can use GPTWeb with validation/fallback.",
    },

    # Resource / human-in-loop
    "resource.extract_requests": {
        "current_agent_file": "operation.py / resource_eval.py",
        "default_profile": "deterministic",
        "allowed_profiles": ["deterministic"],
        "requires_strict_json": True,
        "criticality": "high",
        "stage_family": "deterministic_control",
        "description": "Extract resource requests from agent outputs.",
        "cost_strategy": "Known fields only.",
    },
    "resource.normalize_request_text": {
        "current_agent_file": "resource_eval.py / future resource_normalizer.py",
        "default_profile": "ollama_local_small",
        "allowed_profiles": ["ollama_local_small", "openai_fast_json"],
        "requires_strict_json": True,
        "criticality": "low",
        "stage_family": "context_compression",
        "description": "Normalize resource request wording.",
        "cost_strategy": "Local Ollama in both modes unless parse fails.",
    },
    "resource.resolve_request": {
        "current_agent_file": "resource_eval.py",
        "default_profile": "deterministic",
        "allowed_profiles": ["deterministic"],
        "requires_strict_json": True,
        "criticality": "high",
        "stage_family": "deterministic_control",
        "description": "Record resource resolution and update asset manifest.",
        "cost_strategy": "No LLM.",
    },

    # Token Governor / Context Compiler / future compressor
    "context.preprocess_agent_input": {
        "current_agent_file": "context_compressor.py",
        "default_profile": "ollama_local_small",
        "allowed_profiles": ["ollama_local_small", "ollama_local_medium"],
        "requires_strict_json": False,
        "criticality": "medium",
        "stage_family": "context_compression",
        "description": "Preprocess intended AI input into a shorter temporary input before target agent call.",
        "cost_strategy": "Ollama in both modes; if Ollama compression fails, preserve the original input unchanged.",
    },

    "token.estimate_usage": {
        "current_agent_file": "future_token_governor.py",
        "default_profile": "deterministic",
        "allowed_profiles": ["deterministic"],
        "requires_strict_json": True,
        "criticality": "mission_critical",
        "stage_family": "deterministic_control",
        "description": "Estimate token/cost and enforce budgets.",
        "cost_strategy": "No LLM.",
    },
    "context.compile_task_pack": {
        "current_agent_file": "future_context_compiler.py",
        "default_profile": "deterministic",
        "allowed_profiles": ["deterministic"],
        "requires_strict_json": True,
        "criticality": "mission_critical",
        "stage_family": "deterministic_control",
        "description": "Assemble minimal task-specific context pack.",
        "cost_strategy": "No LLM by default.",
    },
    "context.compress_artifact": {
        "current_agent_file": "future_context_processor.py",
        "default_profile": "ollama_local_small",
        "allowed_profiles": ["ollama_local_small", "ollama_local_medium"],
        "requires_strict_json": False,
        "criticality": "medium",
        "stage_family": "context_compression",
        "description": "Compress large artifact into reusable summary.",
        "cost_strategy": "Ollama in both Power Mode and Development Mode.",
    },
    "context.create_artifact_index": {
        "current_agent_file": "future_context_processor.py",
        "default_profile": "ollama_local_small",
        "allowed_profiles": ["ollama_local_small", "ollama_local_medium", "deterministic"],
        "requires_strict_json": True,
        "criticality": "medium",
        "stage_family": "context_compression",
        "description": "Create section/component/task relevance index.",
        "cost_strategy": "Deterministic parsing first; Ollama in both modes for semantic tagging.",
    },
    "context.section_relevance": {
        "current_agent_file": "future_context_processor.py",
        "default_profile": "ollama_local_small",
        "allowed_profiles": ["ollama_local_small", "ollama_local_medium", "deterministic"],
        "requires_strict_json": True,
        "criticality": "medium",
        "stage_family": "context_compression",
        "description": "Determine section relevance to WorkItem.",
        "cost_strategy": "Keyword/path first; Ollama in both modes for ambiguity.",
    },
    "context.compress_logs": {
        "current_agent_file": "future_context_processor.py",
        "default_profile": "ollama_local_small",
        "allowed_profiles": ["ollama_local_small", "ollama_local_medium"],
        "requires_strict_json": False,
        "criticality": "low",
        "stage_family": "context_compression",
        "description": "Compress build/test/error logs.",
        "cost_strategy": "Ollama in both modes. Preserve exact error lines.",
    },

    # Browser provider utility
    "browser.generate_plain_text": {
        "current_agent_file": "gpt_web_collector.py / browser_model_client.py",
        "default_profile": "browser_manager_text",
        "allowed_profiles": ["browser_manager_text", "browser_low_risk_text"],
        "requires_strict_json": False,
        "criticality": "low",
        "stage_family": "browser_utility",
        "description": "Use browser/private website for plain-text output.",
        "cost_strategy": "Single-thread only; append text-only suffix.",
    },
    "browser.validate_json_output": {
        "current_agent_file": "future_browser_model_client.py",
        "default_profile": "deterministic",
        "allowed_profiles": ["deterministic", "openai_fast_json"],
        "requires_strict_json": True,
        "criticality": "high",
        "stage_family": "deterministic_control",
        "description": "Validate or repair browser JSON-like text.",
        "cost_strategy": "Deterministic parse/extract first; OpenAI repair only if needed.",
    },

    # Documentation / reporting
    "docs.generate_internal_spec": {
        "current_agent_file": "future_docs_agent.py",
        "default_profile": "openai_text_standard",
        "allowed_profiles": ["openai_text_standard", "browser_manager_text"],
        "requires_strict_json": False,
        "criticality": "medium",
        "stage_family": "administrative",
        "description": "Generate internal technical documentation.",
        "cost_strategy": "Power uses OpenAI. Development uses GPTWeb.",
    },
    "docs.summarize_run": {
        "current_agent_file": "coordinator_agent.py / future_docs_agent.py",
        "default_profile": "browser_low_risk_text",
        "allowed_profiles": ["browser_low_risk_text", "ollama_local_small", "openai_text_standard"],
        "requires_strict_json": False,
        "criticality": "low",
        "stage_family": "administrative",
        "description": "Non-critical run summaries or notes.",
        "cost_strategy": "Development uses GPTWeb/local. Power uses OpenAI unless explicitly low-risk.",
    },
}


# ============================================================
# Two run modes only
# ============================================================

RUN_MODE_OVERRIDES: Dict[str, Dict[str, str]] = {
    # Power Mode:
    # - all substantive AI agents use OpenAI
    # - deterministic remains deterministic
    # - context compressor/index/log compression uses Ollama in both modes
    "power": {
        "intake.generate": "openai_fast_json",
        "intake.revise_after_user_feedback": "openai_fast_json",

        "pm.generate": "openai_high_reasoning",
        "pm.review": "openai_standard_json",

        "ux.generate": "openai_high_reasoning",
        "ux.revise_after_resource_decision": "openai_standard_json",
        "ux.light_copy_or_microcopy": "openai_text_standard",

        "eng_lead.generate_plan": "openai_high_reasoning",
        "eng_lead.review": "openai_standard_json",
        "eng_lead.validate_task_graph": "deterministic",
        "eng_lead.runbook": "openai_standard_json",

        "engineer.execute_detailed": "openai_codex_engineer_high",
        "engineer.execute_standard": "openai_codex_engineer_standard",
        "engineer.execute_simple": "openai_codex_engineer_fast",
        "engineer.fix_after_qa": "openai_codex_engineer_standard",
        "engineer.interface_summary": "ollama_local_small",

        "qa.review_work_result": "openai_standard_json",
        "qa.light_review": "openai_fast_json",
        "qa.final_build_evidence_review": "openai_high_reasoning",

        "coordinator.gate_decision": "openai_high_reasoning",
        "coordinator.workflow_decision": "openai_high_reasoning",
        "coordinator.conflict_resolution": "openai_high_reasoning",
        "coordinator.final_handoff": "openai_standard_json",

        "resource.normalize_request_text": "ollama_local_small",

        "context.preprocess_agent_input": "ollama_local_small",
        "context.compress_artifact": "ollama_local_small",
        "context.create_artifact_index": "ollama_local_small",
        "context.section_relevance": "ollama_local_small",
        "context.compress_logs": "ollama_local_small",

        "docs.generate_internal_spec": "openai_text_standard",
        "docs.summarize_run": "openai_text_standard",
    },

    # Development Mode:
    # - implementation agents use OpenAI
    # - operational runbook uses OpenAI
    # - high-level/admin/management agents use GPTWeb/browser
    # - context compressor/index/log compression uses local Ollama
    # - browser/GPTWeb is never selected for parallel worker-loop stages
    "development": {
        "intake.generate": "openai_fast_json",
        "intake.revise_after_user_feedback": "openai_fast_json",

        "pm.generate": "openai_standard_json",
        "pm.review": "openai_fast_json",

        "ux.generate": "openai_standard_json",
        "ux.revise_after_resource_decision": "openai_standard_json",
        "ux.light_copy_or_microcopy": "browser_low_risk_text",

        "eng_lead.generate_plan": "openai_high_reasoning",
        "eng_lead.review": "openai_standard_json",
        "eng_lead.validate_task_graph": "deterministic",
        "eng_lead.runbook": "openai_standard_json",

        "engineer.execute_detailed": "openai_codex_engineer_high",
        "engineer.execute_standard": "openai_codex_engineer_standard",
        "engineer.execute_simple": "openai_codex_engineer_fast",
        "engineer.fix_after_qa": "openai_codex_engineer_standard",
        "engineer.interface_summary": "ollama_local_small",

        "qa.review_work_result": "openai_standard_json",
        "qa.light_review": "openai_fast_json",
        "qa.final_build_evidence_review": "openai_high_reasoning",

        "coordinator.gate_decision": "openai_standard_json",
        "coordinator.workflow_decision": "openai_standard_json",
        "coordinator.conflict_resolution": "openai_standard_json",
        "coordinator.final_handoff": "openai_standard_json",

        "resource.normalize_request_text": "ollama_local_small",

        "context.preprocess_agent_input": "ollama_local_small",
        "context.compress_artifact": "ollama_local_small",
        "context.create_artifact_index": "ollama_local_small",
        "context.section_relevance": "ollama_local_small",
        "context.compress_logs": "ollama_local_small",

        "docs.generate_internal_spec": "browser_manager_text",
        "docs.summarize_run": "browser_low_risk_text",
    },
}


# ============================================================
# Aliases
# ============================================================

STAGE_ALIASES: Dict[str, str] = {
    "intake": "intake.generate",
    "pm": "pm.generate",
    "ux": "ux.generate",
    "designer": "ux.generate",
    "engineering_lead": "eng_lead.generate_plan",
    "eng_lead": "eng_lead.generate_plan",
    "engineer": "engineer.execute_standard",
    "engineer_detailed": "engineer.execute_detailed",
    "engineer_simple": "engineer.execute_simple",
    "qa": "qa.review_work_result",
    "coordinator": "coordinator.gate_decision",
    "coordinator_workflow": "coordinator.workflow_decision",
    "workflow_decision": "coordinator.workflow_decision",
    "final_handoff": "coordinator.final_handoff",
    "artifact_summary": "context.compress_artifact",
    "context_compression": "context.compress_artifact",
    "input_compression": "context.preprocess_agent_input",
    "section_relevance": "context.section_relevance",
    "log_compression": "context.compress_logs",
    "browser_text": "browser.generate_plain_text",
}



# ============================================================
# Concurrency policy
# ============================================================

# Stages in this set may run inside worker loops or may be triggered by multiple
# workers during the same run. They must never use the browser/GPTWeb provider.
# OpenAI API and deterministic/local processing are safe defaults for these paths.
PARALLEL_STAGE_KEYS = {
    "engineer.execute_detailed",
    "engineer.execute_standard",
    "engineer.execute_simple",
    "engineer.fix_after_qa",
    "engineer.interface_summary",
    "qa.review_work_result",
    "qa.light_review",
    "qa.final_build_evidence_review",
    "executor.apply_patch",
    "executor.run_tests",
    "executor.classify_failure_log",
}

# Browser/GPTWeb may only be used on stages that are sequential or lock-protected.
# These stages should be routed through a global browser provider lock at runtime.
BROWSER_ALLOWED_STAGE_KEYS = {
    "intake.generate",
    "intake.revise_after_user_feedback",
    "pm.generate",
    "pm.review",
    "coordinator.gate_decision",
    "coordinator.conflict_resolution",
    "coordinator.final_handoff",
    "ux.light_copy_or_microcopy",
    "browser.generate_plain_text",
    "docs.generate_internal_spec",
    "docs.summarize_run",
}

# ============================================================
# Safety and routing constraints
# ============================================================

ROUTING_CONSTRAINTS: Dict[str, Any] = {
    "browser_single_thread_lock_required": True,
    "browser_output_is_text_first": True,
    "browser_json_requires_validation_and_fallback": True,
    "ollama_json_requires_validation_and_fallback": True,
    "deterministic_stages_must_not_call_llm": True,
    "tools_default": "off",
    "browser_text_only_suffix": (
        "Please output your reply in plain text, not in the form of a document, "
        "so I can copy and paste it."
    ),
    "browser_must_not_route_parallel_stages": True,
    "browser_allowed_stage_keys": sorted(BROWSER_ALLOWED_STAGE_KEYS),
    "parallel_stage_keys": sorted(PARALLEL_STAGE_KEYS),
}


# ============================================================
# Helper functions
# ============================================================

def resolve_stage_key(stage_key_or_alias: str) -> str:
    return STAGE_ALIASES.get(stage_key_or_alias, stage_key_or_alias)


def get_stage_config(stage_key_or_alias: str) -> Dict[str, Any]:
    stage_key = resolve_stage_key(stage_key_or_alias)
    if stage_key not in STAGE_REGISTRY:
        raise KeyError(f"Unknown stage key: {stage_key_or_alias!r} resolved to {stage_key!r}")
    cfg = deepcopy(STAGE_REGISTRY[stage_key])
    cfg["stage_key"] = stage_key
    return cfg


def get_stage_profile_name(stage_key_or_alias: str, run_mode: Optional[str] = None) -> str:
    normalized_mode = normalize_run_mode(run_mode)
    stage_key = resolve_stage_key(stage_key_or_alias)
    stage_cfg = get_stage_config(stage_key)

    profile_name = RUN_MODE_OVERRIDES.get(normalized_mode, {}).get(
        stage_key,
        stage_cfg["default_profile"],
    )

    allowed = stage_cfg.get("allowed_profiles", [])
    if profile_name not in allowed:
        raise ValueError(
            f"Profile {profile_name!r} is not allowed for stage {stage_key!r}. "
            f"Allowed profiles: {allowed}"
        )

    return profile_name


def get_provider_profile(profile_name: str) -> Dict[str, Any]:
    if profile_name not in PROVIDER_PROFILES:
        raise KeyError(f"Unknown provider profile: {profile_name!r}")
    profile = deepcopy(PROVIDER_PROFILES[profile_name])
    profile["profile_name"] = profile_name
    return profile


def get_stage_profile(stage_key_or_alias: str, run_mode: Optional[str] = None) -> Dict[str, Any]:
    normalized_mode = normalize_run_mode(run_mode)
    stage_cfg = get_stage_config(stage_key_or_alias)
    profile_name = get_stage_profile_name(stage_cfg["stage_key"], run_mode=normalized_mode)
    provider_profile = get_provider_profile(profile_name)

    merged = {
        **provider_profile,
        "stage_key": stage_cfg["stage_key"],
        "stage_description": stage_cfg["description"],
        "stage_family": stage_cfg.get("stage_family"),
        "current_agent_file": stage_cfg.get("current_agent_file"),
        "criticality": stage_cfg.get("criticality"),
        "requires_strict_json": stage_cfg.get("requires_strict_json"),
        "cost_strategy": stage_cfg.get("cost_strategy"),
        "allowed_profiles": stage_cfg.get("allowed_profiles", []),
        "run_mode": normalized_mode,
        "prompt_cache_key": stage_cfg.get("prompt_cache_key") or _cache_key_for_stage(stage_cfg["stage_key"]),
        "prompt_cache_retention": stage_cfg.get("prompt_cache_retention") or provider_profile.get("prompt_cache_retention"),
        "prompt_cache_enabled": bool(provider_profile.get("prompt_cache_enabled", False)),
    }

    validate_stage_profile(merged)
    return merged


def validate_stage_profile(profile: Mapping[str, Any]) -> None:
    stage_key = profile.get("stage_key")
    provider = profile.get("provider")
    requires_strict_json = profile.get("requires_strict_json")
    strict_json = profile.get("strict_json")

    if provider == "deterministic" and profile.get("model") is not None:
        raise ValueError(f"Deterministic stage {stage_key!r} must not define a model.")

    if provider == "deterministic" and profile.get("max_input_tokens", 0) != 0:
        raise ValueError(f"Deterministic stage {stage_key!r} must not define token budgets.")

    if provider == "browser":
        if stage_key in PARALLEL_STAGE_KEYS and ROUTING_CONSTRAINTS["browser_must_not_route_parallel_stages"]:
            raise ValueError(
                f"Browser/GPTWeb provider is not allowed for parallel-capable stage {stage_key!r}."
            )
        if stage_key not in BROWSER_ALLOWED_STAGE_KEYS:
            raise ValueError(
                f"Browser/GPTWeb provider is not explicitly allowed for stage {stage_key!r}. "
                f"Add it to BROWSER_ALLOWED_STAGE_KEYS only if the stage is sequential and lock-protected."
            )

    if provider in {"browser", "ollama"} and requires_strict_json and not strict_json:
        # This is allowed by configuration, but provider router must validate/extract JSON
        # and fall back to the configured fallback_profile if parsing fails.
        if not profile.get("json_validation_required", False):
            raise ValueError(
                f"Stage {stage_key!r} requires strict JSON but profile {profile.get('profile_name')!r} "
                f"does not declare json_validation_required=True."
            )


def describe_run_mode(run_mode: Optional[str] = None) -> Dict[str, str]:
    normalized_mode = normalize_run_mode(run_mode)
    if normalized_mode == "power":
        return {
            "run_mode": "power",
            "display_name": "Power Mode",
            "description": "All substantive AI stages use OpenAI API; context compression uses local Ollama.",
        }
    return {
        "run_mode": "development",
        "display_name": "Development Mode",
        "description": (
            "Implementation agents and operational runbook use OpenAI API; "
            "high-level/admin/management agents use lock-protected GPTWeb/browser; "
            "context compression uses local Ollama."
        ),
    }


if __name__ == "__main__":
    for mode in ["power", "development"]:
        print(f"\n=== {describe_run_mode(mode)['display_name']} ===")
        for key in [
            "intake",
            "pm",
            "ux",
            "engineering_lead",
            "engineer",
            "qa",
            "coordinator",
            "artifact_summary",
            "browser_text",
        ]:
            try:
                p = get_stage_profile(key, run_mode=mode)
                print(f"{key:24s} -> {p['profile_name']:24s} ({p['provider']})")
            except Exception as exc:
                print(f"{key:24s} -> ERROR: {exc}")
