"""
gpt_web_collector.py

Simplified single-file Playwright wrapper for interacting with a ChatGPT-like webpage UI.

Features:
- Callable GPTWeb(...) function.
- CLI usage.
- CDP attach mode for an already-open Chrome window.
- Stronger reply-waiting logic to avoid partial replies like "REA".
- Writes GPTWeb logs into outputs/GPTWeb_output/ by default.
- No screenshots.
- No HTML snapshots.
- No preview-document capture.

Install:

    pip install playwright
    python -m playwright install chromium

Start attachable Chrome on macOS:

    /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome \\
      --remote-debugging-port=9222 \\
      --user-data-dir=/path/to/manual_chrome_profile

Verify attachable Chrome:

    curl http://localhost:9222/json/version

CLI example:

    python gpt_web_collector.py ask \\
      --url "https://chatgpt.com" \\
      --prompt "Reply with exactly: READY" \\
      --connect-cdp-url "http://localhost:9222"

Python example:

    from gpt_web_collector import GPTWeb

    reply = GPTWeb(
        "Reply with exactly: READY",
        url="https://chatgpt.com",
        connect_cdp_url="http://localhost:9222"
    )

Important:
- This script automates only the visible webpage UI.
- Do not use it to bypass CAPTCHA, 2FA, login restrictions, paywalls, rate limits, or anti-bot systems.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from urllib.parse import urlparse

from playwright.sync_api import (
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)


# =========================
# Default configuration
# =========================

DEFAULT_PROFILE_DIR = Path("browser_profile")

# GPTWeb now writes to a separate folder inside your general outputs folder.
DEFAULT_OUTPUT_DIR = Path("outputs") / "GPTWeb_output"

DEFAULT_INPUT_SELECTOR = os.getenv(
    "GPTWEB_INPUT_SELECTOR",
    "#prompt-textarea, "
    "[data-testid='prompt-textarea'], "
    "div[contenteditable='true'], "
    "[contenteditable='true'], "
    "textarea, "
    "[role='textbox']"
)

DEFAULT_RESPONSE_SELECTOR = os.getenv(
    "GPTWEB_RESPONSE_SELECTOR",
    "[data-message-author-role='assistant'], "
    "article:has([data-message-author-role='assistant']), "
    "[data-testid*='conversation-turn']:has([data-message-author-role='assistant']), "
    ".assistant-message, "
    "[class*='assistant'], "
    ".markdown, "
    "[class*='message']"
)

DEFAULT_SEND_SELECTOR = os.getenv("GPTWEB_SEND_SELECTOR", "")

DEFAULT_READY_SELECTOR = os.getenv(
    "GPTWEB_READY_SELECTOR",
    "#prompt-textarea, "
    "[data-testid='prompt-textarea'], "
    "div[contenteditable='true'], "
    "textarea, "
    "[role='textbox']"
)

DEFAULT_STOP_SELECTORS = [
    "[data-testid='stop-button']",
    "button[aria-label*='Stop']",
    "button:has-text('Stop')",
    "button:has-text('Stop generating')",
    "button:has-text('Stop streaming')",
]

# Default instruction appended to every prompt sent through GPTWeb.
# Set GPTWEB_TEXT_ONLY_SUFFIX="" to disable globally, or pass text_only_suffix=""
# when calling GPTWeb(...).
DEFAULT_TEXT_ONLY_SUFFIX = os.getenv(
    "GPTWEB_TEXT_ONLY_SUFFIX",
    "Please output your reply in plain text, not in the form of a document, so I can copy and paste it."
)


# Ordered response selector candidates.
# The script tries these from most specific to broadest.
RESPONSE_SELECTOR_CANDIDATES = [
    "[data-message-author-role='assistant']",
    "article:has([data-message-author-role='assistant'])",
    "[data-testid*='conversation-turn']:has([data-message-author-role='assistant'])",
    ".assistant-message",
    "[class*='assistant']",
    ".markdown",
    "[class*='message']",
]


# =========================
# Helpers
# =========================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_filename(value: str, max_len: int = 80) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    value = value.strip("._-")
    if not value:
        value = "file"
    return value[:max_len]


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_prompt_file(path: Union[str, Path]) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: Union[str, Path], text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def append_jsonl(path: Union[str, Path], record: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def transcript_from_turns(turns: Sequence[Dict[str, Any]]) -> str:
    parts = []

    for item in turns:
        turn = item.get("turn", "")
        prompt = item.get("prompt", "")
        reply = item.get("reply", "")

        parts.append(
            f"========== TURN {turn} ==========\n\n"
            f"USER:\n{prompt}\n\n"
            f"WEBSITE:\n{reply}\n"
        )

    return "\n\n".join(parts).strip() + "\n"


def append_text_only_suffix(prompt: str, suffix: str = DEFAULT_TEXT_ONLY_SUFFIX) -> str:
    """Append a plain-text-output instruction to a prompt, unless disabled or already present."""
    suffix = (suffix or "").strip()
    if not suffix:
        return prompt

    prompt_clean = prompt.rstrip()
    if suffix.lower() in prompt_clean.lower():
        return prompt_clean

    return f"{prompt_clean}\n\n{suffix}"


def looks_like_cloudflare_challenge(page) -> bool:
    try:
        body_text = page.locator("body").inner_text(timeout=3000).lower()
    except Exception:
        return False

    indicators = [
        "verify you are human",
        "checking your browser",
        "cloudflare",
        "turnstile",
        "cf-challenge",
        "just a moment",
    ]

    return any(x in body_text for x in indicators)


def same_domain(url_a: str, url_b: str) -> bool:
    try:
        host_a = urlparse(url_a).netloc.lower()
        host_b = urlparse(url_b).netloc.lower()
        return host_a and host_a == host_b
    except Exception:
        return False


# =========================
# Config
# =========================

@dataclass
class GPTWebConfig:
    url: str

    profile_dir: Path = DEFAULT_PROFILE_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR

    headless: bool = True

    input_selector: str = DEFAULT_INPUT_SELECTOR
    response_selector: str = DEFAULT_RESPONSE_SELECTOR
    send_selector: str = DEFAULT_SEND_SELECTOR
    ready_selector: str = DEFAULT_READY_SELECTOR

    timeout_ms: int = 60_000

    # Stronger defaults to avoid partial streaming replies.
    reply_stable_checks: int = 6
    reply_check_interval_sec: float = 2.0
    min_reply_wait_sec: float = 8.0
    final_settle_sec: float = 5.0
    max_reply_wait_sec: float = 240.0

    slow_mo_ms: int = 0

    # If provided, attach to an already-open Chrome instance:
    # example: http://localhost:9222
    connect_cdp_url: str = ""


# =========================
# Client
# =========================

class GPTWebClient:
    def __init__(self, config: GPTWebConfig):
        self.config = config

        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

        self.attached_via_cdp = False
        self.turn_counter = 0

    def start(self) -> None:
        ensure_dir(self.config.profile_dir)
        ensure_dir(self.config.output_dir)

        self.playwright = sync_playwright().start()

        if self.config.connect_cdp_url:
            self._start_by_cdp_attach()
            return

        self._start_by_launching_chrome()

    def _start_by_cdp_attach(self) -> None:
        self.attached_via_cdp = True

        self.browser = self.playwright.chromium.connect_over_cdp(
            self.config.connect_cdp_url
        )

        if self.browser.contexts:
            self.context = self.browser.contexts[0]
        else:
            self.context = self.browser.new_context()

        self.page = self._choose_existing_or_new_page()
        self.page.goto(
            self.config.url,
            wait_until="domcontentloaded",
            timeout=self.config.timeout_ms,
        )

        if looks_like_cloudflare_challenge(self.page):
            raise RuntimeError(
                "Cloudflare/human verification challenge detected. "
                "Complete it manually in the attached Chrome window, then rerun."
            )

        self._wait_until_ready()

    def _start_by_launching_chrome(self) -> None:
        self.context = self.playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.config.profile_dir),
            headless=self.config.headless,
            accept_downloads=True,
            slow_mo=self.config.slow_mo_ms,
            channel="chrome",
            args=[
                "--disable-gpu",
                "--disable-software-rasterizer",
            ],
        )

        self.page = self._choose_existing_or_new_page()
        self.page.goto(
            self.config.url,
            wait_until="domcontentloaded",
            timeout=self.config.timeout_ms,
        )

        if looks_like_cloudflare_challenge(self.page):
            raise RuntimeError(
                "Cloudflare/human verification challenge detected. "
                "Do not bypass it. Complete it manually or use CDP attach mode."
            )

        self._wait_until_ready()

    def _choose_existing_or_new_page(self):
        if not self.context:
            raise RuntimeError("Browser context is not available.")

        pages = self.context.pages

        # Prefer the most recently opened real webpage on the target domain.
        # CDP can expose chrome://newtab, omnibox popups, iframes, or older
        # ChatGPT tabs; choosing the first target is fragile.
        for page in reversed(pages):
            try:
                page_url = str(page.url or "")
                if not page_url.startswith(("http://", "https://")):
                    continue
                if same_domain(page_url, self.config.url):
                    return page
            except Exception:
                continue

        # If no matching real page exists, create one instead of attaching to
        # chrome://newtab or an omnibox popup target.
        return self.context.new_page()

    def _wait_until_ready(self) -> None:
        if not self.page:
            raise RuntimeError("Page is not available.")

        if self.config.ready_selector:
            self.page.locator(self.config.ready_selector).first.wait_for(
                timeout=self.config.timeout_ms
            )
        else:
            self.page.locator(self.config.input_selector).first.wait_for(
                timeout=self.config.timeout_ms
            )

    def close(self) -> None:
        # If attached via CDP, do not close the user's already-open Chrome window.
        try:
            if self.context and not self.attached_via_cdp:
                self.context.close()
        except Exception:
            pass
        finally:
            self.context = None
            self.page = None

        try:
            if self.browser and not self.attached_via_cdp:
                self.browser.close()
        except Exception:
            pass
        finally:
            self.browser = None

        try:
            if self.playwright:
                self.playwright.stop()
        except Exception:
            pass
        finally:
            self.playwright = None

    def ask(self, prompt: str) -> str:
        if not self.page:
            raise RuntimeError("Client is not started. Call client.start() first.")

        self.turn_counter += 1
        turn_id = self.turn_counter

        before_responses_count = self._response_count()

        self._submit_prompt(prompt)

        ack_timeout_sec = float(os.getenv("GPTWEB_SUBMISSION_ACK_TIMEOUT_SEC", "30"))
        if not self._wait_for_submission_ack(before_responses_count, timeout_sec=ack_timeout_sec):
            # One retry for the common failure mode where the composer accepted
            # text but the send action did not fire.
            sent = self._click_first_usable_send_button()
            if not sent:
                try:
                    self.page.keyboard.press("Enter")
                except Exception:
                    pass
            if not self._wait_for_submission_ack(before_responses_count, timeout_sec=ack_timeout_sec):
                raise TimeoutError(
                    "Prompt appears not to have been submitted: no stop button, no new assistant response, "
                    "and no response-count increase appeared after submit. Check that the attached Chrome tab "
                    "is logged into ChatGPT and that the correct tab is active."
                )

        reply = self._wait_for_latest_reply(
            before_responses_count=before_responses_count
        )

        record = {
            "turn": turn_id,
            "timestamp_utc": utc_now_iso(),
            "url": self.page.url,
            "prompt": prompt,
            "reply": reply,
            "prompt_sha256": sha256_text(prompt),
            "reply_sha256": sha256_text(reply),
            "attached_via_cdp": self.attached_via_cdp,
            "user_agent": self._safe_user_agent(),
        }

        append_jsonl(self.config.output_dir / "log.jsonl", record)

        return reply

    def _safe_user_agent(self) -> str:
        try:
            return self.page.evaluate("() => navigator.userAgent")
        except Exception:
            return ""

    def _submit_prompt(self, prompt: str) -> None:
        input_locator = self._best_input_locator()
        input_locator.wait_for(timeout=self.config.timeout_ms)

        try:
            # Prefer JS focus to avoid pointer-event interception from ChatGPT's placeholder layer.
            input_locator.evaluate("el => el.focus()")
        except Exception:
            try:
                input_locator.click(timeout=5_000, force=True)
            except Exception:
                pass

        try:
            input_locator.fill(prompt, timeout=10_000)
        except Exception:
            try:
                self.page.keyboard.press("Meta+A")
            except Exception:
                self.page.keyboard.press("Control+A")
            self.page.keyboard.press("Backspace")
            self.page.keyboard.insert_text(prompt)

        sent = self._click_first_usable_send_button()
        if not sent:
            self.page.keyboard.press("Enter")

    def _best_input_locator(self):
        candidates = [
            "div#prompt-textarea[contenteditable='true']",
            "div[data-testid='prompt-textarea'][contenteditable='true']",
            "div[role='textbox'][contenteditable='true']",
            "textarea[data-testid='prompt-textarea']",
            "textarea[name='prompt-textarea']",
            self.config.input_selector,
        ]
        seen = set()
        for selector in candidates:
            if not selector or selector in seen:
                continue
            seen.add(selector)
            try:
                loc = self.page.locator(selector)
                count = loc.count()
                for idx in range(count - 1, -1, -1):
                    item = loc.nth(idx)
                    try:
                        if item.is_visible(timeout=500) and item.is_enabled(timeout=500):
                            return item
                    except Exception:
                        continue
            except Exception:
                continue
        return self.page.locator(self.config.input_selector).last

    def _click_first_usable_send_button(self) -> bool:
        selectors: List[str] = []
        if self.config.send_selector:
            selectors.append(self.config.send_selector)
        selectors.extend([
            "[data-testid='send-button']",
            "button[data-testid='send-button']",
            "button[aria-label='Send prompt']",
            "button[aria-label='Send message']",
            "button[aria-label*='Send']",
        ])
        seen = set()
        for selector in selectors:
            if not selector or selector in seen:
                continue
            seen.add(selector)
            try:
                locator = self.page.locator(selector)
                count = locator.count()
                for idx in range(count - 1, max(-1, count - 6), -1):
                    button = locator.nth(idx)
                    try:
                        if button.is_visible(timeout=500) and button.is_enabled(timeout=500):
                            button.click(timeout=5_000, force=True)
                            return True
                    except Exception:
                        continue
            except Exception:
                continue
        return False

    def _wait_for_submission_ack(self, before_responses_count: int, *, timeout_sec: float) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            try:
                if self._is_generation_active():
                    return True
                if self._response_count() > before_responses_count:
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        return False

    def _response_selectors(self) -> List[str]:
        if self.config.response_selector == DEFAULT_RESPONSE_SELECTOR:
            return RESPONSE_SELECTOR_CANDIDATES

        return [self.config.response_selector]

    def _response_count(self) -> int:
        for selector in self._response_selectors():
            try:
                count = self.page.locator(selector).count()
                if count > 0:
                    return count
            except Exception:
                continue

        return 0

    def _latest_response_text(self) -> str:
        for selector in self._response_selectors():
            try:
                locator = self.page.locator(selector)
                count = locator.count()

                if count <= 0:
                    continue

                text = locator.nth(count - 1).inner_text(timeout=5_000).strip()

                if text:
                    return text

            except Exception:
                continue

        return ""

    def _is_generation_active(self) -> bool:
        for selector in DEFAULT_STOP_SELECTORS:
            try:
                locator = self.page.locator(selector)
                count = locator.count()

                for i in range(count):
                    try:
                        if locator.nth(i).is_visible(timeout=500):
                            return True
                    except Exception:
                        continue

            except Exception:
                continue

        return False

    def _wait_for_latest_reply(self, before_responses_count: int) -> str:
        """
        Wait for the latest assistant response to finish streaming.

        This version is deliberately conservative:
        - waits for text to appear
        - tracks the longest text seen
        - checks common stop-generating indicators
        - requires the text to remain stable
        - performs a final settle read before returning
        """
        start_time = time.time()
        first_text_time: Optional[float] = None

        stable_count = 0
        last_text = ""
        best_text = ""

        while True:
            elapsed = time.time() - start_time

            if elapsed > self.config.max_reply_wait_sec:
                if best_text:
                    return best_text

                raise TimeoutError("Timed out waiting for webpage reply.")

            current_count = self._response_count()
            current_text = self._latest_response_text()

            if current_text and first_text_time is None:
                first_text_time = time.time()

            if current_text and len(current_text) >= len(best_text):
                best_text = current_text

            generation_active = self._is_generation_active()

            if generation_active:
                stable_count = 0
                last_text = current_text
                time.sleep(self.config.reply_check_interval_sec)
                continue

            has_new_response_text = current_count > before_responses_count and bool(current_text)

            if has_new_response_text and current_text:
                if current_text == last_text:
                    stable_count += 1
                else:
                    stable_count = 0

                last_text = current_text

                waited_since_first_text = (
                    time.time() - first_text_time
                    if first_text_time is not None
                    else 0
                )

                if (
                    stable_count >= self.config.reply_stable_checks
                    and waited_since_first_text >= self.config.min_reply_wait_sec
                ):
                    time.sleep(self.config.final_settle_sec)

                    final_text = self._latest_response_text()

                    if final_text and len(final_text) >= len(best_text):
                        if final_text == current_text:
                            return final_text

                        best_text = final_text
                        stable_count = 0
                        last_text = final_text
                        continue

                    return best_text

            time.sleep(self.config.reply_check_interval_sec)


# =========================
# Public callable
# =========================

def GPTWeb(
    input_data: Union[str, Sequence[Union[str, Path]]],
    output_path: Optional[Union[str, Path]] = None,
    *,
    url: Optional[str] = None,
    headless: bool = True,
    profile_dir: Union[str, Path] = DEFAULT_PROFILE_DIR,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    input_selector: str = DEFAULT_INPUT_SELECTOR,
    response_selector: str = DEFAULT_RESPONSE_SELECTOR,
    send_selector: str = DEFAULT_SEND_SELECTOR,
    ready_selector: str = DEFAULT_READY_SELECTOR,
    auto_setup: bool = True,
    connect_cdp_url: str = "",
    text_only_suffix: str = DEFAULT_TEXT_ONLY_SUFFIX,
) -> Union[str, List[Dict[str, Any]]]:

    target_url = url or os.getenv("GPTWEB_URL")

    if not target_url:
        raise ValueError("Missing target URL. Pass url='https://...' or set GPTWEB_URL.")

    config = GPTWebConfig(
        url=target_url,
        profile_dir=Path(profile_dir),
        output_dir=Path(output_dir),
        headless=headless,
        input_selector=input_selector,
        response_selector=response_selector,
        send_selector=send_selector,
        ready_selector=ready_selector,
        connect_cdp_url=connect_cdp_url,
    )

    client = GPTWebClient(config)

    try:
        client.start()

    except Exception as first_error:
        try:
            client.close()
        except Exception:
            pass

        if connect_cdp_url:
            raise RuntimeError(
                "Could not attach to the existing Chrome window through CDP. "
                "Check that Chrome was started with --remote-debugging-port=9222 "
                "and verify with: curl http://localhost:9222/json/version"
            ) from first_error

        if not auto_setup:
            raise RuntimeError(
                "GPTWeb could not start. The saved login session may be missing, "
                "expired, or the selectors may be wrong."
            ) from first_error

        print("\nGPTWeb could not start with the saved session.")
        print("Opening visible browser for manual login/setup...\n")

        setup_login(
            target_url,
            profile_dir=profile_dir,
            input_selector=input_selector,
        )

        print("\nRetrying GPTWeb with the saved session...\n")

        client = GPTWebClient(config)

        try:
            client.start()

        except Exception as second_error:
            try:
                client.close()
            except Exception:
                pass

            raise RuntimeError(
                "Auto-setup completed, but GPTWeb still could not detect the chat UI. "
                "Most likely causes: wrong selectors, login did not finish, "
                "or the website requires CAPTCHA/2FA again."
            ) from second_error

    try:
        if isinstance(input_data, str):
            prompt = append_text_only_suffix(input_data, text_only_suffix)
            reply = client.ask(prompt)

            if output_path:
                text = f"USER:\n{prompt}\n\nWEBSITE:\n{reply}\n"
                write_text(output_path, text)

            return reply

        if isinstance(input_data, Sequence):
            turns: List[Dict[str, Any]] = []

            for idx, file_path in enumerate(input_data, start=1):
                prompt = append_text_only_suffix(read_prompt_file(file_path), text_only_suffix)
                reply = client.ask(prompt)

                turns.append(
                    {
                        "turn": idx,
                        "prompt_file": str(file_path),
                        "prompt": prompt,
                        "reply": reply,
                        "timestamp_utc": utc_now_iso(),
                        "prompt_sha256": sha256_text(prompt),
                        "reply_sha256": sha256_text(reply),
                    }
                )

                if output_path:
                    write_text(output_path, transcript_from_turns(turns))

            return turns

        raise TypeError("input_data must be a string prompt or a list of prompt file paths.")

    finally:
        client.close()


# =========================
# Setup login
# =========================

def setup_login(
    url: str,
    *,
    profile_dir: Union[str, Path] = DEFAULT_PROFILE_DIR,
    input_selector: str = DEFAULT_INPUT_SELECTOR,
) -> None:
    profile_dir = Path(profile_dir)
    ensure_dir(profile_dir)

    print("\nOpening visible browser for manual login...")
    print(f"Profile directory: {profile_dir.resolve()}")
    print("Log in normally in the browser window.")
    print("When finished, return here and press Enter.\n")

    pw = sync_playwright().start()

    context = pw.chromium.launch_persistent_context(
        user_data_dir=str(profile_dir),
        headless=False,
        accept_downloads=True,
        channel="chrome",
        args=[
            "--disable-gpu",
            "--disable-software-rasterizer",
        ],
    )

    try:
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)

        input("After login is complete, press Enter here to save session and close browser... ")

        try:
            page.locator(input_selector).first.wait_for(timeout=5_000)
            print("Detected chat input. Session likely ready.")
        except PlaywrightTimeoutError:
            print("Could not detect chat input, but session was still saved.")

    finally:
        context.close()
        pw.stop()

    print("\nLogin setup complete.\n")


# =========================
# CLI
# =========================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automate a ChatGPT-like webpage UI through Playwright."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser(
        "setup",
        help="Open visible browser for manual login.",
    )
    setup.add_argument("--url", required=True)
    setup.add_argument("--profile-dir", default=str(DEFAULT_PROFILE_DIR))
    setup.add_argument("--input-selector", default=DEFAULT_INPUT_SELECTOR)

    ask = subparsers.add_parser(
        "ask",
        help="Ask one direct prompt.",
    )
    ask.add_argument("--url", required=True)
    ask.add_argument("--prompt", required=True)
    ask.add_argument("--output", default=None)
    ask.add_argument("--profile-dir", default=str(DEFAULT_PROFILE_DIR))
    ask.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ask.add_argument("--headed", action="store_true", help="Run with visible browser.")
    ask.add_argument("--input-selector", default=DEFAULT_INPUT_SELECTOR)
    ask.add_argument("--response-selector", default=DEFAULT_RESPONSE_SELECTOR)
    ask.add_argument("--send-selector", default=DEFAULT_SEND_SELECTOR)
    ask.add_argument("--ready-selector", default=DEFAULT_READY_SELECTOR)
    ask.add_argument("--connect-cdp-url", default="")
    ask.add_argument("--text-only-suffix", default=DEFAULT_TEXT_ONLY_SUFFIX)

    run_files = subparsers.add_parser(
        "run-files",
        help="Run multiple prompt files.",
    )
    run_files.add_argument("--url", required=True)
    run_files.add_argument("--files", nargs="+", required=True)
    run_files.add_argument("--output", required=True)
    run_files.add_argument("--profile-dir", default=str(DEFAULT_PROFILE_DIR))
    run_files.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run_files.add_argument("--headed", action="store_true", help="Run with visible browser.")
    run_files.add_argument("--input-selector", default=DEFAULT_INPUT_SELECTOR)
    run_files.add_argument("--response-selector", default=DEFAULT_RESPONSE_SELECTOR)
    run_files.add_argument("--send-selector", default=DEFAULT_SEND_SELECTOR)
    run_files.add_argument("--ready-selector", default=DEFAULT_READY_SELECTOR)
    run_files.add_argument("--connect-cdp-url", default="")
    run_files.add_argument("--text-only-suffix", default=DEFAULT_TEXT_ONLY_SUFFIX)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "setup":
            setup_login(
                args.url,
                profile_dir=args.profile_dir,
                input_selector=args.input_selector,
            )
            return 0

        if args.command == "ask":
            reply = GPTWeb(
                args.prompt,
                output_path=args.output,
                url=args.url,
                headless=not args.headed,
                profile_dir=args.profile_dir,
                output_dir=args.output_dir,
                input_selector=args.input_selector,
                response_selector=args.response_selector,
                send_selector=args.send_selector,
                ready_selector=args.ready_selector,
                connect_cdp_url=args.connect_cdp_url,
                text_only_suffix=args.text_only_suffix,
            )

            print(reply)
            return 0

        if args.command == "run-files":
            turns = GPTWeb(
                args.files,
                output_path=args.output,
                url=args.url,
                headless=not args.headed,
                profile_dir=args.profile_dir,
                output_dir=args.output_dir,
                input_selector=args.input_selector,
                response_selector=args.response_selector,
                send_selector=args.send_selector,
                ready_selector=args.ready_selector,
                connect_cdp_url=args.connect_cdp_url,
                text_only_suffix=args.text_only_suffix,
            )

            print(f"Completed {len(turns)} turns.")
            print(f"Transcript saved to: {args.output}")
            print(f"GPTWeb log saved under: {args.output_dir}")
            return 0

        parser.print_help()
        return 2

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())