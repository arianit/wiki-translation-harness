"""Claude Code CLI engine: runs `claude -p` non-interactively under the
caller's existing Claude Code subscription/login. No API key used or
required.

The subprocess-invocation logic (temp-file system prompt, stdin user
prompt, stream-json parsing, the truncation heuristic) is ported from
multimodeltranslationpipeline's mmtp/claude_cli.py — a proven pattern
already running in production for a sibling translation pipeline. Ported
rather than imported: these are independently-deployable projects, not a
shared library.

Exposes ClaudeCodeClient, satisfying the same duck-typed contract
OpenRouterClient does (see engines.LLMEngineClient) so translator.py's
translate_chunk()/repair.py's repair_chunk() need no changes at all to use
either engine.
"""
from __future__ import annotations

import asyncio
import json
import random
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wiki_translation_harness.models import EngineError, InsufficientCreditsError, ModelPricing
from wiki_translation_harness.openrouter import RetryCallback

# See mmtp/claude_cli.py's module docstring for the full incident writeup:
# plain `--output-format json`'s `result` field only reflects the LAST
# internal turn a generation needed, silently dropping every earlier turn's
# text even though it was billed. --output-format stream-json (+ --verbose,
# mandatory alongside it in print mode) instead emits one event per content
# block as it's produced; concatenating every `assistant` event's text
# blocks, in order, reconstructs the true complete output regardless of how
# many internal turns it took.
#
# stop_reason == "max_tokens" (the model ran out of its output budget
# mid-response) is the *authoritative* truncation signal, straight from the
# API -- checked first, below. The chars-per-token ratio is only a
# secondary safety net for a different failure mode entirely: our own
# stream-json parsing dropping/mangling a line so the reconstructed text is
# short even though the model finished normally (stop_reason: end_turn).
# mmtp's original 2.0 threshold (calibrated without much extended thinking
# in the mix) turned out too strict for this use case: a live reproduction
# against real Albanian wikitext output found a confirmed-genuine,
# naturally-completed response (stop_reason: end_turn) at a 1.94 ratio,
# while confirmed-truncated responses measured 0.24-1.14 (see mmtp's own
# calibration notes) down to as low as 0.31-0.59 reproduced directly against
# this project's own real failures before the stop_reason check existed.
# 1.0 sits with real margin below the former and above the latter.
_MIN_CHARS_PER_OUTPUT_TOKEN = 1.0
_TRUNCATION_CHECK_MIN_OUTPUT_TOKENS = 2000


class ClaudeCodeError(EngineError):
    pass


class ClaudeCodeSessionLimitError(ClaudeCodeError, InsufficientCreditsError):
    """An account/org-level Claude Code rate or spend limit (HTTP 429) --
    confirmed live: "You've hit your monthly spend limit", the 5-hour
    session rate window, overage rejected. Raised as an
    InsufficientCreditsError (not just a plain ClaudeCodeError) specifically
    so pipeline.py's ensure_fallback_engine() offers/auto-switches to
    config.fallback_provider (opencode_go by default -- see
    default_model_for_provider and ensure_fallback_engine's `target`
    selection) instead of just failing every remaining chunk against an
    engine that's guaranteed to keep rejecting them. Still a ClaudeCodeError
    too, so existing `except ClaudeCodeError` / `pytest.raises(ClaudeCodeError)`
    call sites keep working unchanged."""


@dataclass
class ClaudeCLIResult:
    is_error: bool
    result_text: str = ""
    total_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0
    model_used: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    stderr: str = ""


def _write_diagnostic_log(
    log_dir: Path | str | None, *, model: str, note: str, stdout: str, stderr: str
) -> str | None:
    """Best-effort dump of a failed call's raw stream + stderr, so a bad
    call can be diagnosed after the fact instead of needing a live,
    real-money reproduction every time — see claude_code_client.py's git
    history for exactly that cost, twice, before this existed. Never
    raises — a logging failure must not break the actual call."""
    if log_dir is None:
        return None
    try:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = log_dir / f"claude-cli-error-{timestamp}-{model}-{uuid.uuid4().hex[:8]}.log"
        log_path.write_text(
            f"note: {note}\nmodel: {model}\n\n--- stdout (raw stream-json) ---\n{stdout}\n\n"
            f"--- stderr ---\n{stderr}\n",
            encoding="utf-8",
        )
        return str(log_path)
    except OSError:
        return None


def _as_text(output: bytes | str | None) -> str:
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output or ""


def run_claude_cli(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    cli_path: str = "claude",
    permission_mode: str = "bypassPermissions",
    log_dir: Path | str | None = None,
    timeout_s: float = 600.0,
    effort: str | None = None,
) -> ClaudeCLIResult:
    """Single-shot, tool-less, blocking call — see ClaudeCodeClient.chat_completion
    for why this runs off the event loop via asyncio.to_thread rather than
    being called directly from async code.

    --tools "" disables all tools (no accidental file writes, no live
    fetching, no tool-round-trip latency — the chunk's wikitext is already
    embedded in user_prompt, there is nothing to fetch live). System prompt
    goes to a temp file (--system-prompt-file) and the user prompt goes over
    stdin, since full articles plus skill content can exceed the OS argv
    size limit.

    effort (low/medium/high/xhigh/max, confirmed via `claude -p --help`)
    maps to the same thinking-depth/token-spend knob as the Messages API's
    output_config.effort -- see models.py's Config.claude_code_effort.
    None omits the flag, letting the CLI use its own default."""
    cmd = [
        cli_path,
        "-p",
        "--model",
        model,
        "--tools",
        "",
        "--permission-mode",
        permission_mode,
        "--output-format",
        "stream-json",
        "--verbose",
        "--no-session-persistence",
    ]
    if effort:
        cmd += ["--effort", effort]

    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", prefix="wth-system-prompt-", delete=False
        ) as f:
            f.write(system_prompt)
            system_prompt_path = f.name
        try:
            proc = subprocess.run(
                [*cmd, "--system-prompt-file", system_prompt_path],
                input=user_prompt,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        finally:
            Path(system_prompt_path).unlink(missing_ok=True)
    except subprocess.TimeoutExpired as exc:
        log_path = _write_diagnostic_log(
            log_dir, model=model, note=f"timed out after {timeout_s}s",
            # TimeoutExpired carries bytes even when run with text=True.
            stdout=_as_text(exc.stdout), stderr=_as_text(exc.stderr),
        )
        stderr_msg = f"claude CLI call timed out after {timeout_s}s: {exc}"
        if log_path:
            stderr_msg += f" (raw output logged to {log_path})"
        return ClaudeCLIResult(is_error=True, model_used=model, stderr=stderr_msg)
    except FileNotFoundError as exc:
        return ClaudeCLIResult(
            is_error=True,
            model_used=model,
            stderr=f"claude CLI not found ({cli_path!r}): {exc}",
        )

    text_blocks: list[str] = []
    result_event: dict | None = None
    saw_any_event = False
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        saw_any_event = True
        event_type = event.get("type")
        if event_type == "assistant":
            for block in (event.get("message") or {}).get("content") or []:
                if block.get("type") == "text" and block.get("text"):
                    text_blocks.append(block["text"])
        elif event_type == "result":
            result_event = event

    if not saw_any_event or result_event is None:
        log_path = _write_diagnostic_log(
            log_dir, model=model,
            note="no parseable stream-json events, or stream never reached a `result` event",
            stdout=proc.stdout, stderr=proc.stderr,
        )
        stderr_msg = (proc.stderr or proc.stdout or "no output")[:2000]
        if log_path:
            stderr_msg += f" (raw output logged to {log_path})"
        return ClaudeCLIResult(is_error=True, model_used=model, stderr=stderr_msg)

    data = result_event
    reassembled_text = "".join(text_blocks)
    # Fall back to the (possibly turn-truncated) top-level `result` field
    # only if no streamed text blocks were captured at all.
    result_text = reassembled_text or (data.get("result") or "")

    usage = data.get("usage") or {}
    output_tokens = usage.get("output_tokens", 0)
    # Prompt caching means most of a call's real input (here: the ~70KB
    # skill text in the system prompt) lands in cache_creation_input_tokens
    # (first time) or cache_read_input_tokens (cache hit on a later call),
    # not input_tokens -- confirmed live: input_tokens alone reported as 2
    # while cache_creation_input_tokens was 30278 for the same call. Only
    # counting input_tokens made the caller's output/input ratio safety
    # check (translation-harness's max_token_ratio) see ~4882:1 and reject
    # a completely normal call as a suspected runaway generation. All three
    # figures reflect tokens that were genuinely part of the input context,
    # cached or not, so all three belong in what gets reported back.
    input_tokens = (
        usage.get("input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0)
        + usage.get("cache_read_input_tokens", 0)
    )
    cost = data.get("total_cost_usd") or 0.0
    duration_ms = data.get("duration_ms", 0)

    # Extended-thinking tokens (usage.output_tokens_details.thinking_tokens)
    # count toward output_tokens but produce zero characters in the text
    # stream -- confirmed live: a genuinely complete, well-formed 9515
    # output-token response (is_error: false, stop_reason: end_turn) was
    # 7765 of those tokens thinking, leaving ~1750 tokens' worth of real
    # text -- comparing chars-recovered against the full output_tokens
    # figure (mmtp's original heuristic, ported from a pipeline that
    # apparently doesn't see this much extended thinking) flagged it as
    # truncated even though nothing was lost. Only the non-thinking tokens
    # could ever have produced visible text, so only those belong in the
    # ratio check.
    thinking_tokens = (usage.get("output_tokens_details") or {}).get("thinking_tokens", 0)
    visible_output_tokens = max(output_tokens - thinking_tokens, 0)

    stop_reason = data.get("stop_reason")
    if stop_reason == "max_tokens":
        log_path = _write_diagnostic_log(
            log_dir, model=model, note="stop_reason=max_tokens",
            stdout=proc.stdout, stderr=proc.stderr,
        )
        stderr_msg = (
            f"Truncated: the model hit its output token budget mid-response "
            f"(stop_reason=max_tokens, {output_tokens} output tokens, "
            f"{thinking_tokens} of them thinking)."
        )
        if log_path:
            stderr_msg += f" (raw output logged to {log_path})"
        return ClaudeCLIResult(
            is_error=True, total_cost_usd=cost, input_tokens=input_tokens,
            output_tokens=output_tokens, duration_ms=duration_ms, model_used=model,
            raw=data, stderr=stderr_msg,
        )

    if (
        visible_output_tokens >= _TRUNCATION_CHECK_MIN_OUTPUT_TOKENS
        and len(result_text) < visible_output_tokens * _MIN_CHARS_PER_OUTPUT_TOKEN
    ):
        log_path = _write_diagnostic_log(
            log_dir, model=model,
            note=f"suspected content loss: {len(result_text)} chars for {visible_output_tokens} visible tokens",
            stdout=proc.stdout, stderr=proc.stderr,
        )
        stderr_msg = (
            f"Suspected content loss during stream reconstruction: model finished "
            f"normally (stop_reason={stop_reason!r}) with {visible_output_tokens} "
            f"non-thinking output tokens ({output_tokens} total, {thinking_tokens} "
            f"thinking), but only {len(result_text)} chars of text were recovered "
            "from the stream."
        )
        if log_path:
            stderr_msg += f" (raw output logged to {log_path})"
        return ClaudeCLIResult(
            is_error=True, total_cost_usd=cost, input_tokens=input_tokens,
            output_tokens=output_tokens, duration_ms=duration_ms, model_used=model,
            raw=data, stderr=stderr_msg,
        )

    cli_reported_error = bool(data.get("is_error"))
    if cli_reported_error:
        log_path = _write_diagnostic_log(
            log_dir, model=model, note="claude CLI reported is_error=true",
            stdout=proc.stdout, stderr=proc.stderr,
        )
        # A CLI-level error (rate limits, spend limits, API errors) is
        # often explained in the result event's own `result` text, not on
        # the process's stderr stream, which can be completely empty --
        # confirmed live: a "You've hit your monthly spend limit" rejection
        # had empty proc.stderr, surfacing upstream as a bare "unknown
        # error" with no way to tell a real account-level block apart from
        # anything else without reading the diagnostic log by hand.
        error_detail = proc.stderr or data.get("result") or "unknown error"
        proc_stderr_with_log = error_detail
        if log_path:
            proc_stderr_with_log += f" (raw output logged to {log_path})"
    else:
        proc_stderr_with_log = proc.stderr or ""

    return ClaudeCLIResult(
        is_error=cli_reported_error,
        result_text=result_text,
        total_cost_usd=cost,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_ms=duration_ms,
        model_used=model,
        raw=data,
        stderr=proc_stderr_with_log,
    )


class ClaudeCodeClient:
    def __init__(
        self,
        model: str,
        cli_path: str = "claude",
        permission_mode: str = "bypassPermissions",
        timeout_s: float = 600.0,
        max_retries: int = 5,
        log_dir: Path | str | None = None,
        effort: str | None = None,
    ):
        self.model = model
        self.cli_path = cli_path
        self.permission_mode = permission_mode
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.log_dir = log_dir
        self.effort = effort

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        on_retry: RetryCallback | None = None,
        usage_out: dict | None = None,
    ) -> tuple[str, int, int]:
        """Returns (text, prompt_tokens, completion_tokens). temperature is
        ignored — the CLI has no equivalent sampling-temperature flag.
        usage_out is accepted for interface parity with OpenRouterClient
        (see engines.LLMEngineClient) but left unfilled -- the CLI has no
        provider-reported inline cost to surface through it.

        messages is always exactly [{"role":"system",...},{"role":"user",...}]
        — the fixed shape skill_loader.build_translation_messages/
        build_repair_messages produce."""
        system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_prompt = next((m["content"] for m in messages if m["role"] == "user"), None)
        if user_prompt is None:
            raise ClaudeCodeError(f"chat_completion got no user-role message: {messages!r}")

        attempt = 0
        while True:
            # subprocess.run() blocks; run it off the event loop so
            # concurrent chunk translations (translate_chunk's slot-queue
            # workers) don't get serialized behind one call.
            result = await asyncio.to_thread(
                run_claude_cli,
                system_prompt,
                user_prompt,
                model=model,
                cli_path=self.cli_path,
                permission_mode=self.permission_mode,
                timeout_s=self.timeout_s,
                log_dir=self.log_dir,
                effort=self.effort,
            )
            if not result.is_error:
                return result.result_text, result.input_tokens, result.output_tokens

            attempt += 1
            if "claude CLI not found" in result.stderr:
                # A missing binary won't fix itself on retry.
                raise ClaudeCodeError(result.stderr)
            if result.raw.get("api_error_status") == 429:
                # An account/org-level rate or spend limit (confirmed live:
                # "You've hit your monthly spend limit", five_hour rate
                # window, overage rejected) -- unlike a transient service
                # hiccup, backing off and retrying within the same run
                # cannot succeed, it just burns the retry budget's wall
                # time (up to ~60s per attempt) for a guaranteed-identical
                # rejection every time, observed across many chunks in a
                # row.
                raise ClaudeCodeSessionLimitError(f"rate/spend limit (HTTP 429): {result.stderr}")
            if attempt > self.max_retries:
                raise ClaudeCodeError(
                    f"claude CLI call failed after {attempt} attempts: {result.stderr}"
                )
            await self._backoff(attempt, on_retry, reason=result.stderr or "unknown error")

    async def _backoff(self, attempt: int, on_retry: RetryCallback | None, reason: str) -> None:
        delay = min(2 ** (attempt - 1), 60) + random.uniform(0, 1)
        if on_retry is not None:
            on_retry(attempt, reason, delay)
        await asyncio.sleep(delay)

    async def get_pricing_for(self, model: str) -> ModelPricing | None:
        # No pricing table — the CLI's own result carries a real
        # total_cost_usd per call, but run_completion()'s cost model
        # (openrouter.compute_cost) multiplies an external per-token price
        # table instead of reading a provider-reported total, so cost
        # reports as $0 here, same as the existing "local" provider already
        # does. A future improvement could thread the CLI's own per-call
        # cost through TranslationResult instead of relying on this.
        return None

    async def fetch_pricing(self) -> dict[str, ModelPricing]:
        return {}

    async def aclose(self) -> None:
        pass  # no persistent connection to close
