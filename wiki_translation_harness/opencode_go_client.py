"""OpenCode Go engine (github.com/sst/opencode's Go CLI): runs `opencode run
--format json` non-interactively, under whichever model/provider the
caller's own opencode config already has authenticated.

This exists specifically as the harness's default fallback target when
`provider: claude_code` hits its own account-level session/rate limit (see
claude_code_client.py's ClaudeCodeSessionLimitError and pipeline.py's
ensure_fallback_engine) — opencode is a separate binary with its own,
independent auth/session, so it isn't affected by Claude Code hitting its
cap. It works standalone too via `--provider opencode_go`.

The wire contract below was confirmed directly against a real, installed
`opencode` v1.18.31 binary (not assumed from docs):

- `opencode run [message..] --format json` with no positional message reads
  the message from stdin instead (confirmed: an empty stdin fails fast with
  "You must provide a message or a command", exit 1 -- it does not hang
  waiting on a TTY). This is what lets the combined system+user prompt go
  over stdin exactly like claude_code_client.run_claude_cli, sidestepping
  the OS argv size limit a full article plus skill text could hit.
- stdout is newline-delimited JSON events, one per line, at minimum
  `step_start` -> `text` (one full block per turn, not incremental deltas,
  in every case observed) -> `step_finish`. `step_finish`'s `part.tokens`
  carries `{total, input, output, reasoning, cache: {write, read}}` and
  `part.cost` a real per-call cost -- both fed into TranslationResult via
  chat_completion's usage_out the same way openrouter.py's Experiential
  Labs branch already does (see run_completion's `reported_cost`), rather
  than the $0-always some other CLI-based engines here report.
- A failure emits a `{"type":"error","error":{"name":...,"data":{"message":
  ...}}}` event **on stdout**, not stderr (confirmed: stderr was empty on a
  bad-model rejection), alongside a non-zero exit code. The default
  (non-JSON) format's own error rendering is stderr instead, so the client
  below always requests --format json and never has to guess which stream
  a diagnostic landed on.

One important caveat this session's live check surfaced, which callers
should know about: `opencode run`'s default "build" agent, on a normal
opencode install, can have full bash/file/network tool permissions --
unlike claude_code_client.py's `claude -p --tools ""`, there is no
`--tools ""` equivalent on `run` itself. Since chunk text here is untrusted
external wiki content, config.opencode_go_agent lets a caller pin `--agent`
to a tools-locked-down agent name (see models.Config's opencode_go_agent
docstring for the config snippet that defines one) -- left unset, this
engine inherits whatever tool permissions opencode's own config already
grants its default agent, same as it would for a human running `opencode
run` by hand.

Exposes OpenCodeGoClient, satisfying the same duck-typed contract
OpenRouterClient/ClaudeCodeClient do (see engines.LLMEngineClient) so
translator.py's translate_chunk()/repair.py's repair_chunk() need no changes
to use it.
"""
from __future__ import annotations

import asyncio
import json
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wiki_translation_harness.models import (
    EngineError,
    InsufficientCreditsError,
    ModelPricing,
    looks_like_usage_limit,
)
from wiki_translation_harness.openrouter import RetryCallback


class OpenCodeGoError(EngineError):
    pass


class OpenCodeGoSessionLimitError(OpenCodeGoError, InsufficientCreditsError):
    """opencode's own configured provider rejected the call for a
    rate/usage/spend-limit reason. Raised as an InsufficientCreditsError (not
    just a plain OpenCodeGoError) for interface parity with
    claude_code_client.ClaudeCodeSessionLimitError -- retrying the same
    engine can't succeed, so pipeline.py's ensure_fallback_engine() would
    otherwise be the right response, though since opencode_go is already
    used as the harness's fallback-of-last-resort, in practice this just
    fails the chunk fast instead of burning the retry budget."""


@dataclass
class OpenCodeCLIResult:
    is_error: bool
    result_text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    model_used: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    stderr: str = ""


def run_opencode_cli(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    cli_path: str = "opencode",
    agent: str | None = None,
    timeout_s: float = 600.0,
) -> OpenCodeCLIResult:
    """Single-shot, blocking call -- see OpenCodeGoClient.chat_completion for
    why this runs off the event loop via asyncio.to_thread rather than being
    called directly from async code.

    model == "auto" (config.default_model_for_provider's pick for this
    provider) omits --model entirely, letting `opencode` use its own
    configured default provider/model instead of a guessed id that opencode
    might not even be authenticated for.

    The system and user prompt are combined into a single stdin payload
    (opencode's `run` has no separate system-prompt flag) rather than passed
    as a positional argument, since full articles plus skill content can
    exceed the OS argv size limit -- same reasoning as
    claude_code_client.run_claude_cli, and confirmed live that `run` reads
    stdin as the message when none is given positionally."""
    cmd = [cli_path, "run", "--format", "json"]
    if model != "auto":
        cmd += ["--model", model]
    if agent:
        cmd += ["--agent", agent]

    combined_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt

    try:
        proc = subprocess.run(
            cmd,
            input=combined_prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return OpenCodeCLIResult(
            is_error=True,
            model_used=model,
            stderr=f"opencode CLI call timed out after {timeout_s}s: {exc}",
        )
    except FileNotFoundError as exc:
        return OpenCodeCLIResult(
            is_error=True,
            model_used=model,
            stderr=f"opencode CLI not found ({cli_path!r}): {exc}",
        )

    text_blocks: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    cost_usd: float | None = None
    error_message: str | None = None
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
        part = event.get("part") or {}
        if event_type == "text" and part.get("text"):
            text_blocks.append(part["text"])
        elif event_type == "step_finish":
            tokens = part.get("tokens") or {}
            cache = tokens.get("cache") or {}
            # Mirrors claude_code_client's cache-token accounting: opencode
            # also reports cache reads/writes separately from `input`, and
            # all three genuinely were part of that step's input context.
            prompt_tokens += (
                tokens.get("input", 0) + cache.get("write", 0) + cache.get("read", 0)
            )
            completion_tokens += tokens.get("output", 0)
            step_cost = part.get("cost")
            if step_cost is not None:
                cost_usd = (cost_usd or 0.0) + float(step_cost)
        elif event_type == "error":
            error = event.get("error") or {}
            error_message = (
                (error.get("data") or {}).get("message")
                or error.get("name")
                or json.dumps(error)
            )

    if proc.returncode != 0 or error_message is not None:
        stderr_msg = (
            error_message
            or (proc.stderr or proc.stdout or "unknown error").strip()
        )[:2000]
        return OpenCodeCLIResult(
            is_error=True,
            model_used=model,
            stderr=stderr_msg or "opencode CLI exited non-zero with no output",
            raw={"returncode": proc.returncode},
        )

    if not saw_any_event:
        stderr_msg = (proc.stderr or proc.stdout or "no parseable JSON events").strip()[:2000]
        return OpenCodeCLIResult(is_error=True, model_used=model, stderr=stderr_msg)

    return OpenCodeCLIResult(
        is_error=False,
        result_text="".join(text_blocks).strip(),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        model_used=model,
        stderr=proc.stderr or "",
    )


class OpenCodeGoClient:
    def __init__(
        self,
        model: str,
        cli_path: str = "opencode",
        agent: str | None = None,
        timeout_s: float = 600.0,
        max_retries: int = 5,
        log_dir: Path | str | None = None,
    ):
        self.model = model
        self.cli_path = cli_path
        self.agent = agent
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        # Accepted for interface parity with ClaudeCodeClient (both are
        # constructed generically by engines.build_llm_client) but unused --
        # run_opencode_cli has no per-call diagnostic dump the way
        # claude_code_client._write_diagnostic_log does; a failed call's
        # full JSON stream is short enough to just live in the raised
        # exception's message.
        self.log_dir = log_dir

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        on_retry: RetryCallback | None = None,
        usage_out: dict | None = None,
    ) -> tuple[str, int, int]:
        """Returns (text, prompt_tokens, completion_tokens). temperature is
        ignored (no CLI equivalent). usage_out, when passed, is filled with
        {"cost": <float>} whenever opencode reports one for the call --
        openrouter.run_completion (used for every engine, not just
        OpenRouterClient) prefers this reported cost over its external
        per-token pricing-table estimate, the same path Experiential Labs
        already uses.

        messages is always exactly [{"role":"system",...},{"role":"user",...}]
        -- the fixed shape skill_loader.build_translation_messages/
        build_repair_messages produce."""
        system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_prompt = next((m["content"] for m in messages if m["role"] == "user"), None)
        if user_prompt is None:
            raise OpenCodeGoError(f"chat_completion got no user-role message: {messages!r}")

        attempt = 0
        while True:
            # subprocess.run() blocks; run it off the event loop so
            # concurrent chunk translations (translate_chunk's slot-queue
            # workers) don't get serialized behind one call.
            result = await asyncio.to_thread(
                run_opencode_cli,
                system_prompt,
                user_prompt,
                model=model,
                cli_path=self.cli_path,
                agent=self.agent,
                timeout_s=self.timeout_s,
            )
            if not result.is_error:
                if usage_out is not None and result.cost_usd is not None:
                    usage_out["cost"] = result.cost_usd
                return result.result_text, result.prompt_tokens, result.completion_tokens

            attempt += 1
            if "opencode CLI not found" in result.stderr:
                # A missing binary won't fix itself on retry.
                raise OpenCodeGoError(result.stderr)
            if looks_like_usage_limit(result.stderr):
                # Same reasoning as claude_code_client's 429 handling: an
                # account/session-level limit cannot be fixed by backing off
                # within the same run, so fail fast rather than burn the
                # retry budget's wall time on a guaranteed-identical
                # rejection every time.
                raise OpenCodeGoSessionLimitError(
                    f"opencode session/usage limit: {result.stderr}"
                )
            if attempt > self.max_retries:
                raise OpenCodeGoError(
                    f"opencode CLI call failed after {attempt} attempts: {result.stderr}"
                )
            await self._backoff(attempt, on_retry, reason=result.stderr or "unknown error")

    async def _backoff(self, attempt: int, on_retry: RetryCallback | None, reason: str) -> None:
        delay = min(2 ** (attempt - 1), 60) + random.uniform(0, 1)
        if on_retry is not None:
            on_retry(attempt, reason, delay)
        await asyncio.sleep(delay)

    async def get_pricing_for(self, model: str) -> ModelPricing | None:
        # No pricing table -- real per-call cost, when opencode reports one,
        # reaches TranslationResult through chat_completion's usage_out
        # instead (see run_completion's reported_cost preference), same
        # mechanism Experiential Labs uses. Zero-cost calls (a free
        # opencode/* model, or a step that reports no cost field) still
        # report $0, same as this returning None always would.
        return None

    async def fetch_pricing(self) -> dict[str, ModelPricing]:
        return {}

    async def aclose(self) -> None:
        pass  # no persistent connection to close
