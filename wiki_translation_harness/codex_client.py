"""Subscription-authenticated Codex CLI translation engine."""
from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile

from wiki_translation_harness.models import EngineError, InsufficientCreditsError


class CodexError(EngineError):
    pass


def run_codex_cli(cli_path: str, model: str, prompt: str, timeout_s: float):
    # Do not inherit a configured API provider, tools, MCP servers or project
    # instructions. Auth remains the CLI's existing ChatGPT login.
    common = [cli_path, "-c", 'forced_login_method="chatgpt"']
    try:
        status = subprocess.run(
            common + ["login", "status"], capture_output=True, text=True, timeout=15,
        )
        if status.returncode or "logged in using chatgpt" not in (status.stdout + status.stderr).lower():
            raise CodexError("Codex requires ChatGPT subscription authentication. Run `codex login`.")
        with tempfile.TemporaryDirectory(prefix="wiki-codex-") as workdir:
            proc = subprocess.run(
                common + ["exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check", "--json",
                          "--sandbox", "read-only", "--model", model,
                          "-c", 'model_provider="openai"',
                          "-c", 'model_reasoning_effort="high"',
                          "-c", 'service_tier="default"',
                          "-c", 'features.shell_tool=false',
                          "-c", 'features.multi_agent=false',
                          "-C", workdir, "-"],
                input=prompt, capture_output=True, text=True, timeout=timeout_s,
            )
    except FileNotFoundError as exc:
        raise CodexError(f"Codex CLI not found: {cli_path}") from exc
    except subprocess.TimeoutExpired as exc:
        raise CodexError(f"Codex CLI timed out after {timeout_s}s") from exc

    messages = []
    errors = []
    usage = {}
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "item.completed" and event.get("item", {}).get("type") == "agent_message":
            messages.append(event["item"].get("text", ""))
        elif kind == "turn.completed":
            usage = event.get("usage") or {}
        elif kind in ("error", "turn.failed"):
            errors.append(event.get("message") or (event.get("error") or {}).get("message") or str(event))
    if proc.returncode or errors:
        detail = "\n".join(errors) or proc.stderr[-2000:] or "Codex exited without a response"
        if any(s in detail.lower() for s in ("usage limit", "rate limit", "rate_limit", "quota", "429")):
            raise InsufficientCreditsError(f"Codex subscription limit: {detail}")
        raise CodexError(detail)
    if not messages or not messages[-1].strip():
        raise CodexError("Codex returned no translation")
    return messages[-1].strip(), usage.get("input_tokens", 0), usage.get("output_tokens", 0)


class CodexClient:
    def __init__(self, cli_path="codex", timeout_s=600, max_retries=5):
        self.cli_path = cli_path
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    async def chat_completion(self, model, messages, temperature=0.0, on_retry=None, usage_out=None):
        prompt = "Return only the requested wikitext. Do not use tools.\n\n" + "\n\n".join(
            m["content"] for m in messages
        )
        # Session/auth/model failures should surface immediately; repeating
        # whole CLI sessions would consume subscription allowance needlessly.
        result = await asyncio.to_thread(run_codex_cli, self.cli_path, model, prompt, self.timeout_s)
        if usage_out is not None:
            usage_out["cost"] = 0.0  # Subscription usage, no API dollar charge.
        return result

    async def get_pricing_for(self, model):
        return None

    async def fetch_pricing(self):
        return {}

    async def aclose(self):
        pass
