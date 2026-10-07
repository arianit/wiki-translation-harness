import json
import subprocess

import pytest

from wiki_translation_harness.codex_client import CodexClient, CodexError, run_codex_cli
from wiki_translation_harness.config import build_config
from wiki_translation_harness.engines import build_llm_client
from wiki_translation_harness.models import InsufficientCreditsError


def fake_cli(monkeypatch, events, *, login="Logged in using ChatGPT", code=0):
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if "login" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "", login)
        return subprocess.CompletedProcess(cmd, code, "\n".join(json.dumps(e) for e in events), "")

    monkeypatch.setattr(subprocess, "run", run)
    return calls


@pytest.mark.asyncio
async def test_subscription_translation_and_usage(monkeypatch):
    calls = fake_cli(monkeypatch, [
        {"type": "item.completed", "item": {"type": "agent_message", "text": "== Historia ==\nPërkthimi."}},
        {"type": "turn.completed", "usage": {"input_tokens": 42, "output_tokens": 12}},
    ])
    usage = {}
    result = await CodexClient().chat_completion("gpt-6-luna", [{"role": "user", "content": "Translate"}], usage_out=usage)
    assert result == ("== Historia ==\nPërkthimi.", 42, 12)
    assert usage["cost"] == 0
    cmd, kwargs = calls[-1]
    assert '--ignore-user-config' in cmd
    assert 'forced_login_method="chatgpt"' in cmd
    assert 'features.shell_tool=false' in cmd
    assert cmd[cmd.index('--sandbox') + 1] == 'read-only'
    assert kwargs['input'].endswith('Translate')


def test_api_auth_is_rejected_before_translation(monkeypatch):
    calls = fake_cli(monkeypatch, [], login="Logged in using an API key")
    with pytest.raises(CodexError, match="subscription authentication"):
        run_codex_cli("codex", "gpt-6-luna", "Translate", 60)
    assert len(calls) == 1


def test_subscription_limit_is_detected(monkeypatch):
    fake_cli(monkeypatch, [{"type": "turn.failed", "error": {"message": "You've hit your usage limit"}}], code=1)
    with pytest.raises(InsufficientCreditsError):
        run_codex_cli("codex", "gpt-6-luna", "Translate", 60)


def test_codex_provider_and_fallback_config(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = build_config(None, {"provider": "codex", "wikimedia_contact": "test@example.com"})
    client, model = build_llm_client(config)
    assert isinstance(client, CodexClient)
    assert model == "gpt-6-luna"
    config = build_config(None, {
        "wikimedia_contact": "test@example.com", "fallback_provider": "codex",
        "fallback_model": "gpt-6-luna", "fallback_auto_switch": True,
    })
    assert config.provider == "claude_code"
    assert config.fallback_provider == "codex"
    assert config.fallback_auto_switch
