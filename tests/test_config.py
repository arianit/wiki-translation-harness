from pathlib import Path

import pytest

from wiki_translation_harness.config import (
    build_config,
    resolve_review_model,
    resolve_review_provider,
)
from wiki_translation_harness.models import Config

_CONTACT = {"wikimedia_contact": "test@example.com"}


def test_missing_api_key_raises(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No OpenRouter API key"):
        build_config(None, {**_CONTACT, "provider": "openrouter"})


def test_default_provider_and_workers():
    cfg = build_config(None, _CONTACT)
    assert cfg.provider == "claude_code"
    assert cfg.model == "claude-sonnet-5-5"
    assert cfg.workers == 2


def test_switching_to_openrouter_without_model_gets_openrouter_default(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    cfg = build_config(None, {**_CONTACT, "provider": "openrouter"})
    assert cfg.model == "deepseek/deepseek-v3.2"


def test_explicit_model_survives_provider_switch(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    cfg = build_config(None, {**_CONTACT, "provider": "openrouter", "model": "qwen/qwen3-235b-a22b"})
    assert cfg.model == "qwen/qwen3-235b-a22b"


def test_opencode_go_needs_no_api_key(monkeypatch):
    # Unlike openrouter/experiential, opencode_go authenticates via the
    # opencode CLI's own separate login -- build_config must not raise for
    # lack of an API key the way it does for those two providers.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("EXPLABS_API_KEY", raising=False)
    cfg = build_config(None, {**_CONTACT, "provider": "opencode_go"})
    assert cfg.provider == "opencode_go"


def test_cli_provider_switch_drops_stale_yaml_model(tmp_path: Path, monkeypatch):
    """Regression test: --provider claude_code alone, against a config.yaml
    with provider: openrouter and an OpenRouter model id, must not silently
    carry that model id over -- confirmed live to get rejected by the
    Claude Code CLI as an unrecognized model."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("provider: openrouter\nmodel: deepseek/deepseek-chat-v3-0324\n")
    cfg = build_config(config_file, {**_CONTACT, "provider": "claude_code"})
    assert cfg.provider == "claude_code"
    assert cfg.model == "claude-sonnet-5-5"


def test_cli_overrides_win_over_yaml(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config_file = tmp_path / "config.yaml"
    config_file.write_text("model: deepseek/deepseek-v3\nworkers: 4\nopenrouter_api_key: sk-yaml\n")
    cfg = build_config(config_file, {**_CONTACT, "model": "google/gemini-2.5-flash", "workers": 16})
    assert cfg.model == "google/gemini-2.5-flash"
    assert cfg.workers == 16


def test_env_var_overrides_yaml_key(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("openrouter_api_key: sk-yaml\n")
    cfg = build_config(config_file, _CONTACT)
    assert cfg.openrouter_api_key == "sk-env"


def test_missing_wikimedia_contact_raises(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    monkeypatch.delenv("WIKIMEDIA_CONTACT", raising=False)
    with pytest.raises(ValueError, match="User-Agent policy"):
        build_config(None, {})


def test_user_agent_computed_from_contact(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    cfg = build_config(None, {"wikimedia_contact": "someone@example.com", "wikimedia_tool_name": "mybot"})
    assert cfg.user_agent.startswith("mybot/")
    assert "someone@example.com" in cfg.user_agent


def test_resolve_review_model_inherits_complex_model():
    cfg = Config.model_validate({**_CONTACT, "complex_model": "claude-sonnet-5"})
    assert resolve_review_model(cfg) == "claude-sonnet-5"


def test_resolve_review_provider_inherits_complex_provider():
    cfg = Config.model_validate(
        {
            **_CONTACT,
            "provider": "opencode_go",
            "complex_model": "claude-sonnet-5",
            "complex_provider": "claude_code",
        }
    )
    assert resolve_review_provider(cfg) == "claude_code"


def test_claude_code_effort_defaults_to_medium():
    cfg = Config.model_validate({**_CONTACT})
    assert cfg.claude_code_effort == "medium"


