"""The single place a new translation engine gets registered.

Every engine just needs to satisfy LLMEngineClient's contract — proven
structural/duck-typed, not nominal, by tests/test_translator.py's
FakeOpenRouterClient, which implements chat_completion() alone with no
inheritance from OpenRouterClient — and be wired into build_llm_client()
below. translator.py's translate_chunk() and repair.py's repair_chunk()
never need to change when a new engine is added; only this file and a new
client module do.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from wiki_translation_harness.config import (
    resolve_complex_provider,
    resolve_llm_endpoint,
    resolve_review_model,
    resolve_review_provider,
)
from wiki_translation_harness.models import Config, ModelPricing
from wiki_translation_harness.openrouter import OpenRouterClient, RetryCallback


class LLMEngineClient(Protocol):
    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        on_retry: RetryCallback | None = None,
        usage_out: dict | None = None,
    ) -> tuple[str, int, int]: ...

    async def get_pricing_for(self, model: str) -> ModelPricing | None: ...

    async def fetch_pricing(self) -> dict[str, ModelPricing]: ...

    async def aclose(self) -> None: ...


def build_llm_client(config: Config) -> tuple[LLMEngineClient, str]:
    """Returns (client, effective_model). effective_model may differ from
    config.model (e.g. provider: local's local_model fallback, resolved by
    resolve_llm_endpoint) — callers should
    config.model_copy(update={"model": effective_model}) if it differs, the
    same way both call sites did before this function replaced their inline
    resolve_llm_endpoint(...) -> OpenRouterClient(...) construction."""
    if config.provider == "codex":
        from wiki_translation_harness.codex_client import CodexClient

        return CodexClient(config.codex_cli_path, config.request_timeout_s, config.max_retries), config.model

    if config.provider == "claude_code":
        # Local import: keeps claude_code_client.py's subprocess/tempfile
        # imports out of the path for anyone only ever using openrouter/local.
        from wiki_translation_harness.claude_code_client import ClaudeCodeClient

        client = ClaudeCodeClient(
            model=config.model,
            cli_path=config.claude_code_cli_path,
            permission_mode=config.claude_code_permission_mode,
            timeout_s=config.request_timeout_s,
            max_retries=config.max_retries,
            log_dir=config.log_dir,
            effort=config.claude_code_effort,
        )
        return client, config.model

    if config.provider == "opencode_go":
        # Local import, same reasoning as claude_code above: keeps this
        # engine's subprocess-invocation imports out of the path for anyone
        # only ever using openrouter/local/experiential/claude_code.
        from wiki_translation_harness.opencode_go_client import OpenCodeGoClient

        client = OpenCodeGoClient(
            model=config.model,
            cli_path=config.opencode_go_cli_path,
            agent=config.opencode_go_agent,
            timeout_s=config.request_timeout_s,
            max_retries=config.max_retries,
            log_dir=config.log_dir,
        )
        return client, config.model

    base_url, api_key, model = resolve_llm_endpoint(config)
    client = OpenRouterClient(
        api_key=api_key,
        base_url=base_url,
        user_agent=config.user_agent,
        timeout=config.request_timeout_s,
        max_retries=config.max_retries,
        provider=config.provider,
        disable_reasoning=config.disable_reasoning,
    )
    return client, model


@dataclass
class ClientPool:
    """One LLMEngineClient per distinct provider this run actually needs.

    Every engine client here accepts `model` as a per-call chat_completion
    argument rather than binding to one fixed model (confirmed for all five
    providers — see claude_code_client.py/opencode_go_client.py's
    chat_completion signatures and OpenRouterClient's), so the only reason
    to ever build a second client is a genuinely different PROVIDER, never
    a different model on the same provider. The common case (every tier on
    one provider) therefore builds exactly one client, same as
    build_llm_client always did; a run that routes complex/review chunks to
    a different provider than the draft tier builds one more, shared across
    every tier that resolves to that same provider."""

    clients: dict[str, LLMEngineClient]

    def get(self, provider: str) -> LLMEngineClient:
        return self.clients[provider]

    async def aclose(self) -> None:
        for client in self.clients.values():
            await client.aclose()


def build_client_pool(config: Config) -> tuple[ClientPool, Config]:
    """Builds build_llm_client's client for the primary provider, plus one
    more per additional distinct provider needed by the complex/review
    tiers (see config.resolve_complex_provider/resolve_review_provider).

    Returns (pool, config) -- config.model may come back updated the same
    way build_llm_client's own (client, effective_model) pair already
    signals a substitution (see its docstring), but only ever for the
    primary provider: complex/review calls always pass their own model
    string explicitly (translator.py's effective_model, pipeline.py's
    review_model) rather than relying on config.model, so no equivalent
    substitution is meaningful for a secondary provider's client."""
    needed: list[str] = [config.provider]
    if config.complex_model:
        provider = resolve_complex_provider(config)
        if provider not in needed:
            needed.append(provider)
    if resolve_review_model(config):
        provider = resolve_review_provider(config)
        if provider not in needed:
            needed.append(provider)

    clients: dict[str, LLMEngineClient] = {}
    for provider in needed:
        provider_config = (
            config if provider == config.provider else config.model_copy(update={"provider": provider})
        )
        client, effective_model = build_llm_client(provider_config)
        clients[provider] = client
        if provider == config.provider and effective_model != config.model:
            config = config.model_copy(update={"model": effective_model})

    return ClientPool(clients=clients), config
