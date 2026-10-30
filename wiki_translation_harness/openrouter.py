"""OpenRouter chat-completions client: retry/backoff and pricing lookup only.

No translation prompts live here — callers pass in whatever messages the
skill_loader built. This module only knows how to talk to OpenRouter.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Callable

import httpx

from wiki_translation_harness.models import (
    EngineError,
    InsufficientCreditsError,
    ModelPricing,
    TranslationResult,
)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503}

RetryCallback = Callable[[int, str, float], None]


class OpenRouterError(EngineError):
    pass


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        user_agent: str,
        timeout: float = 120.0,
        max_retries: int = 5,
        provider: str = "openrouter",
        disable_reasoning: bool = False,
    ):
        self.max_retries = max_retries
        self.provider = provider
        self.disable_reasoning = disable_reasoning
        # Confirmed in practice, twice: httpx's own `timeout=` did not
        # reliably fire under real network conditions (sockets sat in
        # CLOSE-WAIT with unread data for 20+ minutes past the configured
        # timeout, hanging the whole batch run indefinitely). This is a
        # hard, independent backstop around every individual call attempt
        # — asyncio.wait_for guarantees cancellation regardless of what
        # httpx/the underlying transport is doing.
        self._hard_timeout = timeout + 30.0
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": user_agent,
        }
        if provider == "openrouter":
            # OpenRouter-specific attribution headers; meaningless (and
            # unwanted) against a local OpenAI-compatible server.
            headers["HTTP-Referer"] = "https://github.com/arianit/enwiki-sqwiki-translation"
            headers["X-Title"] = "wiki-translation-harness"
        self._client_kwargs = dict(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
        )
        self._client = httpx.AsyncClient(**self._client_kwargs)
        self._pricing_cache: dict[str, ModelPricing] | None = None

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _reset_connection(self) -> None:
        # asyncio.wait_for cancels the in-flight httpx call on our hard
        # timeout, but cancellation doesn't guarantee httpx cleanly evicts
        # the underlying connection from its keep-alive pool — confirmed in
        # practice: repeated retries after a timeout kept failing with the
        # identical error, on what should be independent attempts, exactly
        # as if every retry kept reusing the same wedged socket. Closing
        # and rebuilding the client forces a genuinely fresh connection.
        old_client = self._client
        self._client = httpx.AsyncClient(**self._client_kwargs)
        # Swap the client in immediately so the next attempt never waits on
        # this — a genuinely wedged connection could make aclose() itself
        # hang, which would defeat the whole point of resetting it. Give it
        # a bounded grace period to shut down cleanly, then abandon it.
        try:
            await asyncio.wait_for(old_client.aclose(), timeout=5.0)
        except (asyncio.TimeoutError, httpx.TransportError):
            pass

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        on_retry: RetryCallback | None = None,
        usage_out: dict | None = None,
    ) -> tuple[str, int, int]:
        """Returns (text, prompt_tokens, completion_tokens).

        usage_out, when passed, is filled in-place with the response's raw
        `usage` object -- a fresh dict per call (never shared/mutated
        concurrently), so run_completion() can read provider-reported fields
        (e.g. Experiential Labs' inline `usage.cost`) without widening this
        method's return type for every engine (see engines.LLMEngineClient).
        """
        attempt = 0
        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
        }
        # Claude Sonnet 5 / Opus 4.7+ reject non-default sampling params with
        # a 400, so only send temperature to non-Anthropic models.
        if not model.startswith(("anthropic/", "claude-")):
            payload["temperature"] = temperature
        if self.disable_reasoning and self.provider == "openrouter":
            payload["reasoning"] = {"enabled": False}
        if self.provider == "experiential":
            # OpenAI-standard attribution field (Experiential Labs' Cost API
            # docs: "Pass the OpenAI-standard `safety_identifier` on every
            # request") -- a body field, not a header, unlike OpenRouter's
            # HTTP-Referer/X-Title above.
            payload["safety_identifier"] = "wiki-translation-harness"
        while True:
            try:
                resp = await asyncio.wait_for(
                    self._client.post("/chat/completions", json=payload),
                    timeout=self._hard_timeout,
                )
            except (httpx.TransportError, asyncio.TimeoutError) as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise OpenRouterError(
                        f"OpenRouter connection failed after {attempt} attempts: {exc}"
                    ) from exc
                await self._reset_connection()
                await self._backoff(attempt, on_retry, reason=f"connection error: {exc}")
                continue

            if resp.status_code == 429:
                quota_error = self._insufficient_quota_error(resp)
                if quota_error is not None:
                    raise quota_error

            if resp.status_code in RETRYABLE_STATUS_CODES:
                attempt += 1
                if attempt > self.max_retries:
                    raise OpenRouterError(
                        f"OpenRouter returned HTTP {resp.status_code} after {attempt} attempts: "
                        f"{resp.text[:500]}"
                    )
                await self._backoff(attempt, on_retry, reason=f"HTTP {resp.status_code}")
                continue

            if resp.status_code == 402:
                # Payment Required -- out of (or insufficient) account
                # balance for this request. Not in RETRYABLE_STATUS_CODES on
                # purpose: retrying the same engine can't help, unlike a
                # transient 429/5xx. A distinct exception type so pipeline.py
                # can offer a fallback-provider switch instead of just
                # failing the chunk like any other EngineError.
                raise InsufficientCreditsError(
                    f"OpenRouter request failed: HTTP 402 (insufficient credits): {resp.text[:500]}"
                )

            if resp.status_code >= 400:
                raise OpenRouterError(
                    f"OpenRouter request failed: HTTP {resp.status_code}: {resp.text[:500]}"
                )

            try:
                data = resp.json()
            except json.JSONDecodeError as exc:
                # A 200 status with a truncated/corrupt body — observed in
                # practice right after flaky-connection retries, i.e. the
                # same underlying transient network issue, just manifesting
                # after headers were already sent. Retry like a connection
                # error rather than crashing the whole batch.
                attempt += 1
                if attempt > self.max_retries:
                    raise OpenRouterError(
                        f"OpenRouter returned unparseable response after {attempt} attempts: {exc}"
                    ) from exc
                await self._reset_connection()
                await self._backoff(attempt, on_retry, reason=f"malformed response body: {exc}")
                continue
            break

        choices = data.get("choices") or []
        if not choices:
            raise OpenRouterError(f"OpenRouter response had no choices: {data}")
        text = (choices[0].get("message") or {}).get("content")
        if text is None:
            # A null content (refusal, content filter, reasoning-only reply)
            # is an engine failure, not a crash in run_completion's .strip().
            raise OpenRouterError(
                f"OpenRouter response had no message content "
                f"(finish_reason={choices[0].get('finish_reason')!r}): {str(data)[:500]}"
            )
        usage = data.get("usage") or {}
        if usage_out is not None:
            usage_out.update(usage)
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        return text, prompt_tokens, completion_tokens

    def _insufficient_quota_error(self, resp: httpx.Response) -> InsufficientCreditsError | None:
        """HTTP 429 is ambiguous across providers: OpenRouter uses it purely
        for transient rate limiting (retryable, see RETRYABLE_STATUS_CODES),
        while Experiential Labs also uses it for a genuine "out of
        credits/quota" condition -- distinguished only by the JSON error
        body's stable `code` field (docs explicitly warn the `message` text
        isn't stable), never by status code alone. Parsed generically here
        (not gated on self.provider) since a body that doesn't match this
        shape just falls through to the existing retryable-429 path either
        way -- OpenRouter's own 429s have never been observed to carry this
        envelope."""
        try:
            body = resp.json()
        except json.JSONDecodeError:
            return None
        error = body.get("error") if isinstance(body, dict) else None
        if not isinstance(error, dict):
            return None
        if error.get("code") != "insufficient_quota":
            return None
        return InsufficientCreditsError(
            f"{self.provider} request failed: HTTP 429 (code=insufficient_quota): {resp.text[:500]}"
        )

    async def _backoff(
        self, attempt: int, on_retry: RetryCallback | None, reason: str
    ) -> None:
        delay = min(2 ** (attempt - 1), 60) + random.uniform(0, 1)
        if on_retry is not None:
            on_retry(attempt, reason, delay)
        await asyncio.sleep(delay)

    async def fetch_pricing(self) -> dict[str, ModelPricing]:
        if self.provider != "openrouter":
            # Local servers have no meaningful pricing endpoint; local
            # inference is treated as free (cost always reports as 0.0).
            # Experiential Labs is also skipped here (unconfirmed whether
            # its /v1/models exposes OpenRouter's per-entry `pricing` shape)
            # -- it reports real per-call cost via chat_completion's
            # usage_out instead, read directly off each response's inline
            # `usage.cost` in run_completion() below, so a $0 pricing table
            # doesn't matter for it the way it would for a truly free local
            # engine.
            return {}
        if self._pricing_cache is not None:
            return self._pricing_cache
        resp = await self._client.get("/models")
        resp.raise_for_status()
        data = resp.json()

        pricing: dict[str, ModelPricing] = {}
        for entry in data.get("data", []):
            model_id = entry.get("id")
            price_info = entry.get("pricing") or {}
            if not model_id:
                continue
            try:
                prompt_price = float(price_info.get("prompt", 0) or 0)
                completion_price = float(price_info.get("completion", 0) or 0)
            except (TypeError, ValueError):
                continue
            pricing[model_id] = ModelPricing(
                model_id=model_id,
                prompt_price_per_token=prompt_price,
                completion_price_per_token=completion_price,
            )
        self._pricing_cache = pricing
        return pricing

    async def get_pricing_for(self, model: str) -> ModelPricing | None:
        pricing = await self.fetch_pricing()
        return pricing.get(model)


def compute_cost(pricing: ModelPricing | None, prompt_tokens: int, completion_tokens: int) -> float:
    if pricing is None:
        return 0.0
    return (
        prompt_tokens * pricing.prompt_price_per_token
        + completion_tokens * pricing.completion_price_per_token
    )


async def run_completion(
    client: OpenRouterClient,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    pricing: ModelPricing | None,
    on_retry: RetryCallback | None = None,
) -> TranslationResult:
    """Shared call+timing+cost path used by both translate and repair invocations."""
    start = time.monotonic()
    usage: dict = {}
    text, prompt_tokens, completion_tokens = await client.chat_completion(
        model, messages, temperature=temperature, on_retry=on_retry, usage_out=usage
    )
    latency = time.monotonic() - start
    # Prefer a provider-reported settled cost (Experiential Labs stamps
    # `usage.cost` on every response) over the external per-token pricing
    # table -- OpenRouter/local responses never carry this key, so `cost`
    # stays None for them and compute_cost's table-based estimate applies
    # exactly as before.
    reported_cost = usage.get("cost")
    if reported_cost is not None:
        cost = float(reported_cost)
    else:
        cost = compute_cost(pricing, prompt_tokens, completion_tokens)
    return TranslationResult(
        text=text.strip(),
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost,
        latency_s=latency,
        cached=False,
    )
