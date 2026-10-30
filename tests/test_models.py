from wiki_translation_harness.models import RunStats, TranslationResult


def _result(**overrides) -> TranslationResult:
    base = dict(text="translated", model="test-model", prompt_tokens=100, completion_tokens=50, cost_usd=0.01, latency_s=1.5)
    base.update(overrides)
    return TranslationResult(**base)


def test_record_usage_tracks_per_model_breakdown():
    stats = RunStats()
    stats.record_usage("cheap-model", _result(prompt_tokens=100, completion_tokens=50, cost_usd=0.01))
    stats.record_usage("strong-model", _result(prompt_tokens=200, completion_tokens=80, cost_usd=0.05))

    assert set(stats.model_usage.keys()) == {"cheap-model", "strong-model"}
    assert stats.model_usage["cheap-model"].tokens_in == 100
    assert stats.model_usage["cheap-model"].tokens_out == 50
    assert stats.model_usage["cheap-model"].cost_usd == 0.01
    assert stats.model_usage["cheap-model"].calls == 1
    assert stats.model_usage["strong-model"].tokens_in == 200
    assert stats.model_usage["strong-model"].calls == 1
    # Aggregate totals still sum across both models.
    assert stats.tokens_in == 300
    assert round(stats.estimated_cost_usd, 4) == 0.06


def test_merge_usage_from_sums_tokens_cost_and_per_model_breakdown():
    total = RunStats()
    article1 = RunStats()
    article1.record_usage("cheap-model", _result(prompt_tokens=100, completion_tokens=50, cost_usd=0.01))
    article2 = RunStats()
    article2.record_usage("cheap-model", _result(prompt_tokens=20, completion_tokens=10, cost_usd=0.002))
    article2.record_usage("strong-model", _result(prompt_tokens=300, completion_tokens=100, cost_usd=0.08))

    total.merge_usage_from(article1)
    total.merge_usage_from(article2)

    assert total.tokens_in == 420
    assert total.tokens_out == 160
    assert round(total.estimated_cost_usd, 4) == 0.092
    assert total.model_usage["cheap-model"].tokens_in == 120
    assert total.model_usage["cheap-model"].calls == 2
    assert total.model_usage["strong-model"].tokens_in == 300
    assert total.model_usage["strong-model"].calls == 1




def test_usage_limit_needs_whole_words():
    from wiki_translation_harness.models import looks_like_usage_limit

    assert looks_like_usage_limit("HTTP 429 Too Many Requests")
    assert looks_like_usage_limit("You've hit your usage limit")
    assert looks_like_usage_limit("rate_limit_exceeded")
    assert not looks_like_usage_limit("parse error at line 1429")
    assert not looks_like_usage_limit("wrote 14290 tokens")
