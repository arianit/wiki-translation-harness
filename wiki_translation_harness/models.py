"""Typed data models shared across the harness.

No translation logic lives here — only the shapes that carry data between
the fetch / chunk / cache / invoke-skill / validate / repair / save stages.
"""

from __future__ import annotations

import time
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Shared between Config.provider and Config.fallback_provider's validators
# so the two lists of accepted engine names can't drift apart.
_VALID_PROVIDERS = ("openrouter", "local", "claude_code", "experiential", "opencode_go", "codex")

# The `claude` CLI's own accepted --effort values (confirmed via `claude -p
# --help`), matching the Messages API's output_config.effort levels.
_VALID_EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh", "max"})


class ArticleStatus(str, Enum):
    PENDING = "pending"
    FETCHED = "fetched"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


class ChunkStatus(str, Enum):
    PENDING = "pending"
    CACHED = "cached"
    TRANSLATED = "translated"
    REPAIRED = "repaired"
    FAILED = "failed"


class Section(BaseModel):
    """A logical section of an article, as split by parser.py."""

    title: str
    level: int
    order: int
    wikitext: str


class Chunk(BaseModel):
    """A translation unit: one or more merged sections, sized ~1500-2500 tokens.

    Small adjacent sections are merged into one chunk to amortize the fixed
    per-request skill-prompt cost; an oversized section is split internally
    (never inside a template/table/reference/list) and keeps only its own
    title across the resulting sub-chunks.
    """

    article_title: str
    section_titles: list[str]
    order: int
    text: str
    token_estimate: int
    source_lang: str = "en"
    status: ChunkStatus = ChunkStatus.PENDING
    translated_text: str | None = None
    is_complex: bool = False

    @property
    def section_title(self) -> str:
        return "; ".join(self.section_titles)

    @property
    def chunk_id(self) -> str:
        return f"{self.article_title}::{self.order}"


class ValidationIssue(BaseModel):
    kind: str
    message: str
    # Informational only — does not change ValidationResult.valid gating
    # (any issue, of any severity, still invalidates). Lets callers surface
    # a defect's seriousness in reports/state without silently tolerating
    # anything.
    severity: Literal["error", "warning"] = "error"
    # Both approximate/best-effort where set: for live (HTML-derived)
    # findings there is no wikitext line mapping, so these are located by
    # searching the source wikitext for the finding's identifying string.
    line_number: int | None = None
    snippet: str | None = None

    def as_finding(self) -> dict[str, object]:
        """The {severity, line_number, snippet, explanation} shape used for
        repair prompts and the needs_human_review record."""
        return {
            "severity": self.severity,
            "line_number": self.line_number,
            "snippet": self.snippet,
            "explanation": self.message,
        }


class ValidationResult(BaseModel):
    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.valid


class EngineError(Exception):
    """Base class for a translation-engine call failure, regardless of which
    engine (OpenRouter, Claude Code CLI, a future one) raised it — the two
    places that catch this to gracefully fail one chunk/article instead of
    crashing the whole run (pipeline.py, benchmark.py) shouldn't need a new
    except clause every time a new engine is added."""


class InsufficientCreditsError(EngineError):
    """The configured engine refused a call for lack of funds (e.g.
    OpenRouter's HTTP 402) rather than any transient/retryable reason.
    Retrying the same engine won't help — pipeline.py catches this
    specifically to offer a switch to config.fallback_provider instead of
    just failing the chunk like a generic EngineError."""


class TranslationResult(BaseModel):
    """Result of one skill invocation (translate or repair)."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    cached: bool = False


class ModelPricing(BaseModel):
    model_id: str
    prompt_price_per_token: float
    completion_price_per_token: float


class ArticleSource(BaseModel):
    """Where an article came from and its raw wikitext, prior to chunking."""

    title: str
    wikitext: str
    revid: int | None = None
    revision_timestamp: str | None = None
    source_lang: str = "en"


class ArticleJob(BaseModel):
    """Tracks one article's progress through the pipeline."""

    title: str
    status: ArticleStatus = ArticleStatus.PENDING
    output_path: Path | None = None
    error: str | None = None
    sections_total: int = 0
    sections_done: int = 0


class ModelUsage(BaseModel):
    """Per-model token/cost breakdown -- one entry per distinct model id
    actually passed to an engine's chat_completion call. See
    RunStats.model_usage/record_usage."""

    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    calls: int = 0


class RunStats(BaseModel):
    """Aggregate counters for stats.json, updated throughout a run."""

    started_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    # Effective provider/model in force when this run ended -- set from the
    # (possibly fallback-switched, see pipeline.py's ensure_fallback_engine)
    # config, not necessarily what the run started with. None if the run
    # crashed before build_llm_client() ever ran.
    provider: str | None = None
    model: str | None = None
    articles_completed: int = 0
    articles_failed: int = 0
    articles_skipped: int = 0
    # Separate from articles_failed: an article whose assembled output still
    # had unresolved defects after the assembly-level repair loop, rather
    # than one that crashed/timed out. See review_queue.py.
    articles_needs_human_review: int = 0
    sections_translated: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    estimated_cost_usd: float = 0.0
    translation_time_s: float = 0.0
    validation_failures: int = 0
    repair_attempts: int = 0
    repairs_succeeded: int = 0
    repairs_failed: int = 0
    retries: int = 0
    # Effective review model/provider in force when this run ended (see
    # config.resolve_review_model/resolve_review_provider) -- None if the
    # review pass was disabled for this run. Separate from repair_attempts/
    # repairs_succeeded/repairs_failed above (the per-chunk structural
    # repair loop), same naming convention.
    review_model: str | None = None
    review_provider: str | None = None
    review_attempts: int = 0
    review_corrections_applied: int = 0
    review_findings_total: int = 0
    # Separate from articles_needs_human_review: an article whose review
    # pass still had unresolved findings after review_max_repair_attempts,
    # but was saved anyway (see review_queue.record_review_flags) rather
    # than withheld -- a fidelity concern is a spot-check prompt, not a
    # publish blocker.
    articles_flagged_by_review: int = 0
    # Per-model breakdown of tokens_in/tokens_out/estimated_cost_usd above --
    # keyed by the exact model id string passed to chat_completion for that
    # call, so a run using complex_model/review_model (see Config) can show
    # e.g. how much the cheap draft model spent vs. the stronger
    # complex/review model, instead of only one lumped total. Every call
    # that updates the aggregate totals below is expected to go through
    # record_usage() so the two never drift apart.
    model_usage: dict[str, ModelUsage] = Field(default_factory=dict)

    def record_usage(self, model: str, result: TranslationResult) -> None:
        """The single place any LLM call's usage should be recorded: updates
        both the run-wide aggregate totals and this specific model's own
        breakdown in one call, so they can't drift out of sync. Replaces
        translator.py's old private _accumulate() and pipeline.py's several
        inline stats.tokens_in += ... blocks."""
        self.tokens_in += result.prompt_tokens
        self.tokens_out += result.completion_tokens
        self.estimated_cost_usd += result.cost_usd
        self.translation_time_s += result.latency_s
        usage = self.model_usage.setdefault(model, ModelUsage())
        usage.tokens_in += result.prompt_tokens
        usage.tokens_out += result.completion_tokens
        usage.cost_usd += result.cost_usd
        usage.calls += 1

    def merge_usage_from(self, other: "RunStats") -> None:
        """Adds another RunStats' token/cost/timing counters and per-model
        breakdown into this one -- used by queue_runner.py to aggregate
        spend across every article processed in a `queue` run, since each
        article gets its own fresh RunStats from run_pipeline (so the
        top-level queue-run tracker otherwise never sees any of it).
        Deliberately excludes articles_completed/articles_failed/etc: those
        are queue_runner's own outcome bookkeeping, decided independently
        of what run_pipeline's per-article stats say (e.g. an article that
        crashed mid-translation still needs to count as a queue-level
        failure even though run_pipeline's own counters never got that
        far)."""
        self.sections_translated += other.sections_translated
        self.cache_hits += other.cache_hits
        self.cache_misses += other.cache_misses
        self.tokens_in += other.tokens_in
        self.tokens_out += other.tokens_out
        self.estimated_cost_usd += other.estimated_cost_usd
        self.translation_time_s += other.translation_time_s
        self.validation_failures += other.validation_failures
        self.repair_attempts += other.repair_attempts
        self.repairs_succeeded += other.repairs_succeeded
        self.repairs_failed += other.repairs_failed
        self.retries += other.retries
        self.review_attempts += other.review_attempts
        self.review_corrections_applied += other.review_corrections_applied
        self.review_findings_total += other.review_findings_total
        for model, usage in other.model_usage.items():
            target = self.model_usage.setdefault(model, ModelUsage())
            target.tokens_in += usage.tokens_in
            target.tokens_out += usage.tokens_out
            target.cost_usd += usage.cost_usd
            target.calls += usage.calls


class Config(BaseModel):
    """Full harness configuration: config.yaml merged with CLI overrides."""

    # "claude_code" (default, uses the caller's existing Claude Code CLI
    # login — no API key), "openrouter", "local", or "experiential" (the
    # Experiential Labs gateway, platform.experientiallabs.ai — an
    # OpenAI-compatible multi-model gateway, reuses OpenRouterClient). See
    # engines.py.
    provider: str = "claude_code"
    # Engine to switch to (mid-run, one-way) if `provider` raises
    # InsufficientCreditsError -- interactively confirmed for a manual run,
    # auto-switched (just logged) under `queue`. None lets pipeline.py apply
    # its own default (claude_code whenever `provider` isn't already
    # claude_code); set equal to `provider` to disable the offer entirely.
    # See pipeline.py's ensure_fallback_engine().
    fallback_provider: str | None = None
    fallback_model: str | None = None
    # Explicit opt-in for automatic switching even during interactive runs.
    fallback_auto_switch: bool = False
    codex_cli_path: str = "codex"
    model: str = "claude-sonnet-5-5"
    # When set, complex chunks (Infoboxes, large tables, dense ref lists)
    # are routed to this model instead of `model` — a hybrid strategy that
    # uses a cheaper/faster model for ~80% of standard body text while
    # reserving the stronger model for markup-critical sections.
    complex_model: str | None = None
    # Provider to run complex_model on, when it should differ from
    # `provider` (e.g. a cheap draft tier on opencode_go, complex chunks on
    # a stronger model via claude_code). None reuses the primary client —
    # today's behavior, only the model string changes per call. See
    # engines.build_client_pool.
    complex_provider: str | None = None
    # Independent semantic-fidelity review pass: after assembly-level
    # structural repair passes clean, the WHOLE assembled article is handed
    # to this model, alongside the English source, to look for
    # mistranslation, hallucinated/dropped facts, grammar errors, and
    # cross-article transliteration inconsistency — defects structural
    # validation (validator.py) cannot see, and that wikiqa's same-model
    # self-check is not independent enough to reliably catch either. See
    # pipeline.run_review_pass / review.py.
    #
    # Orthogonal to complex_model: complex_model routes structurally
    # complex chunks to a stronger model at DRAFT time; review_model
    # re-reads the whole ASSEMBLED article afterward for translation
    # fidelity, independent of any one chunk's structural complexity.
    # Presence is the on/off switch for the review pass, same convention as
    # complex_model. Defaults to complex_model when unset (and complex_model
    # is set) — a run that already configured a stronger complex-chunk
    # model gets review "for free" on that same model, matching the common
    # case where both roles share one higher-quality model. See
    # config.resolve_review_model.
    review_model: str | None = None
    # Defaults to complex_provider (then provider) when unset, same
    # inheritance rule as review_model -> complex_model above. See
    # config.resolve_review_provider.
    review_provider: str | None = None
    # Repair-round budget for the review pass, mirroring
    # max_repair_attempts/max_assembly_repair_rounds. Unresolved findings
    # after this cap do NOT block delivery (see review_queue.record_review_flags)
    # — the article is still saved, with a companion flagged-addendum file,
    # since a fidelity concern is a human spot-check prompt, not a publish
    # blocker the way an unresolved structural defect is.
    review_max_repair_attempts: int = 2
    workers: int = 2
    # Process articles one at a time (still using up to `workers` concurrent
    # chunk translations within each article) instead of starting every
    # requested article's chunks concurrently. Bounds the blast radius of a
    # mid-run failure (e.g. a provider rate/session limit) to the single
    # article in flight, and makes multi-article runs easier to reason
    # about/resume. See run_pipeline()'s article-dispatch branch.
    sequential: bool = True
    temperature: float = 0.0
    max_retries: int = 5
    cache: bool = True
    validate_output: bool = Field(default=True, alias="validate")
    repair: bool = True
    max_repair_attempts: int = 2

    # Whole-article checks (live parse-API + the static template/table
    # checks in validator.py, re-run on the assembled article) that only
    # make sense once every chunk is in place — independent from the
    # per-chunk loop above, with its own repair-round budget.
    live_validate: bool = True
    live_validate_timeout_s: float = 30.0
    max_assembly_repair_rounds: int = 3
    # Chunks repaired at once within one assembly-repair round. A round on
    # a long article can touch ~20 chunks; repaired one at a time that is
    # ~15 minutes per round.
    assembly_repair_concurrency: int = 4

    source_lang: str = "en"
    target_lang: str = "sq"

    # Optional override for the MediaWiki API endpoint used for source_lang
    # specifically (e.g. a non-Wikipedia wiki, or a custom mirror). Any other
    # source language encountered (via a `lang:Title` prefix or full URL —
    # see sources.parse_source_ref) resolves generically to
    # https://{lang}.wikipedia.org/w/api.php, not this override.
    source_wiki_api: str | None = None

    chunk_min_tokens: int = 1500
    chunk_max_tokens: int = 2500

    # A single skill directory, or a list of them: the translation-judgment
    # guidance sent on *every* normal translation/repair call (see
    # skill_loader.load_skill). Split across two directories by default —
    # enwiki-sqwiki-translation (translate) and wikiterms (terminology/link
    # conventions, e.g. {{ill}} usage) — concatenated into one system
    # prompt, translate listed first since its content frames the other.
    # wikiqa (the pre-delivery QA checklist) deliberately is NOT here: it's
    # loaded separately as qa_skill_path below and only ever reaches a
    # repair call, which only happens once validate_wikitext has already
    # found a real problem — sending its checklist on every ordinary
    # section (most of which never fail validation) would just be wasted
    # tokens repeated wholesale on every request.
    skill_path: Path | list[Path] = Field(
        default_factory=lambda: [
            Path.home() / ".claude" / "skills" / "enwiki-sqwiki-translation",
            Path.home() / ".claude" / "skills" / "wikiterms",
        ]
    )
    # Loaded the same way as skill_path but kept separate and appended only
    # to repair messages (skill_loader.build_repair_messages) — never sent
    # on a normal translation call. None disables it (repair then relies
    # solely on skill_path's content plus the specific errors listed).
    qa_skill_path: Path | list[Path] | None = Field(
        default_factory=lambda: Path.home() / ".claude" / "skills" / "wikiqa"
    )
    include_skill_references: bool = False
    # If set (e.g. "HEAD"), the skill is read from this git revision instead
    # of the working tree, so local uncommitted edits to the skill's repo
    # don't silently change translation behavior. skill_path must then sit
    # inside a git working copy of the skill's repo. None reads plain files.
    skill_git_ref: str | None = None

    # Defaults into the shared wiki-translation-queue repo's output/ folder
    # (github.com/arianit/wiki-translate-queue) so articles produced by this
    # harness land in the same place as wikitranslateautorun's/mmtp's/
    # wikipedia-articles-translation's -- override in config.yaml for a
    # purely local run.
    output_dir: Path = Path("~/code/wiki-translation-queue/output")
    cache_db_path: Path = Path("cache") / "translation_memory.sqlite3"
    log_dir: Path = Path("logs")
    stats_path: Path = Path("stats.json")
    # Local-only "kill-safe" snapshot of in-progress articles — rewritten
    # after every chunk, deleted once save_article() succeeds. Never
    # shared/synced (unlike output_dir, a cross-machine directory that must
    # hold only finished, reviewed translations) so a partial/WIP article
    # never reaches another consumer of that shared queue.
    partial_output_dir: Path = Path("partial_output")

    # Harness-side link/template verification against Wikidata + the target
    # wiki (see verification.py) — the harness's own, growing equivalent of
    # the skill's sqwiki-verified.md, fed into each chunk's translation
    # request as pre-checked facts instead of the model having to guess.
    verify_links: bool = True
    verification_db_path: Path = Path("cache") / "verified_facts.sqlite3"
    generate_reports: bool = True
    # Wikidata/target-wiki lookups should complete in a few seconds — this
    # is deliberately its own (short) setting, not request_timeout_s, which
    # is sized for slow LLM completions and would let a single hung lookup
    # run for two minutes before even hitting the outer deadline below.
    wikidata_timeout_s: float = 20.0
    # Hard outer deadline for the whole verification step, independent of
    # httpx's own per-call timeouts — a network call hanging past its
    # configured timeout must never stall an entire batch run.
    verification_timeout_s: float = 60.0

    # Fills any citation missing |language= in the final assembled article:
    # visits the cited URL and reads its declared language, falling back to
    # guessing from the citation title. See citation_language.py.
    fill_citation_languages: bool = True
    max_citation_url_fetches: int = 40
    citation_fetch_concurrency: int = 5
    citation_fetch_timeout_s: float = 10.0
    citation_fill_timeout_s: float = 60.0

    # Deterministic backstop: renames citation parameter *names* that got
    # mistranslated into the target language back to their English CS1
    # equivalents (e.g. |titulli= -> |title=). See citation_language.py.
    fix_citation_param_names: bool = True

    # Reconciles {{sfn}}/{{harvnb}} calls that share the same auto-generated
    # anchor (same author+year+page) but ended up with differently
    # -paraphrased |ps= quotes because the same source citation was split
    # across independently-translated chunks. See citation_language.py.
    dedupe_short_footnotes: bool = True

    # Maximum allowed time (seconds) to spend translating a single article (excluding
    # verification and post-processing). If translation exceeds this limit, the article is
    # marked as failed and the harness moves on to the next article.
    # Was 1800s, then 7200s while `provider: claude_code` + workers=1 forced
    # every chunk of an article through one serialized CLI subprocess call
    # (a 39-section article needed more than 1800s that way). Now that the
    # default config is `provider: openrouter` + workers=4 -- true
    # concurrent HTTP calls, not CLI-subprocess-serialized -- wall-clock
    # cost per article drops substantially, so this comes back down to
    # 3600s rather than staying at the workers=1 figure. Not quartered back
    # to 1800s: openrouter's own 429/5xx backoff retries (unlike
    # claude_code_client.py, which fails fast and never retries) now
    # genuinely consume time inside this same per-article budget, so some
    # headroom above the old default is kept. Single constant, not a
    # `workers`-derived formula, since workers is meant to be tunable
    # per-run without silently starving this budget.
    article_timeout_s: float = 3600.0  # 1 hour

    # Maximum allowed ratio of output tokens to input tokens per article.
    # A ratio > 1 is expected (translation often expands text), but extremely high ratios
    # suggest a runaway generation (e.g., the model started repeating or hallucinating).
    max_token_ratio: float = 5.0

    # Maximum total tokens (input + output) allowed per article. Zero means unlimited.
    max_article_tokens: int = 0

    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Send `reasoning: {"enabled": false}` on OpenRouter calls. Needed for
    # reasoning-by-default models (e.g. qwen/qwen3.6-27b) that otherwise spend
    # the whole output budget on hidden thinking (no max_tokens is sent, so a
    # single call can run past request_timeout_s and retry in a loop). Off by
    # default; only applies when provider is openrouter.
    disable_reasoning: bool = False

    # Local OpenAI-compatible server (llama.cpp server, Ollama, LM Studio,
    # vLLM, ...) as an alternative to OpenRouter. Selected via provider: local.
    local_base_url: str = "http://localhost:8080/v1"
    local_api_key: str | None = None
    # Falls back to `model` when unset, so --model keeps working for local too.
    local_model: str | None = None

    # Experiential Labs (platform.experientiallabs.ai) -- a curated
    # multi-model gateway speaking the same OpenAI wire protocol as
    # OpenRouter (POST /v1/chat/completions, Bearer auth), selected via
    # provider: experiential. See resolve_llm_endpoint() and
    # openrouter.py's provider-aware insufficient-quota/cost handling.
    experiential_api_key: str | None = None
    experiential_base_url: str = "https://api.experientiallabs.ai/v1"
    # Falls back to `model` when unset, same pattern as local_model.
    experiential_model: str | None = None

    # Claude Code CLI engine (provider: claude_code, the default) — runs
    # `claude -p` under the caller's existing Claude Code subscription/login,
    # no API key needed. See claude_code_client.py.
    claude_code_cli_path: str = "claude"
    claude_code_permission_mode: str = "bypassPermissions"
    # Passed to `claude -p --effort <level>` (confirmed via `claude -p
    # --help`) -- the same thinking-depth/token-spend knob as the Messages
    # API's output_config.effort, whose own default is "high". Defaults to
    # "medium" here instead: this harness's calls are short, well-specified,
    # single-section translate/repair/review tasks, not long-horizon
    # agentic work, so the top of the effort range buys little quality for
    # a lot of extra token spend on this workload. One client (and setting)
    # per provider (see engines.build_client_pool), so this applies to
    # every model routed through provider: claude_code in a given run --
    # currently always claude-sonnet-5-5 by default (config.default_model_for_provider).
    # None omits the flag, letting the CLI use its own default (currently "high").
    claude_code_effort: str | None = "medium"
    # Sized for the largest oversized-section chunks (never split further,
    # so a protected table/template can push a single chunk to 10k+ input
    # tokens). Confirmed directly against the raw API (bypassing this
    # harness entirely) that a real 12k-input/7k-output completion took
    # 375s end to end — genuine model inference latency, not a stuck
    # connection. 120s, then 300s, both proved short; 600s leaves real
    # headroom above the slowest observed case.
    request_timeout_s: float = 600.0

    # OpenCode Go (github.com/sst/opencode's Go CLI, provider: opencode_go) —
    # runs `opencode run` under whichever model/provider the caller's own
    # opencode config already has authenticated, independent of the Claude
    # Code CLI's own subscription/session. This is what pipeline.py's
    # ensure_fallback_engine() switches to by default when `provider` is
    # claude_code and it hits a session/rate limit -- see
    # claude_code_client.py's ClaudeCodeSessionLimitError and
    # opencode_go_client.py. No API key needed here either. `model` of
    # "auto" (config.default_model_for_provider's pick for this provider)
    # means: don't pass --model at all, let opencode use its own configured
    # default provider/model.
    opencode_go_cli_path: str = "opencode"
    # `opencode run` has no `--tools ""` equivalent the way Claude Code's CLI
    # does -- confirmed live that a normal opencode install's default "build"
    # agent can carry full bash/file/network tool permissions. Since chunk
    # text here is untrusted external wiki content, set this to the name of
    # a locked-down agent (passed straight to `run --agent`) to disable tool
    # use for this engine specifically. Define one in opencode's own config
    # (~/.config/opencode/opencode.jsonc, or a project-local opencode.jsonc
    # -- `opencode agent create` is an interactive-only wizard, not
    # scriptable) with every tool set to false, e.g.:
    #   "agent": {
    #     "wiki-translation-harness": {
    #       "mode": "all",
    #       "tools": {"bash": false, "edit": false, "webfetch": false,
    #                 "read": false, "write": false, "glob": false,
    #                 "grep": false, "task": false, "todowrite": false,
    #                 "websearch": false, "lsp": false, "skill": false}
    #     }
    #   }
    # then set opencode_go_agent: wiki-translation-harness. None (the
    # default) leaves whichever tool permissions opencode's own config
    # already grants its default agent untouched.
    opencode_go_agent: str | None = None

    # Wikimedia's User-Agent policy (foundation.wikimedia.org/wiki/Policy:User-Agent_policy)
    # requires automated requests to self-identify with a contact (email or URL) so the
    # operator can be reached — unidentified bulk traffic risks throttling/blocking.
    # wikimedia_contact has no default on purpose; config.py refuses to build a Config
    # without it (or an explicit user_agent override) rather than send anonymous traffic.
    wikimedia_tool_name: str = "wiki-translation-harness"
    wikimedia_contact: str | None = None
    user_agent: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        if v not in _VALID_PROVIDERS:
            raise ValueError(
                "provider must be 'openrouter', 'local', 'claude_code', 'experiential', "
                f"or 'opencode_go' or 'codex', got {v!r}"
            )
        return v

    @field_validator("fallback_provider")
    @classmethod
    def _validate_fallback_provider(cls, v: str | None) -> str | None:
        if v is not None and v not in _VALID_PROVIDERS:
            raise ValueError(
                "fallback_provider must be 'openrouter', 'local', 'claude_code', "
                f"'experiential', or 'opencode_go' or 'codex', got {v!r}"
            )
        return v

    @field_validator("complex_provider", "review_provider")
    @classmethod
    def _validate_secondary_provider(cls, v: str | None) -> str | None:
        if v is not None and v not in _VALID_PROVIDERS:
            raise ValueError(
                "complex_provider/review_provider must be 'openrouter', 'local', "
                f"'claude_code', 'experiential', or 'opencode_go' or 'codex', got {v!r}"
            )
        return v

    @field_validator("claude_code_effort")
    @classmethod
    def _validate_claude_code_effort(cls, v: str | None) -> str | None:
        if v is not None and v not in _VALID_EFFORT_LEVELS:
            raise ValueError(
                f"claude_code_effort must be one of {sorted(_VALID_EFFORT_LEVELS)} or None, got {v!r}"
            )
        return v

    @field_validator(
        "output_dir",
        "cache_db_path",
        "log_dir",
        "stats_path",
        "partial_output_dir",
        "verification_db_path",
        mode="before",
    )
    @classmethod
    def _expand_user(cls, v: object) -> object:
        if isinstance(v, (str, Path)):
            return Path(v).expanduser()
        return v

    @field_validator("skill_path", "qa_skill_path", mode="before")
    @classmethod
    def _expand_user_skill_path(cls, v: object) -> object:
        if v is None:
            return v
        if isinstance(v, (str, Path)):
            return Path(v).expanduser()
        if isinstance(v, (list, tuple)):
            return [Path(p).expanduser() for p in v]
        return v
