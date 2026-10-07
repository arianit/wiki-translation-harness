# wiki-translation-harness

Batch tool that translates Wikipedia articles into wikitext for another
language edition. Built and tested for English → Albanian (sq.wikipedia).

The harness contains no translation prompts. All translation judgment comes
from the
[enwiki-sqwiki-translation](https://github.com/arianit/enwiki-sqwiki-translation)
skill. The harness only fetches, splits, calls the model, validates,
repairs, caches, verifies facts, and saves. It never publishes anything.

## How it works

![Translation pipeline in three stages: prepare (fetch, verify facts, split), per chunk (cache, translate, validate, repair), and whole article (assemble, fix and validate, repair or withhold for human review, optional semantic review, write output and report).](docs/architecture.svg)

1. Fetch the article and split it into chunks.
2. Look up link targets, templates and infobox parameters on Wikidata and
   the target wiki (see **Fact verification**).
3. Translate each chunk (in parallel, up to `workers`). Chunks already in
   the translation-memory cache are reused.
4. Validate each chunk. On a defect, ask the model to repair it; if repair
   fails, the article fails and is not saved (partial progress is kept).
   Before starting new translations, the harness retries unfinished articles
   from `partial_output_dir`, one at a time, reusing matching cached sections.
   Each unfinished article gets one attempt per run before new work begins,
   including with `--no-sequential`. Queue mode retries this local backlog
   before claiming new articles, counting retries toward `--max-articles`.
   New checkpoints preserve source language and local input paths; older
   partial drafts use a matching requested input or infer the title from the
   filename with the configured source language.
5. Assemble the article, apply deterministic fixes, validate it again
   (statically and by rendering it through the target wiki's parse API),
   and write the `.wiki` file plus a report.

### How the skill is used

A single chat-completion call has no tools or internet, so the harness:

- Loads the skill's `SKILL.md` files from disk and uses them as the system
  prompt, with a short fixed frame saying "no tools in this call" (see
  `skill_loader.py`). By default the skill is read from a pinned git
  revision (`skill_git_ref: HEAD`) so uncommitted edits don't change a
  running batch.
- Sends `skill_path` (translation + `wikiterms`) on every call, and
  `qa_skill_path` (`wikiqa`) only on repair calls, since that checklist is
  only useful once a defect was found. This cuts the system prompt on a
  normal call by about 29%.
- Does the skill's live lookups itself and passes the results to the model
  as plain facts.
- Keeps a per-article terminology registry: when a chunk renders an
  unlinked term as `{{ill|...}}`, later chunks mentioning the same term are
  told to reuse that exact rendering.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp config.example.yaml config.yaml
export WIKIMEDIA_CONTACT=you@example.com   # required, or set wikimedia_contact in config.yaml
# Only for the API-key engines:
# export OPENROUTER_API_KEY=sk-or-...
# export EXPLABS_API_KEY=xpl_...
```

`wikimedia_contact` is required by Wikimedia's
[User-Agent policy](https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy).
The harness refuses to start without it (or an explicit `user_agent`).

## Usage examples

All commands are run from the repo folder, with the virtual environment
from **Setup** active. With the default config, each article is translated
from English Wikipedia into Albanian using Claude Code.

### Translate one article

Pass the article's title exactly as it appears on English Wikipedia (the
part after `/wiki/` in the address, with spaces instead of underscores):

```bash
wiki-translation-harness --title "Dresnik archaeological site"
```

When it finishes you get `Dresnik_archaeological_site.wiki` (the Albanian wikitext,
ready to paste into sq.wikipedia) and `Dresnik_archaeological_site.report.md` (what
was checked and what to review) in the output folder (see **Output**).

### Translate several articles

Put one title per line in a text file. Empty lines and lines starting with
`#` are ignored:

```text
# articles.txt
Drini Bridge
Cërmjan
https://en.wikipedia.org/wiki/Paleokastra_Castle
```

```bash
wiki-translation-harness --titles articles.txt
```

Articles are translated one at a time by default, so if one fails the
others are not affected.

### Translate every article in a category

```bash
wiki-translation-harness --category "Archaeological sites in Kosovo"
```

This translates the articles directly in that English Wikipedia category
(not its subcategories). Categories can be large, so check the category
page first. Articles that already exist on sq.wikipedia are translated
too; their report marks them as a REWRITE.

### Translate from a language other than English

Add a language code before the title, or paste the full Wikipedia link:

```bash
wiki-translation-harness --title "de:Kalivo"
wiki-translation-harness --title "https://de.wikipedia.org/wiki/Konispol-Höhle"
```

Without a prefix, `source_lang` from config.yaml is used (default `en`).
The output language is always `target_lang` (default `sq`).

### Translate wikitext you already have

If you copied an article's source (from "Edit source" on Wikipedia) into a
`.wiki` or `.txt` file, translate that file, or a whole folder of them:

```bash
wiki-translation-harness --file Drini_Bridge.wiki
wiki-translation-harness --directory my_articles/
```

The file name (minus the extension) is used as the article title.

### Choose a different model or engine

```bash
wiki-translation-harness --title "Paleokastra Castle" --provider openrouter --model deepseek/deepseek-v3.2
```

See **Engines** for the options.

### If a run stops halfway

Run the same command again. Finished articles are skipped, and sections
already translated are reused from the cache
(`cache/translation_memory.sqlite3`), so you don't pay for them twice. To
translate an article again from scratch, add `--force`.

### Other flags

`--workers` (how many sections are translated at once),
`--no-sequential` (start all articles at once), and `--no-cache` /
`--no-validate` / `--no-repair` / `--no-live-validate` to turn off
individual steps. Run `wiki-translation-harness --help` for the full list.

## Output

Written to `output_dir` (default `~/code/wiki-translation-queue/output`;
set it to `output` for a local folder):

- `Article_Name.wiki`: the translated wikitext.
- `Article_Name.report.md`: link/template verification tables, infobox
  parameters, citation languages, which sections needed repair, a REWRITE
  flag if the target article already exists, and (for English sources) a
  ready-to-paste `{{Përkthyer nga}}` Talk-page block and edit summary.
- `Article_Name.review-flags.md`: only if the semantic review pass left
  unresolved findings (see **Model tiers**).
- `needs_human_review.json`: articles withheld because a structural defect
  could not be repaired.

Also: `logs/run.log`, `logs/errors.log`, and `stats.json` (live counters,
including tokens and cost per model).

## Engines

Choose with `--provider` or `provider:` in config.yaml:

| Provider | What it runs | Key needed | Cost reporting |
|---|---|---|---|
| `claude_code` (default) | `claude -p` with your Claude Code login | no | always $0.00 (not wired up yet) |
| `openrouter` | OpenRouter API | `OPENROUTER_API_KEY` | from OpenRouter's pricing table |
| `local` | any OpenAI-compatible server (llama.cpp, Ollama, LM Studio, vLLM) | no | $0.00 |
| `experiential` | [Experiential Labs](https://platform.experientiallabs.ai/) (OpenAI-compatible) | `EXPLABS_API_KEY` | real per-call cost |
| `opencode_go` | `opencode run` with its own login | no | real per-call tokens and cost |

If you change `--provider` without `--model`, a default model for that
provider is picked (`claude-sonnet-5-5`, `deepseek/deepseek-v3.2`,
`qwen3.8-27b`, or, for `opencode_go`, whatever opencode is configured to
use).

Caveat for `opencode_go`: `opencode run` has no way to turn tools off, so
pin `opencode_go_agent` to a locked-down agent (see `config.example.yaml`).

Example for a local llama.cpp server:

```bash
wiki-translation-harness --title "Drini Bridge" --provider local \
  --base-url http://127.0.0.1:8080/v1 --model qwen3-8b-q5-k-m
```

or in config.yaml:

```yaml
provider: local
local_base_url: http://127.0.0.1:8080/v1
local_model: qwen3-8b-q5-k-m
```

To add an engine, add a branch to `build_llm_client()` in `engines.py` and
a client implementing `chat_completion()`, `get_pricing_for()`,
`fetch_pricing()` and `aclose()`. Nothing else needs to change.

### Fallback when credits run out

If the provider reports insufficient credits (OpenRouter HTTP 402,
Experiential Labs HTTP 429 with `code: insufficient_quota`, or Claude Code
hitting its session/spend limit), the harness offers to switch engines for
the rest of the run. It asks interactively, or switches automatically in
`queue` mode.

The target is `--fallback-provider`. By default it is `opencode_go` when
running on `claude_code`, otherwise `claude_code`. Set it equal to
`--provider` to disable the switch.

To automatically continue with an efficient OpenAI subscription model after Claude's
session limit, set these in `config.yaml`:

```yaml
fallback_provider: codex
fallback_model: gpt-6-luna
fallback_auto_switch: true
```

Sign in using `codex login` with your ChatGPT Pro account. The harness
checks that login is ChatGPT-based and refuses API-key authentication.
GPT-6 Luna is OpenAI's efficient model for focused, high-volume tasks;
Albanian translations still need human review. Existing validation and
repair run as before. The Codex engine uses an isolated temporary working
directory and a read-only sandbox, with shell tools disabled.

`fallback_model` can pin another model; omitted, the target provider's
default is used. `fallback_auto_switch` defaults to false, preserving the
interactive prompt. Explicit complex/review tiers retain their own routing.
You can also use `--provider codex --model gpt-6-luna` directly.
If the OpenAI subscription is also exhausted, the chunk fails rather than
silently switching to paid API usage.

## Model tiers

All optional. With none set, one model does everything.

- **Draft** (`model`, `provider`): ordinary text chunks.
- **Complex** (`complex_model`, `complex_provider`): infoboxes, large
  tables and dense reference lists go here instead.
- **Review** (`review_model`, `review_provider`): after the article passes
  structural checks, this model reads the whole translation next to the
  source and looks for mistranslation, missing or invented facts, grammar
  errors, and inconsistent names across sections. Findings are repaired up
  to `review_max_repair_attempts` times. Unlike structural defects,
  unresolved review findings do **not** block the output; they go into
  `.review-flags.md` and the report.

`review_model` defaults to `complex_model`, so setting a stronger complex
model also turns on review. Providers inherit the same way (review →
complex → main).

## Queue mode

To translate "the next article" from the shared list in
[wiki-translate-queue](https://github.com/arianit/wiki-translate-queue)
(`totranslate.txt`) instead of naming titles:

```bash
wiki-translation-harness queue                     # up to 10 articles
wiki-translation-harness queue --max-articles 1
wiki-translation-harness queue --provider openrouter --model deepseek/deepseek-v3.2
```

Each run pulls the queue repo, claims the first unclaimed line (commits and
pushes the claim first, so two machines don't take the same article),
translates it, then marks it `DONE` or `FAILED` along with the engine that
actually ran (e.g. `DONE\tclaude-sonnet-5-5@claude_code`). A claim older than
`--stale-hours` (default 3) is treated as abandoned. The clone location is
`--queue-repo-dir` (default `~/code/wiki-translation-queue`).

## Fact verification

Done by the harness in `verification.py` and passed to the model as data,
not instructions. Cached in `cache/verified_facts.sqlite3`.

- **Links**: checked against Wikidata sitelinks. For terms with no
  target-wiki article, the harness counts how many other languages have
  one, as a hint for whether the term is worth an interwiki link.
- **Templates**: checked for existence; for infoboxes, the target wiki's
  real parameter names are fetched (many sq.wikipedia infoboxes keep English
  parameter names, and models tend to invent translated ones).
- **Existing article**: if the article already exists on the target wiki,
  the run is flagged as a rewrite and that article's links are passed as
  established terminology.
- **Citation parameters**: the model is told CS1 templates (`{{cite web}}`
  etc.) keep English parameter names on every wiki.

## Deterministic fixes

Applied after translation, in `citation_language.py`:

- **Citation language**: adds a missing `|language=`, guessed from the
  title or read from the cited page. The title wins if they disagree.
- **Citation parameter names**: renames mistranslated CS1 names back to
  English (`|titulli=` → `|title=`, `|botues=` → `|publisher=`, ...).
- **Sfn/harvnb page parameters**: `|f=`/`|ff=` or a positional `f. 161`
  become `|p=`/`|pp=`.
- **Redundant `<ref>` around `{{sfn}}`**: `{{sfn}}`, `{{sfnp}}` and
  `{{sfnm}}` create their own `<ref>`, so an extra wrapper nests refs and
  breaks Cite on sq.wikipedia. Only bare wrappers are removed;
  `{{harvnb}}` is left alone.
- **Short-footnote dedup**: identical `{{sfn}}` citations translated in
  different chunks can come back with slightly different `|ps=` text, which
  breaks the shared anchor. All copies are made identical to the first.

Toggles: `fill_citation_languages`, `fix_citation_param_names`,
`dedupe_short_footnotes`, `verify_links`. The two sfn fixes always run.

Validation also catches leaked commentary (the model breaking character or
inventing content for a near-empty chunk) and treats it as a defect to
repair.

## Benchmark mode

Translate one article with several models and compare runtime, tokens and
cost. It uses `provider` from config.yaml (there is no `--provider` flag
here):

```bash
wiki-translation-harness benchmark --title "Dresnik archaeological site" \
  --model deepseek/deepseek-v3.2 --model google/gemini-2.5-flash \
  --judge-model anthropic/claude-sonnet-4.5
```

Output goes to `quality/` (`--output` to change). With `--judge-model`
(which must not be one of the compared models), a separate model receives
the source and the translations labeled A, B, C... in random order, scores
each 1 to 10 on five criteria, and ranks them. Results go to
`quality/Article_Name/` (`comparison.md`, `evaluation/`, `mapping.json`).
`--no-evaluation` skips the judge.

Example results for "Enji (deity)" are in
[issue #1](https://github.com/arianit/wiki-translation-harness/issues/1):
`deepseek/deepseek-v3.2` gave the best quality for the cost;
`google/gemini-2.5-flash` matched its quality and was about 3× faster at
twice the price.

### September 2026 re-run: Enji (deity)

The same article re-benchmarked on 2026-09-30 with current models, run through
three engines (`claude_code`, `opencode_go`, `openrouter`) and judged blind by
**two** independent judges, Grok 4.7 (`opencode-go/grok-4.7`) and GPT-6 Sol
(`openai/gpt-6-sol`), each seeing the four translations under a different
random A-D labelling. Per-chunk translation only: no complex/review tier, no
fallback provider, translation cache off, `opencode_go` pinned to a
tools-disabled agent.

| Model | Engine | Grok 4.7 | GPT-6 Sol | Runtime | Tokens in / out | Cost |
|---|---|---|---|---|---|---|
| **deepseek-v4.1-flash** | opencode_go | **7** | **7** | 543 s | 265k / 23k | $0.12 |
| deepseek-v4-pro | opencode_go | 6 | 6 | 912 s | 301k / 21k | $0.54 |
| claude-sonnet-5-5 | claude_code | 6 | 5 | 148 s | 460k / 31k | not reported |
| qwen3.6-27b (reasoning off) | openrouter | 3 | 2 | 430 s | 287k / 26k | $0.09 |
| glm-5.3 | opencode_go | failed | failed | - | - | - |

Both judges produced the same ranking: **deepseek-v4.1-flash > deepseek-v4-pro >
claude-sonnet-5-5 > qwen3.6-27b**. Per-criterion scores (accuracy / Albanian /
terminology / MediaWiki / publication readiness):

| Model | Grok 4.7 | GPT-6 Sol |
|---|---|---|
| deepseek-v4.1-flash | 8 / 8 / 8 / 7 / 7 | 8 / 7 / 7 / 7 / 7 |
| deepseek-v4-pro | 6 / 8 / 8 / 5 / 5 | 7 / 8 / 8 / 5 / 6 |
| claude-sonnet-5-5 | 6 / 7 / 6 / 7 / 5 | 7 / 5 / 5 / 7 / 5 |
| qwen3.6-27b | 3 / 3 / 3 / 4 / 2 | 4 / 2 / 2 / 3 / 2 |

Judge cost: $0.27 (Grok) and $0.26 (GPT-6 Sol). What the judges said:
deepseek-v4.1-flash kept the full lead and translated the quotations; its
remaining problems were copyediting and template cleanup. deepseek-v4-pro had
the best prose but truncated the lead and split the bibliography. Sonnet left
two long quotations in English and rendered `drangue` as `dragua`. Qwen had
repeated sense errors, garbled word forms, and stray Cyrillic characters.

Caveats: one article, and the gap between the top three is small. Sonnet ran
without the repair/review passes the full pipeline adds, so this understates it.
**glm-5.3** never produced output through OpenCode Go: on the ~18k-token skill
prompt it stalls or reasons for many minutes, and it tries to call tools (the
skill text reads like agent instructions), which the locked-down agent blocks.
**qwen3.6-27b** only completes with `disable_reasoning: true`; by default it
spends its whole output budget on hidden reasoning (9,000 of 9,000 tokens, no
content) and the request times out and retries. The two judges were run by a
one-off script that merges the per-engine results into one blind pass (the
`benchmark` command takes a single `--provider` and the built-in judge handles
at most four translations through OpenRouter only). Raw outputs are under
`quality/2026-09-30/` (gitignored).

## Reliability

Every network call has a hard `asyncio.wait_for` deadline on top of the
HTTP client's own timeout, because httpx's timeout alone did not always
fire and could hang a whole batch. CLI-based engines are bounded by
`request_timeout_s`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Known limitations

- Some of the skill's live-research steps are not replicated (terminology
  searches in nearby sq.wikipedia articles, category-name translation);
  those stay the model's own judgment.
- The parameter-name fixes only cover patterns seen so far. A new
  mistranslated name won't be caught until added to
  `ALBANIAN_TO_ENGLISH_CS1_PARAMS`.

## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE). The `workshop/` directory keeps
its own licence (CC BY-SA 4.0, see `workshop/LICENSE`).
