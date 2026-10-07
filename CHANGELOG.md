# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
- **Review pass no longer undoes post-processing**: after a review-driven repair, `run_review_pass` rebuilt the article with plain `assemble_chunks()`, so the saved file lost every `_post_process` fix (filled `|language=`, sfn renames, restored refs, reference-list repairs). The fixes now live in module-level `pipeline.post_process_assembled`, which both loops call after reassembling, along with comment reconciliation; comments removed during review are added to the report.
- **Reference lists invented at chunk boundaries removed** (`pipeline.py` `_post_process`): when a list-defined `<references>` block is split across chunks, the model ends each later piece with its own `<references />` / `<references group="…"/>`, so every definition after the first chunk falls outside the block. `restore_ref_group_names` maps a translated group name back to the source's (`group="shënim"` → `note`), `drop_stray_reference_lists` keeps only as many renderers per group as the source has, and `extend_references_block_over_trailing_defs` moves `</references>` past the definitions (and model comments) left trailing after it. Albert Einstein: 96 live Cite findings → 6, all of them real dropped citations.
- **Split `<references>` block rejoined**: `pipeline.merge_split_references_blocks` (in `_post_process`) drops the spurious first `</references>` when chunking cut a list-defined `<references>` block in two, which had produced dozens of "no text given" / "not used" Cite errors that per-chunk repair could not fix (Alexander the Great: 58 live findings → 38).
- **Bundled `<ref>{{Sfn}}, {{Sfn}}</ref>` no longer nests refs**: `citation_language.convert_nested_sfn_to_harvnb` renames `{{sfn}}`/`{{sfnp}}` to `{{harvnb}}`/`{{harvp}}` inside any `<ref>`, restoring the source's form (the skill's "bundled harvnb → consecutive Sfn" row made the model swap templates but keep the wrapper). Cleared 15 `FOOTNOTE…` errors on Alexander the Great.
- **New deterministic post-processing fix for a live sq.wikipedia Cite bug**: `citation_language.unwrap_redundant_sfn_ref`, wired into `pipeline.py`'s `_post_process`, strips a redundant `<ref>...</ref>` wrapped directly around a lone `{{sfn}}`/`{{sfnp}}`/`{{sfnm}}` call. `{{sfn}}` already expands to its own `<ref name="FOOTNOTE...">...</ref>` internally, so an extra outer `<ref>` nests one ref inside another — confirmed live against sq.wikipedia (2026-09-22, `Conservatism`) that this corrupts Cite's usage-tracking for the auto-generated name, rendering a live "defined in `<references>` but not used in prior text" error even for that name's only occurrence in the article. The previous behavior (relying on the LLM-based assembly-repair loop) made things worse: since the auto-generated ref name never appears literally in the wikitext, the repair loop couldn't localize the fix to one chunk and broadcast it to all of them, escalating a single cosmetic double-wrap into a genuine "defined multiple times with different content" collision after a few rounds. Only unwraps a *bare* `<ref>` (no `name=`) whose entire content is that one template call — a named ref or one with extra prose alongside is left alone. Does not touch `{{harvnb}}`/`{{harv}}`/`{{harvp}}`, which don't self-wrap and are routinely (correctly) wrapped in an explicit `<ref>` by design.

### Added
- **HTML comments reconciled with the source** (`comments.reconcile_comments`, run per chunk before every assembly in `run_assembly_repair`): comments carried over from enwiki are labelled `<!-- Nga en.wiki: … -->` so sqwiki editors don't read an enwiki editor's note ("don't change without consensus on the talk page") as their own wiki's; comments the model wrote itself are removed and listed in the report under "Model notes removed from the article". Matched in order by relative position and untranslated tokens (URLs, numbers, names, markup). On existing outputs it removed exactly the three model notes (Albert Einstein's repair-loop "Nuk u identifikuan me siguri ndryshimet…", Napoleon's two) and labelled the other 63 comments across eight articles.
- **`{{Cref2}}`/`{{Cnote2}}` expansion before chunking** (`notes.expand_cref_notes`, called from `_plan_article`): each marker becomes a self-contained `<ref name="cnote-x" group="G">note</ref>` (reuse → self-closing ref), the `{{Cnote2 Begin}}…{{Cnote2 End}}` block becomes `{{Reflist|group=G}}`, with G from `|liststyle=` (default `upper-alpha`, which sqwiki renders as letters). Inner `<ref>`s in a note are moved out after it, and `{{sfn}}` in a note becomes `{{harvnb}}`. Previously the marker and its text landed in different chunks and the model dropped markers and left a stray `<ref group>` with no `<references>`.
- **`disable_reasoning` config flag** (default off): sends `reasoning: {"enabled": false}` on OpenRouter calls. Reasoning-by-default models such as `qwen/qwen3.6-27b` otherwise burn the entire output budget on hidden thinking (measured: 9,000/9,000 tokens, zero content, ~255s per call; with it off, the same call finished in 82s), which runs past `request_timeout_s` and retries indefinitely since no `max_tokens` is sent.

### Changed
- **Assembly repairs run in parallel**: a round's chunk repairs now run concurrently, up to `assembly_repair_concurrency` (default 4), instead of one at a time — a 19-chunk round on Albert Einstein took 15 minutes serially.
- `skill_path` now accepts a single directory or a list of directories, to match the upstream skill being split into `enwiki-sqwiki-translation` (translation), `wikiterms`, and `wikiqa`. `skill_loader.load_skill`/`load_skill_cached` concatenate each directory's `SKILL.md` (and, if `include_skill_references` is set, its `references/*.md`) in the given order; reference filenames are namespaced by source skill directory only when more than one path is loaded, so existing single-path configs are unaffected. `config.example.yaml`'s default now lists all three directories.
- **`wikiqa` (the pre-delivery QA checklist) no longer rides along on every normal translation/repair call.** It's now loaded separately via a new `qa_skill_path` config field and appended (`skill_loader.build_repair_messages`'s new `qa_skill` parameter) only to a repair call — the one case where `validate_wikitext` has already found a real defect. `skill_path`'s default dropped `wikiqa`, keeping only `enwiki-sqwiki-translation` + `wikiterms`. Measured against the real skill files: system-prompt tokens on an ordinary translation call drop ~29% (32.4k → 22.9k, tiktoken cl100k_base estimate); a live `claude_code`-provider translation of a real enwiki lead section (Sustainable architecture) showed the same shape live (54.9k → 40.9k input tokens per the CLI's own usage reporting) with both before/after outputs passing static validation and reading as equally fluent, faithful Albanian. `translate_chunk`/`repair_chunk`/`run_assembly_repair`/benchmark mode all thread the new `qa_skill` parameter through; existing single-skill-path configs without `wikiqa` are unaffected.
- **New compact, growing article-level terminology registry** (`VerifiedFacts.established_renderings` in `verification.py`) narrows one remaining cross-section consistency gap: for a source-language term with no confirmed target-wiki sitelink, the first chunk to translate it typically renders it as `{{ill|display|en|Title}}` — that specific rendering is now recorded (first occurrence wins, mechanically parsed from the model's own output, no translation judgment invented by the harness) and surfaced in `build_verified_facts_block` to every later chunk of the same article mentioning the same term ("already rendered elsewhere in this article as ... — reuse that exact rendering"), instead of each chunk independently reinventing its own phrasing. Confirmed against a real model output using this exact pattern (`{{ill|Shtëpia me energji pozitive|en|Energy-plus-house|lt=...}}`).

## [0.2.0] - 2026-08-06

### Added
- Blind evaluation with judge model in benchmark mode (`--judge-model`)
- New `evaluation.py` module for judge-model quality assessment
- Randomized labeling (A-D) with structured JSON output
- Five evaluation criteria: translation accuracy, Albanian language quality, terminology quality, MediaWiki quality, publication readiness
- Integration with existing benchmark pipeline
- Example benchmark results for "Enji (deity)" article

### Changed
- Updated `benchmark.py` to return translations and support evaluation
- Updated `cli.py` with `--judge-model` and `--no-evaluation` options
- Enhanced README with example benchmark and usage instructions

### Removed
- Test files `albanian_mythology`, `albanian_mythology.missing`, `filter_existing.py` from repository (moved to .gitignore)

## [0.1.0] - 2026-08-04

### Added
- Initial release of wiki-translation-harness
- Batch translation of Wikipedia articles via OpenRouter
- Delegation to enwiki-sqwiki-translation Pi skill
- Translation memory caching
- Fact verification (Wikidata + target wiki)
- Post-processing fixes (citation language, parameter names, short-footnote dedup)
- Report generation
- Benchmark mode for comparing multiple models