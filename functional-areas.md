# wiki-translation-harness: functional areas

Based on the README and module names; not every module was read.

## 1. Orchestration and entry points
- `cli.py`: command line.
- `pipeline.py`: runs the stages from fetch to output.
- `queue_runner.py`, `review_queue.py`, `resume.py`: batch queues, review backlog, retrying unfinished articles.
- `config.py`, `logging_setup.py`, `progress.py`, `statistics.py`: settings, logs, run progress.

## 2. Source retrieval and parsing
- `mediawiki.py`: Wikipedia API access.
- `sources.py`: loads input from a title or a local file.
- `parser.py`: splits an article into chunks.
- `comments.py`, `notes.py`: wikitext comments and translator notes.

## 3. Fact verification
- `verification.py`: looks up link targets, templates and infobox parameters on Wikidata and the target wiki; results go to the model as plain facts.
- `citation_language.py`: works out and marks the source language of each citation.

## 4. Translation engines
- `translator.py`: builds and sends translation calls.
- `engines.py`: selects the engine.
- `claude_code_client.py`, `codex_client.py`, `openrouter.py`, `opencode_go_client.py`: one client per backend.
- `skill_loader.py`: loads the external enwiki-sqwiki-translation skill as the system prompt. The harness holds no translation prompts.

## 5. Validation and repair
- `validator.py`: static checks on chunks and the whole article.
- `live_validator.py`: renders the result through the target wiki's parse API.
- `repair.py`: asks the model to fix a defect.
- `review.py`: optional semantic review.

## 6. Caching
- `cache.py`: translation-memory cache; partial drafts are kept so a failed run can resume.

## 7. Output and reporting
- `output.py`: writes `.wiki` files.
- `report.py`: writes the per-article report.
- A failed article is withheld for human review. Nothing is ever published.

## 8. Evaluation tooling
- `evaluation.py`, `benchmark.py`, `models.py`: compare engines and models; shared data types.

## Other
- `tests/`: one test file per module, plus fixtures.
- `workshop/`: walkthrough notebook.
- `docs/architecture.svg`: pipeline diagram.
