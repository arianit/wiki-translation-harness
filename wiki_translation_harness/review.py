"""Independent semantic-fidelity review of an already-assembled, structurally
valid translation.

This is a genuine second opinion, not a self-check: unlike wikiqa (the same
model/session grepping its own output for known defect patterns), the
review model here has no memory of having produced the translation and
reads the whole English source and the whole Albanian translation together,
for the first time, specifically to catch what a translator working section
by section cannot self-audit -- mistranslation, hallucinated or dropped
facts, target-language grammar errors, and cross-article transliteration
inconsistency. It is also not evaluation.py's judge: that ranks several
candidate translations against each other with holistic 1-10 scores; this
reads one translation and returns concrete, locatable findings a repair
call can act on.
"""

from __future__ import annotations

import json
import logging
import re

from wiki_translation_harness.engines import LLMEngineClient
from wiki_translation_harness.models import ModelPricing, TranslationResult, ValidationIssue
from wiki_translation_harness.openrouter import RetryCallback, run_completion

REVIEW_CRITERIA = """You are an independent quality reviewer for a Wikipedia article translation. You did not write this translation -- you are reading it for the first time, side by side with its English source, specifically to catch problems the translator itself would not notice in its own output.

You will be given the full English source wikitext and the full translated wikitext for the same article. Find concrete, locatable defects. Do not score or rank anything, and do not comment on style preferences that don't change meaning or correctness.

Check for:

1. Fidelity -- any claim, number, date, or name in the translation that is not traceable to the English source (a hallucination), and any fact present in the source but dropped from the translation.

2. Target-language grammar -- case/declension, gender, and verb agreement errors, especially in terse infobox/table fields where a wrong case is easy to miss.

3. Named-entity and transliteration consistency across the WHOLE article -- the same source name or term rendered differently in different parts of the translation. This is the one class of error a translator working section by section cannot self-audit; you are seeing the complete assembled article, so this check is yours specifically to make.

4. Infobox/template parameter correctness -- if a list of confirmed real parameter names for a template used in this article is given below, check the translation didn't invent, mistranslate, or drop a parameter name (an unrecognized parameter name is silently ignored by the wiki software, which drops that field from the rendered page with no visible error).

Do NOT report any of the following -- they are already handled automatically elsewhere in this pipeline before you ever see this text, so flagging them again just wastes a repair round:
- <ref>{{sfn}}</ref> double-wrapping
- Citations missing |language=
- Citation or {{sfn}}/{{harvnb}} parameter names mistranslated into the target language

## Output format

Respond with ONLY a JSON array (inside a ```json code block), one object per finding, in this exact shape:

```json
[
  {
    "kind": "semantic_fidelity",
    "severity": "error",
    "message": "Explanation of the specific problem.",
    "snippet": "A short, EXACT, verbatim excerpt copied from the TRANSLATED wikitext that contains the problem -- must match the translated text character for character so it can be located automatically."
  }
]
```

`kind` must be one of: "semantic_fidelity", "grammar_case", "entity_consistency", "infobox_param_mismatch". `severity` is "error" or "warning". `snippet` must be copied verbatim from the translated wikitext, not paraphrased or re-typed, and should be short (a phrase or sentence, not a whole paragraph) so it uniquely locates the problem.

If the translation has no defects worth reporting, respond with an empty array:

```json
[]
```
"""

logger = logging.getLogger(__name__)

_VALID_REVIEW_KINDS = frozenset(
    {"semantic_fidelity", "grammar_case", "entity_consistency", "infobox_param_mismatch"}
)

_JSON_ARRAY_RE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)


def build_review_messages(
    source_wikitext: str,
    translated_wikitext: str,
    article_title: str,
    target_lang: str,
    template_params: dict[str, list[str]],
) -> list[dict[str, str]]:
    params_block = ""
    if template_params:
        lines = [
            f"- {{{{{name}}}}}: " + ", ".join(params)
            for name, params in sorted(template_params.items())
        ]
        params_block = (
            "\n\nConfirmed real infobox/template parameter names on the target wiki "
            "(use these to check the translation didn't invent or mistranslate a "
            "parameter name):\n" + "\n".join(lines)
        )
    user = (
        f"Article title: {article_title}\n"
        f"Target language: {target_lang}"
        f"{params_block}\n\n"
        f"--- BEGIN ENGLISH SOURCE ---\n{source_wikitext}\n--- END ENGLISH SOURCE ---\n\n"
        f"--- BEGIN TRANSLATED WIKITEXT (target language: {target_lang}) ---\n"
        f"{translated_wikitext}\n--- END TRANSLATED WIKITEXT ---"
    )
    return [
        {"role": "system", "content": REVIEW_CRITERIA},
        {"role": "user", "content": user},
    ]


def _extract_json_array(text: str) -> list | None:
    """Same two-step strategy as evaluation.extract_json_block (fenced code
    block first, then a naive first-'['-to-last-']' fallback), just for a
    top-level JSON array instead of an object -- the two shapes don't share
    a regex, so this isn't a literal reuse of that function."""
    match = _JSON_ARRAY_RE.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def parse_review_findings(text: str) -> list[ValidationIssue]:
    data = _extract_json_array(text)
    if data is None:
        # Indistinguishable from a clean review to the caller, so at least
        # leave a trail: the article was not actually reviewed.
        logger.warning("Review response had no parseable JSON findings array: %r", text[:300])
        return []
    if not data:
        return []
    issues: list[ValidationIssue] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        message = str(entry.get("message") or "").strip()
        if not message:
            continue
        kind = str(entry.get("kind") or "semantic_fidelity").strip()
        if kind not in _VALID_REVIEW_KINDS:
            kind = "semantic_fidelity"
        severity = str(entry.get("severity") or "error").strip().lower()
        if severity not in ("error", "warning"):
            severity = "error"
        snippet = entry.get("snippet")
        snippet = str(snippet).strip() if snippet else None
        issues.append(ValidationIssue(kind=kind, message=message, severity=severity, snippet=snippet))
    return issues


async def review_article(
    client: LLMEngineClient,
    model: str,
    temperature: float,
    source_wikitext: str,
    translated_wikitext: str,
    article_title: str,
    target_lang: str,
    template_params: dict[str, list[str]],
    pricing: ModelPricing | None,
    on_retry: RetryCallback | None = None,
) -> tuple[list[ValidationIssue], TranslationResult]:
    """One review call over the whole assembled article. Returns (findings,
    the raw TranslationResult) -- the caller (pipeline.run_review_pass)
    accumulates cost/token stats from the latter itself, the same way it
    already does for repair_chunk's TranslationResult."""
    messages = build_review_messages(
        source_wikitext, translated_wikitext, article_title, target_lang, template_params
    )
    result = await run_completion(client, model, messages, temperature, pricing, on_retry=on_retry)
    findings = parse_review_findings(result.text)
    return findings, result
