"""Judge-model evaluation of translation quality."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wiki_translation_harness.openrouter import OpenRouterClient, OpenRouterError

EVALUATION_CRITERIA = """
You are an expert Albanian Wikipedia editor and translator evaluator.
Your task is to evaluate the quality of several Albanian translations of an English Wikipedia article.

## Evaluation Criteria

1. **Translation accuracy**
   - Is the meaning preserved?
   - Are facts unchanged?
   - Are there omissions or additions?

2. **Albanian language quality**
   - Grammar
   - Naturalness
   - Fluency
   - Encyclopedic style

3. **Terminology quality**
   - Correct Albanian terminology.
   - Consistency with Wikipedia style.
   - Appropriate handling of names and technical terms.

4. **MediaWiki quality**
   - Preservation of templates.
   - Infobox correctness.
   - References.
   - Links.
   - Tables.
   - Formatting.

5. **Publication readiness**
   - How much human editing would be required before publishing?

## Output Format

You must produce a structured evaluation with two parts:

### Part 1: Markdown summary
Provide a clear markdown evaluation with:
- Scores for each translation (by its label) for each category (1-10).
- Overall score per translation (1-10).
- Ranking from best to worst (e.g., "A > C > B > D").
- Detailed explanations for each translation's strengths and weaknesses.
- Recommendation which translation model (based solely on the labels) would be best for English → Albanian Wikipedia translation in general, and why.

### Part 2: JSON data
After the markdown, include a JSON code block with the following structure:

```json
{
  "scores": {
    "A": {
      "translation_accuracy": <1-10>,
      "albanian_quality": <1-10>,
      "terminology_quality": <1-10>,
      "mediawiki_quality": <1-10>,
      "publication_readiness": <1-10>,
      "overall": <1-10>
    },
    "B": { ... },
    ...
  },
  "ranking": ["A", "C", "B", "D"],
  "detailed_explanations": {
    "A": "Explanation markdown...",
    ...
  },
  "recommendation": "A is the best because..."
}
```

All scores are numbers (integers 1-10, overall can be float).

You are not told which model produced which translation; the labels are random.
"""


@dataclass
class EvaluationScores:
    """Per-translation scores."""
    label: str
    translation_accuracy: int
    albanian_quality: int
    terminology_quality: int
    mediawiki_quality: int
    publication_readiness: int
    overall_score: float  # the judge is told overall may be fractional


@dataclass
class EvaluationResult:
    """Full evaluation result."""
    judge_model: str
    label_to_model: dict[str, str]
    scores: list[EvaluationScores]
    ranking: list[str]  # labels in order best to worst
    detailed_explanations: dict[str, str]  # label -> markdown explanation
    recommendation: str  # text recommendation
    judge_raw_response: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float


def build_evaluation_prompt(
    source_english: str,
    translations: dict[str, str],  # label -> translation wikitext
) -> list[dict[str, str]]:
    """Build messages for judge model."""
    # Shuffle labels to randomize order but keep mapping
    labels = list(translations.keys())
    random.shuffle(labels)
    shuffled_translations = {label: translations[label] for label in labels}
    
    translation_texts = []
    for label, wikitext in shuffled_translations.items():
        translation_texts.append(f"## Translation {label}\n\n```wikitext\n{wikitext}\n```")
    
    translations_block = "\n\n".join(translation_texts)
    label_list = ", ".join(sorted(translations))
    
    user_content = f"""# Albanian Wikipedia Translation Evaluation

## Original English Wikipedia Article

```wikitext
{source_english}
```

## Albanian Translations (labels {label_list})

The {len(translations)} translations are presented in random order.

{translations_block}

## Instructions

{EVALUATION_CRITERIA}

Please provide your evaluation now."""
    
    return [
        {"role": "system", "content": "You are an expert Albanian Wikipedia editor and translator evaluator."},
        {"role": "user", "content": user_content}
    ]


def extract_json_block(text: str) -> dict[str, Any] | None:
    """Extract JSON object from a code block marked with ```json ... ```."""
    import re
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    # Fallback: find any {...} that looks like JSON
    # naive: find first { and last } but could be nested
    # For simplicity, we assume the JSON is the outermost object.
    try:
        # Find first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end > start:
            json_str = text[start:end+1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    return None


def parse_evaluation_response(response_text: str) -> dict[str, Any]:
    """Parse judge's response into structured data."""
    data = extract_json_block(response_text)
    if not data:
        return {}
    # Normalize data
    result = {}
    scores = data.get('scores', {})
    # Convert scores to our format
    parsed_scores = {}
    for label, score_dict in scores.items():
        if isinstance(score_dict, dict):
            parsed_scores[label] = {
                'translation_accuracy': score_dict.get('translation_accuracy', 0),
                'albanian_quality': score_dict.get('albanian_quality', 0),
                'terminology_quality': score_dict.get('terminology_quality', 0),
                'mediawiki_quality': score_dict.get('mediawiki_quality', 0),
                'publication_readiness': score_dict.get('publication_readiness', 0),
                'overall': score_dict.get('overall', 0),
            }
    result['scores'] = parsed_scores
    result['ranking'] = data.get('ranking', [])
    result['detailed_explanations'] = data.get('detailed_explanations', {})
    result['recommendation'] = data.get('recommendation', '')
    return result


def _score(value: Any, cast: type = int) -> int | float:
    """A judge-reported score as a number; 0 when it is missing or not
    numeric ("7.5", None, "n/a"), so one malformed field can't discard the
    whole (already paid-for) evaluation."""
    try:
        return cast(float(value))
    except (TypeError, ValueError):
        return 0


def create_evaluation_result(
    judge_model: str,
    label_to_model: dict[str, str],
    parsed_data: dict[str, Any],
    raw_response: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    latency_s: float,
) -> EvaluationResult:
    """Create EvaluationResult from parsed data."""
    scores = []
    for label in label_to_model:
        score_data = parsed_data.get('scores', {}).get(label, {})
        scores.append(EvaluationScores(
            label=label,
            translation_accuracy=_score(score_data.get('translation_accuracy')),
            albanian_quality=_score(score_data.get('albanian_quality')),
            terminology_quality=_score(score_data.get('terminology_quality')),
            mediawiki_quality=_score(score_data.get('mediawiki_quality')),
            publication_readiness=_score(score_data.get('publication_readiness')),
            overall_score=_score(score_data.get('overall'), float),
        ))
    ranking = parsed_data.get('ranking', [])
    detailed_explanations = parsed_data.get('detailed_explanations', {})
    recommendation = parsed_data.get('recommendation', '')
    return EvaluationResult(
        judge_model=judge_model,
        label_to_model=label_to_model,
        scores=scores,
        ranking=ranking,
        detailed_explanations=detailed_explanations,
        recommendation=recommendation,
        judge_raw_response=raw_response,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        latency_s=latency_s,
    )


async def evaluate_translations(
    judge_model: str,
    source_english: str,
    translations: dict[str, str],  # model_id -> translation wikitext
    openrouter_api_key: str,
    openrouter_base_url: str = "https://openrouter.ai/api/v1",
    user_agent: str = "wiki-translation-harness-evaluation",
    timeout: float = 300.0,
    provider: str = "openrouter",
) -> tuple[EvaluationResult | None, str | None]:
    """
    Evaluate translations using a judge model.
    
    Returns (evaluation_result, error_message). If error occurs, returns (None, error).
    """
    # Map models to labels A-D
    models = list(translations.keys())
    if len(models) > 4:
        return None, "Cannot evaluate more than 4 translations"
    labels = ["A", "B", "C", "D"][:len(models)]
    label_to_model = dict(zip(labels, models))
    translations_by_label = {label: translations[model] for label, model in label_to_model.items()}
    
    messages = build_evaluation_prompt(source_english, translations_by_label)
    
    client = OpenRouterClient(
        api_key=openrouter_api_key,
        base_url=openrouter_base_url,
        user_agent=user_agent,
        timeout=timeout,
        max_retries=3,
        provider=provider,
    )
    
    try:
        # Get pricing for cost estimation
        pricing = await client.get_pricing_for(judge_model)
        
        start = time.monotonic()
        text, prompt_tokens, completion_tokens = await client.chat_completion(
            judge_model, messages, temperature=0.0
        )
        latency = time.monotonic() - start
        cost = 0.0
        if pricing:
            cost = (
                prompt_tokens * pricing.prompt_price_per_token
                + completion_tokens * pricing.completion_price_per_token
            )
        
        parsed = parse_evaluation_response(text)
        if not parsed:
            # fallback: empty result but keep raw response
            parsed = {}
        
        result = create_evaluation_result(
            judge_model=judge_model,
            label_to_model=label_to_model,
            parsed_data=parsed,
            raw_response=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            latency_s=latency,
        )
        return result, None
    except OpenRouterError as e:
        return None, str(e)
    finally:
        await client.aclose()


def save_evaluation_results(
    result: EvaluationResult,
    output_dir: Path,
    source_title: str,
    translations: dict[str, str],  # model_id -> wikitext
    benchmark_results: dict[str, dict],  # model_id -> runtime, tokens, cost
) -> None:
    """Save evaluation results to disk with the requested directory structure."""
    # Create directories
    translations_dir = output_dir / "translations"
    evaluation_dir = output_dir / "evaluation"
    translations_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    
    # Save translations as model_a.wiki etc.
    for label, model in result.label_to_model.items():
        wikitext = translations.get(model)
        if wikitext:
            (translations_dir / f"model_{label.lower()}.wiki").write_text(wikitext, encoding="utf-8")
    
    # Save mapping JSON
    mapping = {
        "label_to_model": result.label_to_model,
        "benchmark_results": benchmark_results,
    }
    (output_dir / "mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    
    # Save evaluation markdown
    markdown = f"# Evaluation by {result.judge_model}\n\n"
    markdown += f"**Prompt tokens:** {result.prompt_tokens}  \n"
    markdown += f"**Completion tokens:** {result.completion_tokens}  \n"
    markdown += f"**Cost:** ${result.cost_usd:.4f}  \n"
    markdown += f"**Latency:** {result.latency_s:.2f}s  \n\n"
    
    # Scores table
    markdown += "## Scores\n\n"
    markdown += "| Label | Translation Accuracy | Albanian Quality | Terminology Quality | MediaWiki Quality | Publication Readiness | Overall |\n"
    markdown += "|-------|---------------------|------------------|---------------------|-------------------|------------------------|---------|\n"
    for score in result.scores:
        markdown += f"| {score.label} | {score.translation_accuracy} | {score.albanian_quality} | {score.terminology_quality} | {score.mediawiki_quality} | {score.publication_readiness} | {score.overall_score} |\n"
    
    # Ranking
    markdown += f"\n## Ranking\n\n{' > '.join(result.ranking)}\n"
    
    # Detailed explanations
    markdown += "\n## Detailed Explanations\n\n"
    for label, explanation in result.detailed_explanations.items():
        markdown += f"### Translation {label}\n\n{explanation}\n\n"
    
    # Recommendation
    markdown += f"\n## Recommendation\n\n{result.recommendation}\n"
    
    # Raw response
    markdown += f"\n## Raw Judge Response\n\n```markdown\n{result.judge_raw_response}\n```"
    
    (evaluation_dir / "sonnet_judge.md").write_text(markdown, encoding="utf-8")
    
    # Save raw JSON
    (evaluation_dir / "evaluation.json").write_text(json.dumps({
        "judge_model": result.judge_model,
        "label_to_model": result.label_to_model,
        "scores": [{
            "label": s.label,
            "translation_accuracy": s.translation_accuracy,
            "albanian_quality": s.albanian_quality,
            "terminology_quality": s.terminology_quality,
            "mediawiki_quality": s.mediawiki_quality,
            "publication_readiness": s.publication_readiness,
            "overall_score": s.overall_score,
        } for s in result.scores],
        "ranking": result.ranking,
        "detailed_explanations": result.detailed_explanations,
        "recommendation": result.recommendation,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "cost_usd": result.cost_usd,
        "latency_s": result.latency_s,
    }, indent=2), encoding="utf-8")
    
    # Create comparison.md combining automatic metrics and evaluation
    comparison_path = output_dir / "comparison.md"
    with open(comparison_path, "w", encoding="utf-8") as f:
        f.write(f"# Comparison: {source_title}\n\n")
        f.write("## Benchmark Results\n\n")
        f.write("| Model | Runtime (s) | Input Tokens | Output Tokens | Cost | Failed Sections |\n")
        f.write("|-------|-------------|--------------|---------------|------|-----------------|\n")
        for model, res in benchmark_results.items():
            f.write(f"| {model} | {res['runtime_s']:.1f} | {res['tokens_in']} | {res['tokens_out']} | ${res['estimated_cost_usd']:.4f} | {res['failed_sections']} |\n")
        f.write("\n## Evaluation Summary\n\n")
        f.write(f"**Judge model:** {result.judge_model}  \n")
        f.write(f"**Ranking:** {' > '.join(result.ranking)}\n")
        f.write("\n### Scores\n\n")
        f.write("| Model | Label | Translation Accuracy | Albanian Quality | Terminology Quality | MediaWiki Quality | Publication Readiness | Overall |\n")
        f.write("|-------|-------|---------------------|------------------|---------------------|-------------------|------------------------|---------|\n")
        for score in result.scores:
            model = result.label_to_model.get(score.label, "unknown")
            f.write(f"| {model} | {score.label} | {score.translation_accuracy} | {score.albanian_quality} | {score.terminology_quality} | {score.mediawiki_quality} | {score.publication_readiness} | {score.overall_score} |\n")
        f.write("\n### Recommendation\n\n")
        f.write(f"{result.recommendation}\n")
        f.write("\n---\n\n*Evaluation performed blind with randomized labels.*\n")
