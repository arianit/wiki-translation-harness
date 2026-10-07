from pathlib import Path
from unittest.mock import patch

import pytest

from wiki_translation_harness.models import Config
from wiki_translation_harness.output import discard_partial_article
from wiki_translation_harness.resume import checkpoint_input, unfinished_inputs
from wiki_translation_harness.sources import ArticleInput


def config(tmp_path):
    return Config(partial_output_dir=tmp_path / 'partial', output_dir=tmp_path / 'output')


def test_checkpoint_preserves_local_source_and_languages(tmp_path):
    cfg = config(tmp_path)
    item = ArticleInput('Title: exact', tmp_path / 'source.wiki', 'de')
    checkpoint_input(cfg, item)
    assert unfinished_inputs(cfg, []) == [item]
    assert unfinished_inputs(cfg.model_copy(update={'target_lang': 'fr'}), []) == []
    discard_partial_article(cfg.partial_output_dir, item.title)
    assert unfinished_inputs(cfg, []) == []


def test_legacy_snapshots_prioritize_requested_identity_and_skip_completed(tmp_path):
    cfg = config(tmp_path)
    cfg.partial_output_dir.mkdir()
    for name in ['Old_article', 'Done', 'Local']:
        (cfg.partial_output_dir / f'{name}.wiki').write_text('partial')
    cfg.output_dir.mkdir()
    (cfg.output_dir / 'Done.wiki').write_text('complete')
    local = ArticleInput('Local', tmp_path / 'input.wiki', 'fr')
    assert unfinished_inputs(cfg, [local]) == [local, ArticleInput('Old article', source_lang=cfg.source_lang)]


@pytest.mark.asyncio
async def test_pipeline_awaits_backlog_before_loading_new_work(tmp_path):
    import wiki_translation_harness.pipeline as pipeline
    cfg = config(tmp_path)
    old = ArticleInput('Old', source_lang='en')
    checkpoint_input(cfg, old)
    real_pipeline = pipeline.run_pipeline
    events = []

    async def retry(config, inputs, **kwargs):
        assert inputs == [old]
        assert config.sequential
        assert kwargs['resume_unfinished'] is False
        events.append('retried')

    def load(*args):
        assert events == ['retried']
        events.append('new work')
        raise RuntimeError('stop before engine setup')

    with patch.object(pipeline, 'run_pipeline', side_effect=retry), patch.object(pipeline, 'load_skill', side_effect=load):
        with pytest.raises(RuntimeError, match='stop before engine setup'):
            await real_pipeline(cfg, [ArticleInput('New'), old])
    assert events == ['retried', 'new work']


@pytest.mark.asyncio
async def test_queue_retries_backlog_before_claim_and_counts_it_toward_limit(tmp_path):
    import wiki_translation_harness.pipeline as pipeline
    import wiki_translation_harness.queue_runner as runner
    cfg = config(tmp_path)
    checkpoint_input(cfg, ArticleInput('Old', source_lang='en'))
    from unittest.mock import Mock
    queue = Mock()
    queue.claim_next_pending.return_value = None
    events = []

    async def retry(config, inputs, **kwargs):
        assert inputs[0].title == 'Old'
        queue.claim_next_pending.assert_not_called()
        events.append('retry')

    with patch.object(runner, 'load_queue_lib', return_value=queue), patch.object(pipeline, 'run_pipeline', side_effect=retry):
        await runner.run_queue_mode(cfg, max_articles=1)
    assert events == ['retry']
    queue.claim_next_pending.assert_not_called()
