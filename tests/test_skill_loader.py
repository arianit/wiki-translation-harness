import subprocess
from pathlib import Path


from wiki_translation_harness.skill_loader import (
    build_repair_messages,
    build_semantic_repair_messages,
    build_translation_messages,
    load_skill,
)


def _make_skill_dir(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "enwiki-sqwiki-translation"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: test\n---\n\n# Body\n\nTranslation rules here.\n"
    )
    refs = skill_dir / "references"
    refs.mkdir()
    (refs / "notes.md").write_text("Some verified reference notes.")
    return skill_dir


def _make_second_skill_dir(tmp_path: Path) -> Path:
    """A second, independent skill directory — mimics wikiterms sitting
    alongside enwiki-sqwiki-translation as a sibling to be concatenated."""
    skill_dir = tmp_path / "wikiterms"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: wikiterms\ndescription: test\n---\n\n# Terms\n\nVerify link targets first.\n"
    )
    refs = skill_dir / "references"
    refs.mkdir()
    (refs / "notes.md").write_text("Verified sqwiki template cache.")
    return skill_dir


def test_frontmatter_stripped(tmp_path: Path):
    skill_dir = _make_skill_dir(tmp_path)
    skill = load_skill(skill_dir, include_references=False)
    assert "name: test-skill" not in skill.skill_md
    assert "Translation rules here." in skill.skill_md


def test_references_included_when_requested(tmp_path: Path):
    skill_dir = _make_skill_dir(tmp_path)
    skill = load_skill(skill_dir, include_references=True)
    assert "notes.md" in skill.reference_texts
    assert "verified reference notes" in skill.combined


def test_translation_messages_contain_skill_and_input(tmp_path: Path):
    skill_dir = _make_skill_dir(tmp_path)
    skill = load_skill(skill_dir, include_references=False)
    messages = build_translation_messages(skill, "en", "sq", "Paris", "Lead", "Paris is a city.")
    assert messages[0]["role"] == "system"
    assert "Translation rules here." in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "Paris is a city." in messages[1]["content"]
    assert "Source language: en" in messages[1]["content"]
    assert "Target language: sq" in messages[1]["content"]


def test_harness_never_embeds_translation_prompt(tmp_path: Path):
    """The system prompt's translation-relevant content must come only from
    the skill file, not from any harness-authored text."""
    skill_dir = _make_skill_dir(tmp_path)
    skill = load_skill(skill_dir, include_references=False)
    messages = build_translation_messages(skill, "en", "sq", "Paris", "Lead", "text")
    system_content = messages[0]["content"]
    # everything besides the skill body should be pure invocation framing —
    # spot check that no grammar/vocabulary guidance appears outside the skill
    framing_only = system_content.split("Translation rules here.")[0]
    assert "case" not in framing_only.lower() or "no tools" in framing_only.lower()


def test_content_hash_changes_when_skill_body_edited(tmp_path: Path):
    skill_dir = _make_skill_dir(tmp_path)
    before = load_skill(skill_dir, include_references=False)

    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: test\n---\n\n# Body\n\nDifferent rules now.\n"
    )
    after = load_skill(skill_dir, include_references=False)

    assert before.content_hash != after.content_hash


def test_translation_messages_never_include_qa_skill_content(tmp_path: Path):
    """The QA/pre-delivery checklist must never reach a normal translation
    call — build_translation_messages has no parameter for it at all."""
    skill_dir = _make_skill_dir(tmp_path)
    skill = load_skill(skill_dir, include_references=False)
    messages = build_translation_messages(skill, "en", "sq", "Paris", "Lead", "text")
    assert "qa checklist" not in messages[0]["content"].lower()


def test_repair_messages_include_qa_skill_when_provided(tmp_path: Path):
    """QA content is appended to a repair call only — the one case where
    validate_wikitext has already found a real problem — never to an
    ordinary translation call."""
    skill_dir = _make_skill_dir(tmp_path)
    skill = load_skill(skill_dir, include_references=False)
    qa_dir = tmp_path / "wikiqa"
    qa_dir.mkdir()
    (qa_dir / "SKILL.md").write_text(
        "---\nname: wikiqa\ndescription: test\n---\n\n# QA\n\nCheck ref names before delivery.\n"
    )
    qa_skill = load_skill(qa_dir, include_references=False)

    messages = build_repair_messages(
        skill, "en", "sq", "Paris", "Lead", "[[broken", ["link: unbalanced"], qa_skill=qa_skill
    )
    assert "Check ref names before delivery." in messages[0]["content"]
    assert "Translation rules here." in messages[0]["content"]


def test_multi_path_concatenates_bodies_in_order(tmp_path: Path):
    first = _make_skill_dir(tmp_path)
    second = _make_second_skill_dir(tmp_path)
    skill = load_skill([first, second], include_references=False)
    assert "Translation rules here." in skill.skill_md
    assert "Verify link targets first." in skill.skill_md
    assert skill.skill_md.index("Translation rules here.") < skill.skill_md.index(
        "Verify link targets first."
    )


def test_multi_path_prefixes_reference_keys_to_avoid_collision(tmp_path: Path):
    first = _make_skill_dir(tmp_path)
    second = _make_second_skill_dir(tmp_path)
    skill = load_skill([first, second], include_references=True)
    assert "enwiki-sqwiki-translation/notes.md" in skill.reference_texts
    assert "wikiterms/notes.md" in skill.reference_texts
    assert "Some verified reference notes." in skill.reference_texts["enwiki-sqwiki-translation/notes.md"]
    assert "Verified sqwiki template cache." in skill.reference_texts["wikiterms/notes.md"]


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _make_git_skill_repo(tmp_path: Path) -> Path:
    """A repo laid out like the real skill: a subdirectory (mimicking the
    symlink target under ~/.claude/skills/) holding SKILL.md + references,
    with one commit, then an uncommitted local edit on top."""
    repo_root = tmp_path / "skill-repo"
    repo_root.mkdir()
    _git(["init", "-q"], cwd=repo_root)
    _git(["config", "user.email", "test@example.com"], cwd=repo_root)
    _git(["config", "user.name", "Test"], cwd=repo_root)

    skill_dir = repo_root / "enwiki-sqwiki-translation"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: test\n---\n\n# Body\n\nCommitted rules.\n"
    )
    refs = skill_dir / "references"
    refs.mkdir()
    (refs / "notes.md").write_text("Committed reference notes.")

    _git(["add", "-A"], cwd=repo_root)
    _git(["commit", "-q", "-m", "initial"], cwd=repo_root)

    # dirty the working copy after the commit, uncommitted
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: test\n---\n\n# Body\n\nUNCOMMITTED local edit.\n"
    )
    (refs / "notes.md").write_text("UNCOMMITTED reference edit.")

    return skill_dir


def test_git_ref_reads_committed_content_not_working_copy(tmp_path: Path):
    skill_dir = _make_git_skill_repo(tmp_path)
    skill = load_skill(skill_dir, include_references=False, git_ref="HEAD")
    assert "Committed rules." in skill.skill_md
    assert "UNCOMMITTED" not in skill.skill_md


def _make_second_git_skill_repo(tmp_path: Path) -> Path:
    """A second, independent skill git repo — mimics wikiterms living in its
    own repo (or its own subdirectory of a shared repo) alongside the main
    skill's repo."""
    repo_root = tmp_path / "skill-repo-2"
    repo_root.mkdir()
    _git(["init", "-q"], cwd=repo_root)
    _git(["config", "user.email", "test@example.com"], cwd=repo_root)
    _git(["config", "user.name", "Test"], cwd=repo_root)

    skill_dir = repo_root / "wikiterms"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: wikiterms\ndescription: test\n---\n\n# Terms\n\nCommitted terms rules.\n"
    )
    refs = skill_dir / "references"
    refs.mkdir()
    (refs / "notes.md").write_text("Committed terms reference notes.")

    _git(["add", "-A"], cwd=repo_root)
    _git(["commit", "-q", "-m", "initial"], cwd=repo_root)

    (skill_dir / "SKILL.md").write_text(
        "---\nname: wikiterms\ndescription: test\n---\n\n# Terms\n\nUNCOMMITTED terms edit.\n"
    )
    return skill_dir


def test_git_ref_multi_path_concatenates_committed_content(tmp_path: Path):
    first = _make_git_skill_repo(tmp_path)
    second = _make_second_git_skill_repo(tmp_path)
    skill = load_skill([first, second], include_references=True, git_ref="HEAD")
    assert "Committed rules." in skill.skill_md
    assert "Committed terms rules." in skill.skill_md
    assert "UNCOMMITTED" not in skill.skill_md
    assert "enwiki-sqwiki-translation/notes.md" in skill.reference_texts
    assert "wikiterms/notes.md" in skill.reference_texts


def test_semantic_repair_messages_show_source_and_allow_rewording(tmp_path: Path):
    skill = load_skill(_make_skill_dir(tmp_path), include_references=False)
    messages = build_semantic_repair_messages(
        skill, "en", "sq", "Paris", "Lead",
        "Paris is a big city.", "Parisi është një qytet i vogël.",
        ["error: says small, source says big"],
        structural_errors=["link: unbalanced"],
    )
    system, user = messages[0]["content"], messages[1]["content"]
    assert "keep all wording" not in system
    assert "Translation rules here." in system
    assert "Paris is a big city." in user
    assert "Parisi është një qytet i vogël." in user
    assert "- error: says small, source says big" in user
    assert "- link: unbalanced" in user
    assert user.index("BEGIN SOURCE WIKITEXT") < user.index("BEGIN TRANSLATED WIKITEXT TO FIX")


def test_semantic_repair_messages_omit_structural_block_when_none(tmp_path: Path):
    skill = load_skill(_make_skill_dir(tmp_path), include_references=False)
    messages = build_semantic_repair_messages(skill, "en", "sq", "Paris", "Lead", "A", "B", ["f"])
    assert "Structural errors" not in messages[1]["content"]
