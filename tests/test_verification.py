from pathlib import Path

import httpx
import pytest
import respx

from wiki_translation_harness.cache import VerificationCache
from wiki_translation_harness.mediawiki import MediaWikiClient
from wiki_translation_harness.verification import (
    VerifiedFacts,
    WikidataVerifier,
    build_verified_facts_block,
    existing_article_matches_by_length,
    extract_link_targets,
    extract_template_names,
    verify_wikitext,
)

WIKIDATA_API = "https://www.wikidata.org/w/api.php"


def test_extract_link_targets_excludes_external_and_file_and_category():
    text = "[[https://example.com]] [[File:Foo.png]] [[Category:Physics]] [[Real Link]]"
    assert extract_link_targets(text) == ["Real Link"]


def test_extract_template_names_excludes_parser_functions():
    text = "{{#if:x|yes|no}} {{PAGENAME}}"
    names = extract_template_names(text)
    assert "#if" not in names
    assert "PAGENAME" in names


def test_existing_article_matches_by_length_allows_language_size_difference():
    facts = VerifiedFacts(existing_target_title="Fëmija", existing_target_bytes=1300)
    assert existing_article_matches_by_length("x" * 1000, facts)
    assert not existing_article_matches_by_length("x" * 1000, VerifiedFacts())
    assert not existing_article_matches_by_length(
        "x" * 1000,
        VerifiedFacts(existing_target_title="Fëmija", existing_target_bytes=1601),
    )


@pytest.mark.asyncio
async def test_wikidata_verifier_normalizes_first_letter_case():
    # queried as "reflist" (lowercase, as commonly written in wikitext), but
    # Wikidata only knows the canonical MediaWiki-capitalized "Reflist" —
    # result must still come back keyed by the original lowercase input.
    with respx.mock() as mock:
        mock.get(url__startswith=WIKIDATA_API).mock(
            return_value=httpx.Response(
                200,
                json={
                    "entities": {
                        "Q5462890": {
                            "sitelinks": {
                                "enwiki": {"site": "enwiki", "title": "Template:Reflist"},
                                "sqwiki": {"site": "sqwiki", "title": "Stampa:Reflist"},
                            }
                        }
                    },
                    "success": 1,
                },
            )
        )
        async with httpx.AsyncClient() as client:
            verifier = WikidataVerifier(client, "en", "sq")
            result = await verifier.check_template_sitelinks(["reflist"])
    assert result == {"reflist": "Stampa:Reflist"}


@pytest.mark.asyncio
async def test_wikidata_verifier_check_template_sitelinks_keyed_by_bare_name():
    with respx.mock() as mock:
        mock.get(url__startswith=WIKIDATA_API).mock(
            return_value=httpx.Response(
                200,
                json={
                    "entities": {
                        "Q1": {
                            "sitelinks": {
                                "enwiki": {"site": "enwiki", "title": "Template:Infobox royalty"},
                                "sqwiki": {"site": "sqwiki", "title": "Stampa:Infobox royalty"},
                            }
                        }
                    },
                    "success": 1,
                },
            )
        )
        async with httpx.AsyncClient() as client:
            verifier = WikidataVerifier(client, "en", "sq")
            result = await verifier.check_template_sitelinks(["Infobox royalty"])
    assert result == {"Infobox royalty": "Stampa:Infobox royalty"}


def test_verification_cache_get_many_absent_vs_missing(tmp_path: Path):
    cache = VerificationCache(tmp_path / "v.sqlite3")
    # nothing cached yet -> empty dict, not a False/None entry
    assert cache.get_many("en", "sq", "link", ["Paris"]) == {}
    cache.set_many("en", "sq", "link", {"Paris": "Parisi", "NopeArticle": None})
    result = cache.get_many("en", "sq", "link", ["Paris", "NopeArticle", "Uncached"])
    assert result == {"Paris": "Parisi", "NopeArticle": None}
    assert "Uncached" not in result
    cache.close()


def test_verification_cache_persists_across_reopen(tmp_path: Path):
    db_path = tmp_path / "v.sqlite3"
    cache1 = VerificationCache(db_path)
    cache1.set_many("en", "sq", "template", {"Infobox royalty": "Stampa:Infobox royalty"})
    cache1.close()

    cache2 = VerificationCache(db_path)
    assert cache2.get_many("en", "sq", "template", ["Infobox royalty"]) == {
        "Infobox royalty": "Stampa:Infobox royalty"
    }
    cache2.close()


def test_build_verified_facts_block_template_with_params():
    facts = VerifiedFacts(
        templates={"Infobox royalty": "Stampa:Infobox royalty"},
        template_params={"Infobox royalty": ["name", "title", "image"]},
    )
    block = build_verified_facts_block("{{Infobox royalty|name=x}}", facts)
    assert "confirmed present as {{Infobox royalty}}" in block
    assert "name, title, image" in block
    assert "do not invent or translate them" in block


def test_record_established_renderings_extracts_ill_calls():
    facts = VerifiedFacts(links={"Sustainable architecture": None})
    facts.record_established_renderings(
        "...ideja e {{ill|arkitekturës së qëndrueshme|en|Sustainable architecture}} u prezantua..."
    )
    assert facts.established_renderings == {"Sustainable architecture": "arkitekturës së qëndrueshme"}


def test_verified_facts_block_reuses_established_rendering_for_unconfirmed_link():
    facts = VerifiedFacts(
        links={"Sustainable architecture": None},
        established_renderings={"Sustainable architecture": "arkitektura e qëndrueshme"},
    )
    block = build_verified_facts_block("[[Sustainable architecture]]", facts)
    assert "already rendered elsewhere in this article" in block
    assert "{{ill|arkitektura e qëndrueshme|en|Sustainable architecture}}" in block
    assert "NOT FOUND" in block
    # Once an established rendering exists, the language-count fallback note
    # is redundant and should not also be printed for the same link.
    assert "other Wikipedia language" not in block


@pytest.mark.asyncio
async def test_verify_wikitext_end_to_end_with_mocks(tmp_path: Path):
    with respx.mock() as mock:
        mock.get(url__startswith=WIKIDATA_API).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "entities": {
                            "Q90": {
                                "sitelinks": {
                                    "enwiki": {"site": "enwiki", "title": "Paris"},
                                    "sqwiki": {"site": "sqwiki", "title": "Parisi"},
                                }
                            }
                        },
                        "success": 1,
                    },
                ),
                httpx.Response(
                    200,
                    json={
                        "entities": {
                            "Q1": {
                                "sitelinks": {
                                    "enwiki": {"site": "enwiki", "title": "Template:Infobox royalty"},
                                    "sqwiki": {"site": "sqwiki", "title": "Stampa:Infobox royalty"},
                                }
                            }
                        },
                        "success": 1,
                    },
                ),
                httpx.Response(
                    200,
                    json={
                        "entities": {"-1": {"site": "enwiki", "title": "Test Article", "missing": ""}},
                        "success": 1,
                    },
                ),
            ]
        )
        with respx.mock(base_url="https://sq.wikipedia.org") as sq_mock:
            sq_mock.get(url__startswith="https://sq.wikipedia.org/w/api.php").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "query": {
                            "pages": [
                                {
                                    "title": "Stampa:Infobox royalty",
                                    "revisions": [
                                        {
                                            "revid": 1,
                                            "timestamp": "2024-01-01T00:00:00Z",
                                            "slots": {"main": {"content": "{{{name|}}} {{{title|}}}"}},
                                        }
                                    ],
                                }
                            ]
                        }
                    },
                )
            )
            cache = VerificationCache(tmp_path / "v.sqlite3")
            target_client = MediaWikiClient("https://sq.wikipedia.org/w/api.php", "test-agent/1.0", "sq")
            async with httpx.AsyncClient() as wikidata_client:
                facts = await verify_wikitext(
                    "Test Article",
                    "[[Paris]] {{Infobox royalty|name=x}}",
                    "en",
                    "sq",
                    wikidata_client,
                    target_client,
                    cache,
                )
            await target_client.aclose()
            cache.close()

    assert facts.links == {"Paris": "Parisi"}
    assert facts.templates == {"Infobox royalty": "Stampa:Infobox royalty"}
    assert facts.template_params == {"Infobox royalty": ["name", "title"]}


@pytest.mark.asyncio
async def test_verify_wikitext_detects_existing_target_article(tmp_path: Path):
    with respx.mock() as mock:
        # wikitext has no links/templates, so _resolve_cached_then_live short
        # -circuits both of those (no titles -> no call); the article's own
        # existence check is the *only* Wikidata call that actually fires.
        mock.get(url__startswith=WIKIDATA_API).mock(
            return_value=httpx.Response(
                200,
                json={
                    "entities": {
                        "Q808052": {
                            "sitelinks": {
                                "enwiki": {"site": "enwiki", "title": "Bardylis"},
                                "sqwiki": {"site": "sqwiki", "title": "Bardhyli"},
                            }
                        }
                    },
                    "success": 1,
                },
            )
        )
        with respx.mock(base_url="https://sq.wikipedia.org") as sq_mock:
            sq_mock.get(url__startswith="https://sq.wikipedia.org/w/api.php").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "query": {
                            "pages": [
                                {
                                    "title": "Bardhyli",
                                    "revisions": [
                                        {
                                            "revid": 1,
                                            "timestamp": "2024-01-01T00:00:00Z",
                                            "slots": {
                                                "main": {"content": "[[Ilirët]] dhe [[Mbretëria e Ilirisë]]"}
                                            },
                                        }
                                    ],
                                }
                            ]
                        }
                    },
                )
            )
            cache = VerificationCache(tmp_path / "v.sqlite3")
            target_client = MediaWikiClient("https://sq.wikipedia.org/w/api.php", "test-agent/1.0", "sq")
            async with httpx.AsyncClient() as wikidata_client:
                facts = await verify_wikitext(
                    "Bardylis", "text with no links or templates", "en", "sq", wikidata_client, target_client, cache
                )
            await target_client.aclose()
            cache.close()

    assert facts.existing_target_title == "Bardhyli"
    assert facts.sibling_links == ["Ilirët", "Mbretëria e Ilirisë"]


def test_verified_facts_block_includes_sibling_note():
    facts = VerifiedFacts(existing_target_title="Bardhyli", sibling_links=["Ilirët", "Mbretëria e Ilirisë"])
    block = build_verified_facts_block("some chunk text with no links", facts)
    assert "already exists on the target wiki as 'Bardhyli'" in block
    assert "Ilirët" in block
    assert "Mbretëria e Ilirisë" in block


def test_verified_facts_block_includes_citation_param_note():
    facts = VerifiedFacts()
    block = build_verified_facts_block("{{cite web|title=Some Source}}", facts)
    assert "fixed, English parameter NAMES" in block
    assert "|titulli=" in block


@pytest.mark.asyncio
async def test_count_other_language_sitelinks_notable_concept():
    with respx.mock() as mock:
        mock.get(url__startswith=WIKIDATA_API).mock(
            return_value=httpx.Response(
                200,
                json={
                    "entities": {
                        "Q93189": {
                            "sitelinks": {
                                "enwiki": {"site": "enwiki", "title": "Egg"},
                                "dewiki": {"site": "dewiki", "title": "Ei"},
                                "frwiki": {"site": "frwiki", "title": "Œuf"},
                                "itwiki": {"site": "itwiki", "title": "Uovo"},
                            }
                        }
                    },
                    "success": 1,
                },
            )
        )
        async with httpx.AsyncClient() as client:
            verifier = WikidataVerifier(client, "en", "sq")
            result = await verifier.count_other_language_sitelinks(["Egg"])
    assert result == {"Egg": 3}  # excludes source (enwiki) itself


def test_verified_facts_block_notable_not_found_link():
    facts = VerifiedFacts(links={"Egg": None}, not_found_link_language_counts={"Egg": 45})
    block = build_verified_facts_block("[[Egg]]", facts)
    assert "45 other Wikipedia language" in block
    assert "likely a real, distinct topic" in block




@pytest.mark.asyncio
async def test_count_other_language_sitelinks_skips_non_wikipedia_sites():
    with respx.mock() as mock:
        mock.get(url__startswith=WIKIDATA_API).mock(
            return_value=httpx.Response(
                200,
                json={
                    "entities": {
                        "Q93189": {
                            "sitelinks": {
                                "enwiki": {"site": "enwiki", "title": "Egg"},
                                "dewiki": {"site": "dewiki", "title": "Ei"},
                                "commonswiki": {"site": "commonswiki", "title": "Category:Eggs"},
                                "specieswiki": {"site": "specieswiki", "title": "Egg"},
                            }
                        }
                    },
                    "success": 1,
                },
            )
        )
        async with httpx.AsyncClient() as client:
            verifier = WikidataVerifier(client, "en", "sq")
            result = await verifier.count_other_language_sitelinks(["Egg"])
    assert result == {"Egg": 1}
