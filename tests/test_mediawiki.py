import time

import httpx
import pytest
import respx

from wiki_translation_harness.mediawiki import (
    MediaWikiClient,
    MediaWikiClientPool,
    MediaWikiError,
)

API_URL = "https://en.wikipedia.org/w/api.php"


def test_client_pool_uses_override_only_for_default_lang():
    pool = MediaWikiClientPool(
        "test-agent/1.0", default_lang="en", default_api_url="https://custom.example.org/w/api.php"
    )
    assert pool.get("en").api_url == "https://custom.example.org/w/api.php"
    assert pool.get("sq").api_url == "https://sq.wikipedia.org/w/api.php"


@pytest.mark.asyncio
async def test_fetch_article_success():
    with respx.mock(base_url=API_URL) as mock:
        mock.get(params={"action": "query"}).mock(
            return_value=httpx.Response(
                200,
                json={
                    "query": {
                        "pages": [
                            {
                                "title": "Paris",
                                "revisions": [
                                    {
                                        "revid": 12345,
                                        "timestamp": "2026-01-01T00:00:00Z",
                                        "slots": {"main": {"content": "'''Paris''' is a city."}},
                                    }
                                ],
                            }
                        ]
                    }
                },
            )
        )
        client = MediaWikiClient(API_URL, "test-agent/1.0")
        source = await client.fetch_article("Paris")
        assert source.title == "Paris"
        assert source.revid == 12345
        assert source.wikitext == "'''Paris''' is a city."
        await client.aclose()


@pytest.mark.asyncio
async def test_fetch_article_missing_raises():
    with respx.mock(base_url=API_URL) as mock:
        mock.get(params={"action": "query"}).mock(
            return_value=httpx.Response(
                200,
                json={"query": {"pages": [{"title": "Nope", "missing": True}]}},
            )
        )
        client = MediaWikiClient(API_URL, "test-agent/1.0")
        with pytest.raises(MediaWikiError, match="does not exist"):
            await client.fetch_article("Nope")
        await client.aclose()


@pytest.mark.asyncio
async def test_fetch_category_members_with_continuation():
    with respx.mock(base_url=API_URL) as mock:
        route = mock.get(url__regex=r".*").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "query": {"categorymembers": [{"title": "Article A"}, {"title": "Article B"}]},
                        "continue": {"cmcontinue": "next-token"},
                    },
                ),
                httpx.Response(
                    200,
                    json={"query": {"categorymembers": [{"title": "Article C"}]}},
                ),
            ]
        )
        client = MediaWikiClient(API_URL, "test-agent/1.0")
        titles = await client.fetch_category_members("Test category")
        assert titles == ["Article A", "Article B", "Article C"]
        assert route.call_count == 2
        await client.aclose()


@pytest.mark.asyncio
async def test_parse_wikitext_success():
    with respx.mock(base_url=API_URL) as mock:
        mock.post(API_URL).mock(
            return_value=httpx.Response(
                200,
                json={"parse": {"title": "API", "text": "<p>hi</p>", "templates": []}},
            )
        )
        client = MediaWikiClient(API_URL, "test-agent/1.0")
        result = await client.parse_wikitext("'''hi'''", title="API")
        assert result["text"] == "<p>hi</p>"
        await client.aclose()


@pytest.mark.asyncio
async def test_parse_wikitext_times_out_instead_of_hanging():
    """Regression test for a real production hang (2026-09-05, Mars and
    Earth articles): a slow/stuck action=parse response left the harness
    frozen indefinitely even with asyncio.wait_for wrapping the call,
    because wait_for waits for a cancelled *native coroutine* to actually
    finish unwinding, which httpx/httpcore didn't do promptly. Running the
    request in a thread executor sidesteps this: wait_for's timeout must
    fire and return control on schedule regardless of whether the
    underlying blocking call (simulated here with a handler far slower
    than the hard timeout) ever completes."""

    def slow_response(request: httpx.Request) -> httpx.Response:
        time.sleep(2.0)
        return httpx.Response(200, json={"parse": {"title": "API", "text": "", "templates": []}})

    with respx.mock(base_url=API_URL, assert_all_called=False) as mock:
        # assert_all_called=False: the abandoned background thread keeps
        # running past this test's own timeout window and hits the mock
        # ~1.8s later, after this `with` block has already exited — that's
        # the traded-off behavior described above, not a test bug.
        mock.post(API_URL).mock(side_effect=slow_response)
        client = MediaWikiClient(API_URL, "test-agent/1.0")
        client._hard_timeout = 0.2  # force a fast, deterministic timeout for the test
        start = time.monotonic()
        with pytest.raises(MediaWikiError, match="timed out"):
            await client.parse_wikitext("text", title="API")
        elapsed = time.monotonic() - start
        assert elapsed < 1.5, f"parse_wikitext should time out promptly, took {elapsed:.2f}s"
        await client.aclose()


@pytest.mark.asyncio
async def test_existing_templates_follows_redirects_and_marks_missing():
    sq_api = "https://sq.wikipedia.org/w/api.php"
    with respx.mock(base_url=sq_api) as mock:
        mock.get(params={"action": "query"}).mock(
            return_value=httpx.Response(
                200,
                json={
                    "query": {
                        "normalized": [{"from": "Template:infobox planet", "to": "Template:Infobox planet"}],
                        "redirects": [{"from": "Template:Infobox old", "to": "Template:Infobox new"}],
                        "pages": [
                            {"title": "Template:Infobox planet"},
                            {"title": "Template:Infobox new"},
                            {"title": "Template:Nope", "missing": True},
                        ],
                    }
                },
            )
        )
        client = MediaWikiClient(sq_api, "test-agent/1.0")
        try:
            found = await client.existing_templates(["infobox planet", "Infobox old", "Nope"])
        finally:
            await client.aclose()
    assert found == {
        "infobox planet": "Template:Infobox planet",
        "Infobox old": "Template:Infobox new",
        "Nope": None,
    }
