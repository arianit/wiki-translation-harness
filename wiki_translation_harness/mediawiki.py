"""MediaWiki Action API client.

Source-fetching (fetch_article, fetch_category_members) is raw wikitext
only, never rendered HTML. parse_wikitext is the one exception — it exists
solely so live_validator.py can inspect a live render for defects (Lua/cite
errors, missing templates) that don't show up in raw wikitext at all.
"""

from __future__ import annotations

import asyncio

import httpx

from wiki_translation_harness.models import ArticleSource

# Hard, independent backstop for every individual MediaWiki API call —
# httpx's own `timeout=` on the client has been confirmed, twice, to not
# reliably fire under real network conditions, hanging the whole batch run
# indefinitely. Generous enough to never trip on a merely-slow-but-alive
# request; its only job is to guarantee eventual cancellation when a socket
# genuinely hangs.
_HARD_TIMEOUT_MARGIN_S = 30.0


class MediaWikiError(Exception):
    pass


def wiki_api_url_for_lang(lang: str) -> str:
    """Standard Wikipedia language-subdomain API endpoint for a language code."""
    return f"https://{lang}.wikipedia.org/w/api.php"


class MediaWikiClient:
    def __init__(self, api_url: str, user_agent: str, source_lang: str = "en", timeout: float = 30.0):
        self.api_url = api_url
        self.source_lang = source_lang
        self._user_agent = user_agent
        self._timeout = timeout
        self._hard_timeout = timeout + _HARD_TIMEOUT_MARGIN_S
        self._client = httpx.AsyncClient(
            headers={"User-Agent": user_agent},
            timeout=timeout,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "MediaWikiClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def fetch_article(
        self, title: str, revid: int | None = None, follow_redirects: bool = False
    ) -> ArticleSource:
        """Fetch raw wikitext + revid/timestamp for a single title via action=query.
        With `revid`, fetches that exact revision instead of the current one.
        With `follow_redirects`, a redirect resolves to its target page (the
        returned title is the target's) instead of its own `#REDIRECT` text.
        Off by default: source articles are saved, locked and checked as done
        under the requested title, so a source fetch must not rename it."""
        params = {
            "action": "query",
            "prop": "revisions",
            "revids" if revid else "titles": revid or title,
            "rvprop": "content|ids|timestamp",
            "rvslots": "main",
            "formatversion": "2",
            "format": "json",
        }
        if follow_redirects and not revid:
            # Not with revids: resolving there would swap the pinned revision
            # for the redirect target's current one.
            params["redirects"] = "1"
        resp = await asyncio.wait_for(
            self._client.get(self.api_url, params=params), timeout=self._hard_timeout
        )
        resp.raise_for_status()
        data = resp.json()

        pages = data.get("query", {}).get("pages", [])
        if not pages:
            raise MediaWikiError(f"No page data returned for title {title!r}")
        page = pages[0]
        if page.get("missing"):
            raise MediaWikiError(f"Article {title!r} does not exist on {self.api_url}")

        revisions = page.get("revisions")
        if not revisions:
            raise MediaWikiError(f"Article {title!r} has no revisions")
        rev = revisions[0]
        content = rev.get("slots", {}).get("main", {}).get("content")
        if content is None:
            raise MediaWikiError(f"Article {title!r} revision has no wikitext content")

        return ArticleSource(
            title=page.get("title", title),
            wikitext=content,
            revid=rev.get("revid"),
            revision_timestamp=rev.get("timestamp"),
            source_lang=self.source_lang,
        )

    async def fetch_category_members(self, category: str, limit: int = 500) -> list[str]:
        """List article titles (main namespace only) in a category, following continuation."""
        cat_title = category if category.startswith("Category:") else f"Category:{category}"
        titles: list[str] = []
        cmcontinue: str | None = None

        while True:
            params: dict[str, str] = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": cat_title,
                "cmnamespace": "0",
                "cmlimit": "500",
                "format": "json",
                "formatversion": "2",
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue

            resp = await asyncio.wait_for(
                self._client.get(self.api_url, params=params), timeout=self._hard_timeout
            )
            resp.raise_for_status()
            data = resp.json()

            members = data.get("query", {}).get("categorymembers", [])
            titles.extend(m["title"] for m in members)

            if len(titles) >= limit:
                return titles[:limit]

            cont = data.get("continue", {}).get("cmcontinue")
            if not cont:
                break
            cmcontinue = cont

        return titles

    async def existing_templates(self, names: list[str]) -> dict[str, str | None]:
        """Bare template name -> its full title on this wiki ("Stampa:Infobox
        planet"), following redirects, or None when no such template exists.
        Asks for "Template:<name>", which every wiki accepts as the canonical
        namespace name."""
        results: dict[str, str | None] = {}
        for start in range(0, len(names), 50):
            batch = names[start : start + 50]
            params = {
                "action": "query",
                "titles": "|".join(f"Template:{name}" for name in batch),
                "redirects": "1",
                "format": "json",
                "formatversion": "2",
            }
            resp = await asyncio.wait_for(
                self._client.get(self.api_url, params=params), timeout=self._hard_timeout
            )
            resp.raise_for_status()
            query = resp.json().get("query", {})
            normalized = {m["from"]: m["to"] for m in query.get("normalized", [])}
            redirects = {m["from"]: m["to"] for m in query.get("redirects", [])}
            pages = {page["title"]: not page.get("missing", False) for page in query.get("pages", [])}
            for name in batch:
                title = f"Template:{name}"
                title = normalized.get(title, title)
                title = redirects.get(title, title)
                results[name] = title if pages.get(title) else None
        return results

    async def expand_templates(self, text: str, title: str) -> str:
        """Expands `text` on this wiki via action=expandtemplates, with `title`
        as the page context (templates such as itwiki's {{Popolazione}} read
        data keyed by the page they appear on). Source-side use only: lets
        the pipeline turn a data template the translator cannot evaluate
        into the text it produces."""
        params = {
            "action": "expandtemplates",
            "text": text,
            "title": title,
            "prop": "wikitext",
            "format": "json",
            "formatversion": "2",
        }
        resp = await asyncio.wait_for(
            self._client.post(self.api_url, data=params), timeout=self._hard_timeout
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise MediaWikiError(f"action=expandtemplates failed for title {title!r}: {data['error']}")
        return data.get("expandtemplates", {}).get("wikitext", "")

    async def existing_pages(self, titles: list[str]) -> dict[str, bool]:
        """Title -> whether a page of that title exists, following redirects.
        A "#section" suffix is ignored."""
        results: dict[str, bool] = {}
        for start in range(0, len(titles), 50):
            batch = titles[start : start + 50]
            params = {
                "action": "query",
                "titles": "|".join(t.split("#")[0] for t in batch),
                "redirects": "1",
                "format": "json",
                "formatversion": "2",
            }
            resp = await asyncio.wait_for(
                self._client.get(self.api_url, params=params), timeout=self._hard_timeout
            )
            resp.raise_for_status()
            query = resp.json().get("query", {})
            normalized = {m["from"]: m["to"] for m in query.get("normalized", [])}
            redirects = {m["from"]: m["to"] for m in query.get("redirects", [])}
            pages = {page["title"]: not page.get("missing", False) for page in query.get("pages", [])}
            for title in batch:
                name = title.split("#")[0]
                name = normalized.get(name, name)
                name = redirects.get(name, name)
                results[title] = pages.get(name, False)
        return results

    async def parse_wikitext(self, text: str, title: str = "API") -> dict:
        """Live-render wikitext via action=parse, for post-translation
        validation only (Lua/cite errors, missing templates, redlinks) —
        never for fetching source content. POST, not GET: translated
        articles can exceed URL length limits. `title` only affects
        namespace-relative link resolution in the render; it does not need
        to be a real page. formatversion=2 is required for `templates[].exists`
        to be a real boolean (formatversion=1 encodes it as a
        present-but-empty-string key, which is easy to misread as falsy).

        Run as a *synchronous* request inside a thread-pool executor rather
        than through self._client (httpx.AsyncClient), unlike every other
        method here. Confirmed in production (2026-09-05, Mars and Earth):
        this specific call — a large POST rendered through Scribunto on the
        target wiki — hung well past `_hard_timeout` even with the same
        asyncio.wait_for backstop used everywhere else, with the process
        left idle (no CPU, no socket activity) rather than erroring. The
        likely cause is asyncio.wait_for's documented limitation: on
        timeout it cancels the awaited task and then *waits for that
        cancellation to be honored* — if httpx/httpcore doesn't process
        cancellation promptly while blocked on a stuck socket read, the
        wait never returns. A `loop.run_in_executor` future doesn't have
        this problem: cancelling it (what wait_for does on timeout)
        detaches the asyncio-visible future immediately and lets the
        caller proceed, regardless of whether the underlying thread's
        blocking call ever completes — the thread is abandoned, not
        awaited. This trades a possible leaked thread (bounded, harmless)
        for a guaranteed-bounded wait (the actual bug being fixed)."""
        params = {
            "action": "parse",
            "contentmodel": "wikitext",
            "text": text,
            "title": title,
            "prop": "text|templates",
            "disablelimitreport": "1",
            "formatversion": "2",
            "format": "json",
        }

        def do_post() -> httpx.Response:
            with httpx.Client(headers={"User-Agent": self._user_agent}, timeout=self._timeout) as client:
                return client.post(self.api_url, data=params)

        loop = asyncio.get_running_loop()
        try:
            resp = await asyncio.wait_for(loop.run_in_executor(None, do_post), timeout=self._hard_timeout)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise MediaWikiError(
                f"action=parse timed out for title {title!r} after {self._hard_timeout:.0f}s"
            ) from exc
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            raise MediaWikiError(f"action=parse failed for title {title!r}: {data['error']}")
        parse = data.get("parse")
        if parse is None:
            raise MediaWikiError(f"action=parse returned no 'parse' result for title {title!r}")
        return parse


class MediaWikiClientPool:
    """Lazily builds and reuses one MediaWikiClient per source language.

    A single run can mix source languages (see sources.parse_source_ref),
    so there's no one fixed source wiki to fetch from. `default_lang`'s API
    endpoint can be overridden (e.g. a non-Wikipedia wiki); every other
    language resolves generically to https://{lang}.wikipedia.org/w/api.php.
    """

    def __init__(
        self,
        user_agent: str,
        default_lang: str,
        default_api_url: str | None = None,
    ):
        self._user_agent = user_agent
        self._default_lang = default_lang
        self._default_api_url = default_api_url
        self._clients: dict[str, MediaWikiClient] = {}

    def get(self, lang: str) -> MediaWikiClient:
        if lang not in self._clients:
            if lang == self._default_lang and self._default_api_url:
                api_url = self._default_api_url
            else:
                api_url = wiki_api_url_for_lang(lang)
            self._clients[lang] = MediaWikiClient(api_url, self._user_agent, lang)
        return self._clients[lang]

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
