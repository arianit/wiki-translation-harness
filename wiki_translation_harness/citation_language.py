"""Guarantees every citation in the final translated output carries
|language=, per the skill's own requirement ("Mark the language of every
source ... on sqwiki even English sources are foreign-language, so en is
not optional"). The model can miss this; this module is a deterministic
post-processing pass that fills only what's missing, never overwrites a
value the translation already set.

Detection per citation missing |language=, using both signals when available:
1. Guess from |title= text (needs a minimum length to bother trying).
2. Visit the cited |url=, if any, and read its own declared language —
   the Content-Language header, or its <html lang="..."> attribute. This
   is a real, mechanical fact the harness can fetch but a single tool-less
   completion call cannot.
3. If both are available and disagree, the title wins: an academic
   publisher/DOI-resolver/aggregator page (Springer, JSTOR, Cambridge UP,
   ...) reports *its own* UI language this way, not the cited work's —
   confirmed in practice on a German-titled book landing on an English-UI
   Springer page. The title is direct evidence about the work itself.
4. If neither signal is available, the parameter is left unset rather
   than filled with a low-confidence guess.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

import httpx
import mwparserfromhell as mwp
from langdetect import LangDetectException, detect_langs

CITATION_TEMPLATE_NAMES = {
    "cite web",
    "cite book",
    "cite news",
    "cite journal",
    "cite magazine",
    "cite press release",
    "cite report",
    "cite thesis",
    "cite av media",
    "cite av media notes",
    "cite encyclopedia",
    "cite conference",
    "cite interview",
    "cite podcast",
    "cite document",
    "cite court",
    "cite speech",
    "citation",
}

# Confirmed in practice: a citation-dense article had every parameter name
# on one {{Cite web}} call translated into Albanian (date -> data, title ->
# titulli, website -> faqja, access-date -> data-e-përdorimit), which
# Module:Citation/CS1 then silently ignored as unrecognized parameters,
# dropping those fields from the rendered citation. The verified-facts note
# telling the model to keep parameter names English (see verification.py)
# catches most of these, but it's a soft instruction, not a hard guarantee
# — this is the deterministic backstop: known Albanian mistranslations of
# standard CS1 parameter names, renamed back mechanically. A param is only
# renamed when its English equivalent isn't already present on the same
# template (ambiguous otherwise — could be a genuine duplicate, not a
# straightforward rename).
ALBANIAN_TO_ENGLISH_CS1_PARAMS = {
    "titulli": "title",
    "titull": "title",
    "autori": "author",
    "autor": "author",
    "autorët": "authors",
    "autoret": "authors",
    "lidhja-e-autorit": "author-link",
    "lidhje-autor": "author-link",
    "lidhja-autorit": "author-link",
    "autor-link": "author-link",
    "autor-linku": "author-link",
    "mbiemri": "last",
    "mbiemër": "last",
    "mbiemer": "last",
    "emri": "first",
    "emër": "first",
    "emer": "first",
    "redaktori": "editor",
    "redaktor": "editor",
    "redaktorët": "editors",
    "redaktoret": "editors",
    "botuesi": "publisher",
    "botues": "publisher",
    "faqja": "website",
    "uebfaqja": "website",
    "faqja-e-internetit": "website",
    "data": "date",
    "data-e-përdorimit": "access-date",
    "data-e-perdorimit": "access-date",
    "data-e-hyrjes": "access-date",
    "data-qasjes": "access-date",
    "data-e-aksesit": "access-date",
    "data-arkivimi": "archive-date",
    "data-e-arkivimit": "archive-date",
    "url-arkivi": "archive-url",
    "url-i-arkivit": "archive-url",
    "viti": "year",
    "vit": "year",
    "faqe": "page",
    "faqja-e-librit": "page",
    "faqet": "pages",
    "vëllimi": "volume",
    "vellimi": "volume",
    "vëllim": "volume",
    "numri": "issue",
    "numër": "issue",
    "vendndodhja": "location",
    "vendndodhje": "location",
    "vendi": "location",
    "vend": "location",
    "vepra": "work",
    "vepër": "work",
    "kapitulli": "chapter",
    "kapitull": "chapter",
    "citimi": "quote",
    "citim": "quote",
    "përkthyesi": "translator",
    "përkthyes": "translator",
    "perkthyesi": "translator",
    "titulli-përkthyer": "trans-title",
    "titull-përkthyer": "trans-title",
    "titulli-perkthyer": "trans-title",
    "revistë": "journal",
    "revista": "journal",
    "gazetë": "newspaper",
    "seria": "series",
    "seri": "series",
    "gjuha": "language",
    "formati": "format",
    "lloji": "type",
    "tipi": "type",
    "botimi": "edition",
    "departamenti": "department",
    "identifikuesi": "id",
    "fq": "page",
}

_TRAILING_DIGITS_RE = re.compile(r"(\d+)$")


def _translate_param_name(name: str) -> str | None:
    """Looks up name in ALBANIAN_TO_ENGLISH_CS1_PARAMS, stripping and
    re-appending a trailing digit first (CS1's numbered-author/editor
    convention: redaktor1 -> editori -> editor1, matching author1/2/3...,
    last1/2/3..., editor1/2/3... on the English side)."""
    match = _TRAILING_DIGITS_RE.search(name)
    suffix = match.group(1) if match else ""
    base = name[: len(name) - len(suffix)] if suffix else name
    english_base = ALBANIAN_TO_ENGLISH_CS1_PARAMS.get(base)
    return f"{english_base}{suffix}" if english_base else None

_HTML_LANG_RE = re.compile(r'<html[^>]*\blang=["\']?([a-zA-Z-]+)', re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

_MIN_TITLE_LENGTH_FOR_GUESS = 20
_MIN_TEXT_LENGTH_FOR_DETECTION = 50
_MIN_CONFIDENCE = 0.90


def _is_citation_template(name: str) -> bool:
    return name.strip().lower() in CITATION_TEMPLATE_NAMES


def _detect_confident(text: str) -> str | None:
    """detect() always returns its single best guess with no notion of
    "I don't know" — confirmed in practice: a 9-character title ("Some
    Page") was confidently called Danish, and a 4-character one ("Home")
    was 99% Danish. detect_langs() exposes the probability distribution,
    so a genuinely low-confidence top guess can be rejected instead of
    silently writing a wrong |language= value."""
    try:
        candidates = detect_langs(text)
    except LangDetectException:
        return None
    if not candidates or candidates[0].prob < _MIN_CONFIDENCE:
        return None
    return candidates[0].lang


def guess_language_from_title(title: str) -> str | None:
    """Heuristic only — a citation title alone is short and often has
    untranslated proper nouns, so this is deliberately a fallback, used
    only when no URL is available to check directly."""
    title = title.strip()
    if len(title) < _MIN_TITLE_LENGTH_FOR_GUESS:
        return None
    return _detect_confident(title)


async def detect_source_language(client: httpx.AsyncClient, url: str, timeout: float = 10.0) -> str | None:
    """Fetches url and reads its own declared language. Never raises —
    any fetch/parse failure just means "couldn't determine", not an error
    that should interrupt translation of an otherwise-fine article.

    Wrapped in asyncio.wait_for as a hard, independent backstop: httpx's own
    `timeout=` has been confirmed, twice, to not reliably fire under real
    network conditions, hanging asyncio.gather (and everything waiting on
    it) indefinitely since it awaits every concurrent task to finish."""
    try:
        resp = await asyncio.wait_for(
            client.get(url, follow_redirects=True, timeout=timeout), timeout=timeout + 15.0
        )
    except (httpx.HTTPError, asyncio.TimeoutError):
        return None

    if resp.status_code >= 400:
        return None

    content_type = resp.headers.get("content-type", "")
    if content_type and "text" not in content_type and "html" not in content_type:
        return None

    content_language = resp.headers.get("content-language")
    if content_language:
        return content_language.split(",")[0].split("-")[0].strip().lower()

    try:
        text = resp.text
    except Exception:
        return None

    match = _HTML_LANG_RE.search(text[:5000])
    if match:
        return match.group(1).split("-")[0].lower()

    visible = _WHITESPACE_RE.sub(" ", _TAG_RE.sub(" ", text[:20000])).strip()
    if len(visible) < _MIN_TEXT_LENGTH_FOR_DETECTION:
        return None
    return _detect_confident(visible[:2000])


@dataclass
class CitationLanguageResult:
    patched_wikitext: str
    filled: dict[str, str]  # citation title (or template name) -> detected language
    attempted: int
    fetched_urls: int


async def fill_missing_citation_languages(
    wikitext: str,
    http_client: httpx.AsyncClient,
    max_url_fetches: int = 40,
    concurrency: int = 5,
    fetch_timeout: float = 10.0,
) -> CitationLanguageResult:
    """Mechanical post-processing pass over the final assembled article —
    never touches prose, only adds a missing |language= parameter."""
    code = mwp.parse(wikitext)
    templates = [t for t in code.filter_templates(recursive=True) if _is_citation_template(str(t.name))]
    missing = [t for t in templates if not t.has("language", ignore_empty=True)]

    if not missing:
        return CitationLanguageResult(patched_wikitext=wikitext, filled={}, attempted=0, fetched_urls=0)

    sem = asyncio.Semaphore(concurrency)
    fetched_urls = 0

    async def resolve(tmpl: mwp.nodes.Template) -> str | None:
        nonlocal fetched_urls
        title = str(tmpl.get("title").value).strip() if tmpl.has("title", ignore_empty=True) else ""
        title_guess = guess_language_from_title(title)

        url = str(tmpl.get("url").value).strip() if tmpl.has("url", ignore_empty=True) else ""
        url_lang = None
        if url and fetched_urls < max_url_fetches:
            fetched_urls += 1
            async with sem:
                url_lang = await detect_source_language(http_client, url, timeout=fetch_timeout)

        if url_lang and title_guess and url_lang != title_guess:
            # Academic-publisher/aggregator/DOI-resolver pages (Springer,
            # JSTOR, Cambridge UP, ...) report *their own* UI language via
            # Content-Language/<html lang>, not the cited work's language —
            # confirmed in practice: a clearly German-titled book landed on
            # an English-UI Springer page and was mis-tagged "en". The
            # title is direct evidence about the cited work itself, so it
            # wins when the two signals disagree.
            return title_guess
        return url_lang or title_guess

    results = await asyncio.gather(*(resolve(t) for t in missing), return_exceptions=True)

    filled: dict[str, str] = {}
    for tmpl, result in zip(missing, results):
        if isinstance(result, BaseException) or not result:
            continue
        tmpl.add("language", result)
        key = str(tmpl.get("title").value).strip() if tmpl.has("title", ignore_empty=True) else str(tmpl.name).strip()
        filled[key] = result

    return CitationLanguageResult(
        patched_wikitext=str(code), filled=filled, attempted=len(missing), fetched_urls=fetched_urls
    )


@dataclass
class CitationParamFixResult:
    patched_wikitext: str
    renamed: dict[str, str]  # albanian param name -> english param name (count of occurrences via len())


def fix_citation_param_names(wikitext: str) -> CitationParamFixResult:
    """Deterministic backstop for citation parameter names mistranslated
    into Albanian (see ALBANIAN_TO_ENGLISH_CS1_PARAMS) — renames them back
    to their English CS1 equivalents. Never touches parameter values, only
    names, and only on recognized citation templates. Skips a rename if the
    English name is already present on the same template call (ambiguous:
    could be a genuine duplicate rather than a straightforward mistranslation,
    not the harness's call to resolve)."""
    code = mwp.parse(wikitext)
    templates = [t for t in code.filter_templates(recursive=True) if _is_citation_template(str(t.name))]

    renamed: dict[str, str] = {}
    for tmpl in templates:
        existing_names = {str(p.name).strip().lower() for p in tmpl.params}
        for param in tmpl.params:
            albanian_name = str(param.name).strip().lower()
            english_name = _translate_param_name(albanian_name)
            if not english_name or english_name in existing_names:
                continue
            param.name = english_name
            existing_names.add(english_name)
            existing_names.discard(albanian_name)
            renamed[albanian_name] = english_name

    return CitationParamFixResult(patched_wikitext=str(code), renamed=renamed)


_SHORT_FOOTNOTE_TEMPLATE_NAMES = {"sfn", "sfnp", "sfnm", "harvnb", "harv", "harvp"}

# The model sometimes translates sfn's |page= to |f= (Albanian "faqe" -> "f")
# and |pages= to |ff= ("faqet" -> "ff"), or writes "f. 161" as a positional
# value. sfn only understands |p= and |pp= — these are mechanically renamed.
_SFN_PAGE_PARAM_FIXES: dict[str, str] = {
    "f": "p",
    "ff": "pp",
    "fq": "p",
    "f.": "p",
    "ff.": "pp",
    "fq.": "p",
    "faqe": "p",
    "faqet": "pp",
    "page": "p",
    "pages": "pp",
}
# Pattern matching a positional value that looks like a translated page
# reference: "f. 161", "f.161", "fq. 44", "f 161-170", etc.
_SFN_POSITIONAL_PAGE_RE = re.compile(r"^f[q]?\.?\s*(\d.*)$", re.IGNORECASE)


@dataclass
class SfnParamFixResult:
    patched_wikitext: str
    renamed: dict[str, str]  # description of each fix applied


def fix_sfn_param_names(wikitext: str) -> SfnParamFixResult:
    """Deterministic fix for {{sfn}}/{{harvnb}} parameter names mistranslated
    into Albanian: |f= -> |p=, |ff= -> |pp=, and positional values like
    'f. 161' converted to named |p=161. The sfn template only understands
    positional params 1=author, 2=year — everything else must be named."""
    code = mwp.parse(wikitext)
    templates = [
        t
        for t in code.filter_templates(recursive=True)
        if str(t.name).strip().lower() in _SHORT_FOOTNOTE_TEMPLATE_NAMES
    ]

    renamed: dict[str, str] = {}
    for tmpl in templates:
        # Fix named params: |f= -> |p=, |ff= -> |pp=
        for param in tmpl.params:
            name = str(param.name).strip().lower()
            if name in _SFN_PAGE_PARAM_FIXES:
                new_name = _SFN_PAGE_PARAM_FIXES[name]
                renamed[f"|{name}="] = f"|{new_name}="
                param.name = new_name

        # Fix positional params: a 3rd positional value like "f. 161" -> named |p=161
        for param in tmpl.params:
            if param.showkey:
                continue
            pos_name = str(param.name).strip()
            if pos_name in ("1", "2"):
                continue  # author and year, leave alone
            value = str(param.value).strip()
            m = _SFN_POSITIONAL_PAGE_RE.match(value)
            if m:
                page_val = m.group(1).strip()
                target = "pp" if "-" in page_val or "," in page_val else "p"
                renamed[f"positional '{value}'"] = f"|{target}={page_val}"
                param.name = target
                param.value = page_val
                param.showkey = True

    return SfnParamFixResult(patched_wikitext=str(code), renamed=renamed)


# Only the templates that generate their OWN <ref> internally (via
# Module:Footnotes' {{#tag:ref|...}}) belong here. {{harvnb}}/{{harv}}/
# {{harvp}} deliberately do NOT self-wrap — they render a bare parenthetical
# link and are routinely wrapped in an explicit <ref> by design, so
# unwrap_redundant_sfn_ref must not touch them (unlike
# _SHORT_FOOTNOTE_TEMPLATE_NAMES above, which groups all six for the |ps=
# dedup check, where self-wrapping doesn't matter).
_SELF_WRAPPING_FOOTNOTE_TEMPLATE_NAMES = {"sfn", "sfnp", "sfnm"}


@dataclass
class RefUnwrapResult:
    patched_wikitext: str
    unwrapped: list[str]  # each sfn/sfnp/sfnm call that had its wrapper stripped


def unwrap_redundant_sfn_ref(wikitext: str) -> RefUnwrapResult:
    """Strips a redundant <ref>...</ref> wrapped directly around a lone
    {{sfn}}/{{sfnp}}/{{sfnm}} call.

    Confirmed live against sq.wikipedia (2026-09-22, 'Conservatism'): {{sfn}}
    already expands to its own <ref name="FOOTNOTE...">...</ref> internally,
    so wrapping it in another explicit <ref> nests one <ref> inside another.
    MediaWiki's Cite extension doesn't reject this outright, but it corrupts
    the extension's usage tracking for the auto-generated FOOTNOTE... name —
    live action=parse rendered it as "Cite error: <ref> tag with name
    'FOOTNOTE...' defined in <references> is not used in prior text", even
    though the (only) occurrence of that name was sitting right there. The
    model does this occasionally despite using bare {{sfn}} correctly
    everywhere else in the same article; whether the LLM-based
    assembly-repair loop (pipeline.py) can even fix it once live-validate
    flags it is beside the point — a single {{sfn}} occurrence is never a
    named ref another chunk could plausibly "reuse", so there's nothing for
    a repair round to reconcile against, and broadcasting the fix to every
    chunk (see pipeline.py's _chunk_mentions_ref) risks manufacturing a real
    same-name collision instead of just removing the wrapper. This is purely
    mechanical, so it belongs here rather than in a repair prompt.

    Only unwraps a *bare* <ref> (no name=) whose entire trimmed content is
    that one template call and nothing else — a named ref, or a ref with
    other prose alongside the sfn call, is left for repair since unwrapping
    either could silently change meaning (a name= may be relied on for
    reuse elsewhere) or drop content (accompanying prose)."""
    code = mwp.parse(wikitext)
    unwrapped: list[str] = []
    for tag in code.filter_tags(recursive=True):
        if str(tag.tag).strip().lower() != "ref" or tag.has("name"):
            continue
        contents = tag.contents
        if contents is None:
            continue
        templates = contents.filter_templates(recursive=False)
        if len(templates) != 1:
            continue
        tmpl = templates[0]
        if str(tmpl.name).strip().lower() not in _SELF_WRAPPING_FOOTNOTE_TEMPLATE_NAMES:
            continue
        if str(contents).strip() != str(tmpl).strip():
            continue  # extra text beside the sfn call -- not a clean wrap
        unwrapped.append(str(tmpl).strip())
        code.replace(tag, str(tmpl))
    return RefUnwrapResult(patched_wikitext=str(code), unwrapped=unwrapped)


# sfn/sfnp's non-self-wrapping twins (same positional/page parameters).
_SFN_TO_HARV = {"sfn": "harvnb", "sfnp": "harvp"}


@dataclass
class NestedSfnResult:
    patched_wikitext: str
    converted: list[str]  # each sfn/sfnp call renamed to its harv* twin


def convert_nested_sfn_to_harvnb(wikitext: str) -> NestedSfnResult:
    """Renames {{sfn}}/{{sfnp}} to {{harvnb}}/{{harvp}} wherever they sit
    inside an explicit <ref>...</ref>.

    unwrap_redundant_sfn_ref only handles a bare <ref> around ONE sfn call.
    The other shape — a <ref> bundling several citations, e.g.
    <ref>{{Sfn|A|1980|p=1}}, {{Sfn|B|2001|p=2}}</ref> — is what the skill's
    "bundled {{harvnb}} inside one <ref> -> consecutive {{Sfn}} calls" row
    produces when the model swaps the template but keeps the wrapper. The
    English source had {{harvnb}} there, which is correct (it renders a
    bare link and is meant to live inside a <ref>); {{sfn}} self-wraps, so
    the swap nests refs and Cite flags every FOOTNOTE... name as "defined
    in <references> but not used" (confirmed live, Alexander the Great,
    2026-10-04: 15 such errors from 5 bundled refs). Restoring harvnb keeps
    the one-footnote-many-sources shape the source had, instead of
    exploding it into several footnotes. Run after unwrap_redundant_sfn_ref
    so a lone bare wrapper is still unwrapped to a plain {{sfn}}."""
    code = mwp.parse(wikitext)
    converted: list[str] = []
    for tag in code.filter_tags(recursive=True):
        if str(tag.tag).strip().lower() != "ref" or tag.contents is None:
            continue
        for tmpl in tag.contents.filter_templates(recursive=True):
            bare = str(tmpl.name).strip()
            target = _SFN_TO_HARV.get(bare.lower())
            if target is None:
                continue
            converted.append(str(tmpl).strip())
            # Keep the original name's surrounding whitespace ("{{Sfn |...").
            tmpl.name = str(tmpl.name).replace(bare, target)
    return NestedSfnResult(patched_wikitext=str(code), converted=converted)


_REF_TOKEN_RE = re.compile(r"<ref\b[^>]*?/\s*>|<ref\b[^>]*>|</ref\s*>", re.IGNORECASE)
_REF_ATTR_RE = re.compile(r"(\w+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s\"'/>]+))")


@dataclass
class NestedRefNoteResult:
    patched_wikitext: str
    converted: list[str]  # opening tag of each note rewritten as {{refn}}


def _escape_top_level_pipes(text: str) -> str:
    """Escapes `|` outside {{...}}, [[...]] and <ref>...</ref> as {{!}} so
    the text survives as a single template argument. Pipes inside <ref>
    content are safe: the preprocessor strips extension tags before it
    splits template arguments."""
    out: list[str] = []
    braces = brackets = ref_depth = 0
    i = 0
    while i < len(text):
        token = _REF_TOKEN_RE.match(text, i)
        if token:
            tag = token.group(0)
            if tag.startswith("</"):
                ref_depth = max(0, ref_depth - 1)
            elif not tag.rstrip(">").rstrip().endswith("/"):
                ref_depth += 1
            out.append(tag)
            i = token.end()
            continue
        pair = text[i : i + 2]
        if pair in ("{{", "}}", "[[", "]]"):
            if pair == "{{":
                braces += 1
            elif pair == "}}":
                braces = max(0, braces - 1)
            elif pair == "[[":
                brackets += 1
            else:
                brackets = max(0, brackets - 1)
            out.append(pair)
            i += 2
            continue
        ch = text[i]
        out.append("{{!}}" if ch == "|" and not (braces or brackets or ref_depth) else ch)
        i += 1
    return "".join(out)


def convert_nested_ref_notes_to_refn(wikitext: str) -> NestedRefNoteResult:
    """Rewrites a <ref> whose content contains another <ref> as {{refn}}.

    English Wikipedia writes explanatory notes that cite their own sources
    as {{refn|group=note|Text.<ref>{{cite ...}}</ref>}} — {{refn}} expands
    through {{#tag:ref}}, which is what lets a ref sit inside a note. The
    model sometimes "simplifies" that to <ref group="note">Text.<ref>...
    </ref></ref>, which Cite cannot parse: the inner </ref> closes the
    outer note early and the leftover </ref> renders as text (confirmed in
    Albert Einstein and Vietnam War partial drafts, 2026-10-06). Restoring
    {{refn}} keeps the group, name and the note's sources exactly as
    translated. sq.wikipedia's Stampa:Refn takes |name=, |group= and |1=.

    Works line by line: notes are single-paragraph, and a ref definition
    cut by chunking (opener on one line, no closer) must never be mistaken
    for a note wrapping everything up to some later stray </ref>. Nesting
    deeper than one level is left alone."""
    converted: list[str] = []
    lines = wikitext.split("\n")
    for line_no, line in enumerate(lines):
        if line.lower().count("<ref") < 2:
            continue
        pieces: list[str] = []
        pos = 0
        depth = max_depth = 0
        outer: re.Match[str] | None = None
        for token in _REF_TOKEN_RE.finditer(line):
            tag = token.group(0)
            if tag.startswith("</"):
                if depth == 0:
                    continue  # stray closer; leave it alone
                depth -= 1
                if depth == 0 and outer is not None:
                    if max_depth == 2:
                        attrs = {
                            m.group(1).lower(): m.group(2) or m.group(3) or m.group(4) or ""
                            for m in _REF_ATTR_RE.finditer(outer.group(0)[4:])
                        }
                        params = "".join(
                            f"|{key}={attrs[key]}" for key in ("group", "name") if attrs.get(key)
                        )
                        content = _escape_top_level_pipes(line[outer.end() : token.start()].strip())
                        pieces.append(line[pos : outer.start()])
                        pieces.append(f"{{{{refn{params}|1={content}}}}}")
                        pos = token.end()
                        converted.append(outer.group(0))
                    outer = None
            elif tag.rstrip(">").rstrip().endswith("/"):
                if depth >= 1:
                    max_depth = max(max_depth, depth + 1)
            else:
                if depth == 0:
                    outer, max_depth = token, 1
                depth += 1
                max_depth = max(max_depth, depth)
        if pos:
            pieces.append(line[pos:])
            lines[line_no] = "".join(pieces)
    return NestedRefNoteResult(patched_wikitext="\n".join(lines), converted=converted)


@dataclass
class FootnoteDedupeResult:
    patched_wikitext: str
    canonicalized: dict[str, int]  # identity key -> number of calls reconciled


def dedupe_short_footnotes(wikitext: str) -> FootnoteDedupeResult:
    """{{sfn}}/{{harvnb}} auto-generate a shared anchor from their
    author+year+page parameters (everything except |ps=). Two calls
    sharing that identity must render byte-identical or MediaWiki's Cite
    system raises "Invalid <ref> tag ... defined multiple times with
    different content" — confirmed in practice: the exact same source
    citation, split across two independently-translated chunks, came back
    with its |ps= quote paraphrased slightly differently each time (even
    inconsistent transliteration of a name within the same quote). Every
    occurrence sharing an identity is canonicalized to the first one's
    |ps= value — mechanical reconciliation, not a translation-quality
    judgment about which paraphrase reads better."""
    code = mwp.parse(wikitext)
    templates = [
        t
        for t in code.filter_templates(recursive=True)
        if str(t.name).strip().lower() in _SHORT_FOOTNOTE_TEMPLATE_NAMES
    ]

    groups: dict[tuple, list] = {}
    for tmpl in templates:
        identity = tuple(
            (str(p.name).strip().lower(), str(p.value).strip())
            for p in tmpl.params
            if str(p.name).strip().lower() != "ps"
        )
        groups.setdefault(identity, []).append(tmpl)

    canonicalized: dict[str, int] = {}
    for identity, group in groups.items():
        if len(group) < 2:
            continue
        ps_values = [str(t.get("ps").value) if t.has("ps", ignore_empty=True) else None for t in group]
        if len(set(ps_values)) <= 1:
            continue  # already consistent, nothing to reconcile

        canonical_ps = ps_values[0]
        for tmpl, ps in zip(group, ps_values):
            if ps == canonical_ps:
                continue
            if canonical_ps is None:
                tmpl.remove("ps")
            elif tmpl.has("ps"):
                tmpl.get("ps").value = canonical_ps
            else:
                tmpl.add("ps", canonical_ps)

        key = "|".join(f"{name}={value}" for name, value in identity)
        canonicalized[key] = len(group)

    return FootnoteDedupeResult(patched_wikitext=str(code), canonicalized=canonicalized)
