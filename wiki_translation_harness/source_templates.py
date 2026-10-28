"""Turns itwiki's data templates into plain wikitext before translation.

{{Popolazione|ITA}}, {{Demografia/<comune>}} and {{Divisione amministrativa}}
compute their content from Lua/data modules on it.wikipedia.org, so the
translator only sees an opaque call and drops it -- the semantic review then
flags a missing population figure, an empty "Evoluzione demografica" section
and a missing infobox on every Italian-source municipality.

Each is replaced on the source side:
- {{Popolazione|...}} -> the number itwiki renders (action=expandtemplates).
- {{Demografia/...}} -> a year/population wikitable built from the rendered
  timeline data, with the ISTAT sources as a <ref>.
- {{Divisione amministrativa}} -> {{Infobox settlement}} with English
  parameter names (sqwiki's copy rejects Albanian ones). Values stay Italian
  so the translator renders them like the rest of the article.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import mwparserfromhell as mwp

from wiki_translation_harness.mediawiki import MediaWikiClient

logger = logging.getLogger(__name__)

_POPULATION_RE = re.compile(r"^[\d\s .,]+$")
_PLOT_BAR_RE = re.compile(r"bar:(\d{4})\s+from:0\s+till:\s*([\d\s .]+?)\s*$", re.MULTILINE)
_SOURCE_LINK_RE = re.compile(r"\[(https?://[^\s\]]+)\s+<span[^>]*>(.*?)</span>\]")

_COUNTRIES = {"ITA": "[[Italia]]"}

# Infobox parameter -> Divisione amministrativa field(s), first non-empty wins.
_INFOBOX_FIELDS: list[tuple[str, tuple[str, ...]]] = [
    ("name", ("Nome",)),
    ("image_skyline", ("Panorama",)),
    ("image_caption", ("Didascalia",)),
    ("image_flag", ("Bandiera",)),
    ("image_shield", ("Stemma",)),
    ("leader_name", ("Amministratore locale",)),
    ("leader_party", ("Partito",)),
    ("elevation_m", ("Altitudine",)),
    ("area_total_km2", ("Superficie",)),
    ("image_map", ("Mappa",)),
    ("map_caption", ("Didascalia mappa",)),
]
# Fields with no infobox parameter of their own: shown as labelled rows.
_EXTRA_FIELDS = (
    "Nome abitanti",
    "Divisioni confinanti",
    "Sottodivisioni",
    "Lingue",
    "Zona sismica",
    "Gradi giorno",
    "Patrono",
    "Festivo",
)
_MAX_EXTRA_ROWS = 8  # blank_name_sec1 + blank1..blank7_name_sec1


@dataclass
class DataTemplateResult:
    wikitext: str
    population: int = 0
    demography: int = 0
    infobox: int = 0
    skipped: int = 0


def _norm_name(template: mwp.nodes.Template) -> str:
    return str(template.name).strip().replace("_", " ").lower()


def _format_number(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if not digits:
        return value
    groups: list[str] = []
    while digits:
        groups.insert(0, digits[-3:])
        digits = digits[:-3]
    return "&nbsp;".join(groups)


def demography_table(expanded: str) -> str | None:
    """Year/population wikitable from an expanded {{Demografia/...}}, or None
    if the timeline data is missing."""
    rows = [(m.group(1), m.group(2)) for m in _PLOT_BAR_RE.finditer(expanded)]
    if not rows:
        return None
    lines = ['{| class="wikitable"', "! Viti !! Banorë"]
    for year, count in rows:
        lines.append("|-")
        lines.append(f"| {year} || {_format_number(count)}")
    lines.append("|}")
    sources = [
        f"{{{{Cite web|url={url}|title={re.sub(r'<[^>]+>', '', title).strip()}|publisher=ISTAT|language=it}}}}"
        for url, title in _SOURCE_LINK_RE.findall(expanded)
    ]
    table = "\n".join(lines)
    if sources:
        table += "\n<ref>Të dhënat nga: " + "; ".join(sources) + "</ref>"
    return table


def infobox_settlement(template: mwp.nodes.Template, population: str | None) -> tuple[str, int]:
    """{{Infobox settlement}} wikitext for a {{Divisione amministrativa}} call,
    plus the number of extra rows dropped for lack of room."""
    fields = {str(p.name).strip(): str(p.value).strip() for p in template.params}

    def get(*keys: str) -> str:
        return next((fields[k] for k in keys if fields.get(k)), "")

    params: list[tuple[str, str]] = []
    for param, keys in _INFOBOX_FIELDS:
        if param == "leader_name" and get(*keys):
            params.append(("leader_title", "Sindaco"))
        if value := get(*keys):
            params.append((param, value))
    if country := get("Stato"):
        params.append(("subdivision_type", "Stato"))
        params.append(("subdivision_name", _COUNTRIES.get(country, country)))
    for level, (type_, key) in enumerate([("Regione", "Divisione amm grado 1"), ("Provincia", "Divisione amm grado 2")], 1):
        if value := get(key):
            link = value if "[[" in value else f"[[{value}]]"
            params.append((f"subdivision_type{level}", type_))
            params.append((f"subdivision_name{level}", link))
    if population:
        params.append(("population_total", population))

    extras = [(label, fields[label]) for label in _EXTRA_FIELDS if fields.get(label)]
    for index, (label, value) in enumerate(extras[:_MAX_EXTRA_ROWS]):
        infix = "" if index == 0 else str(index)
        params.append((f"blank{infix}_name_sec1", label))
        params.append((f"blank{infix}_info_sec1", value))

    body = "\n".join(f"| {name} = {value}" for name, value in params)
    return f"{{{{Infobox settlement\n{body}\n}}}}", max(0, len(extras) - _MAX_EXTRA_ROWS)


async def expand_italian_data_templates(
    wikitext: str, title: str, mw_client: MediaWikiClient
) -> DataTemplateResult:
    """Replaces the itwiki data templates in `wikitext` (see module docstring).
    A call that cannot be expanded is left untouched and counted in `skipped`."""
    code = mwp.parse(wikitext)
    result = DataTemplateResult(wikitext)
    population: str | None = None

    async def expand(template: mwp.nodes.Template) -> str | None:
        try:
            return (await mw_client.expand_templates(str(template), title)).strip()
        except Exception as exc:
            logger.warning("Could not expand %s in %r: %s", str(template.name).strip(), title, exc)
            return None

    # The infobox precedes {{Popolazione}} in an article, so read it first.
    first_population = next((tp for tp in code.filter_templates() if _norm_name(tp) == "popolazione"), None)
    if first_population is not None:
        expanded = await expand(first_population)
        if expanded and _POPULATION_RE.match(expanded):
            population = _format_number(expanded)

    for template in code.filter_templates(recursive=False):
        name = _norm_name(template)
        if name == "popolazione":
            if population is not None:
                code.replace(template, population)
                result.population += 1
            else:
                result.skipped += 1
        elif name.startswith("demografia/"):
            expanded = await expand(template)
            table = demography_table(expanded) if expanded else None
            if table:
                code.replace(template, table)
                result.demography += 1
            else:
                result.skipped += 1
        elif name == "divisione amministrativa":
            box, dropped = infobox_settlement(template, population)
            if dropped:
                logger.warning("Dropped %d extra infobox row(s) for %r", dropped, title)
            code.replace(template, box)
            result.infobox += 1

    result.wikitext = str(code)
    return result
