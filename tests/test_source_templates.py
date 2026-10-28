import pytest

from wiki_translation_harness.source_templates import (
    demography_table,
    expand_italian_data_templates,
)

_EXPANDED_DEMOGRAFIA = (
    "Abitanti censiti<ref>Dati tratti da:\n"
    "* <cite class=\"citation web\">[https://istat.example/a.pdf <span style=\"font-style:italic;\">Popolazione A</span>]"
    " (PDF)</cite></ref><br>\n<timeline>\nPlotData=\n"
    " bar:1861 from:0 till: 638\n bar:1901 from:0 till: 1 042\n"
    "PlotData=\n bar:1861 at: 638 fontsize:S text: 638 shift:(-10,5)\n</timeline>"
)

_INFOBOX = """{{Divisione amministrativa
|Nome = Andali
|Stato = ITA
|Divisione amm grado 1 = Calabria
|Divisione amm grado 2 = [[Catanzaro]]
|Amministratore locale = Mario Rossi
|Altitudine =
|Patrono = [[San Giorgio]]
}}
'''Andali''' ha {{Popolazione|ITA}} abitanti.

== Evoluzione demografica ==
{{Demografia/Andali}}
"""


class _FakeClient:
    def __init__(self, outputs: dict[str, str]):
        self.outputs = outputs
        self.calls: list[str] = []

    async def expand_templates(self, text: str, title: str) -> str:
        self.calls.append(text)
        for key, value in self.outputs.items():
            if text.startswith("{{" + key):
                if isinstance(value, Exception):
                    raise value
                return value
        return text


def test_demography_table_uses_first_plotdata_series_and_formats_thousands():
    table = demography_table(_EXPANDED_DEMOGRAFIA)
    assert table is not None
    assert "| 1861 || 638" in table
    assert "| 1901 || 1&nbsp;042" in table
    assert table.count("| 1861") == 1
    assert "{{Cite web|url=https://istat.example/a.pdf|title=Popolazione A|publisher=ISTAT|language=it}}" in table


def test_demography_table_none_without_timeline_data():
    assert demography_table("no data here") is None


@pytest.mark.asyncio
async def test_expands_population_demography_and_infobox():
    client = _FakeClient({"Popolazione": "1 574 ", "Demografia/Andali": _EXPANDED_DEMOGRAFIA})
    result = await expand_italian_data_templates(_INFOBOX, "Andali", client)

    assert (result.population, result.demography, result.infobox, result.skipped) == (1, 1, 1, 0)
    text = result.wikitext
    assert "{{Divisione amministrativa" not in text
    assert "{{Popolazione" not in text and "{{Demografia" not in text
    assert "ha 1&nbsp;574 abitanti" in text
    assert "| population_total = 1&nbsp;574" in text  # fetched although {{Popolazione}} comes later
    assert "| name = Andali" in text
    assert "| subdivision_name = [[Italia]]" in text
    assert "| subdivision_name1 = [[Calabria]]" in text
    assert "| subdivision_name2 = [[Catanzaro]]" in text
    assert "| leader_title = Sindaco" in text
    assert "| blank_name_sec1 = Patrono" in text
    assert "elevation_m" not in text  # empty source field
    assert '{| class="wikitable"' in text


@pytest.mark.asyncio
async def test_unexpandable_templates_are_left_in_place():
    client = _FakeClient({"Popolazione": RuntimeError("boom"), "Demografia/Andali": "no timeline"})
    result = await expand_italian_data_templates(
        "{{Popolazione|ITA}} e {{Demografia/Andali}}", "Andali", client
    )
    assert result.wikitext == "{{Popolazione|ITA}} e {{Demografia/Andali}}"
    assert (result.population, result.demography, result.skipped) == (0, 0, 2)
