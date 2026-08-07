import json

import pytest

from field_audio_tools import taxonids
from field_audio_tools.taxonids import (
    DISPLAY_IDENTIFIERS,
    TaxonIdentifiers,
    collect_taxon_identifiers,
    write_taxon_ids,
)


def ids(**overrides):
    base = {
        "scientific_name": "Pyrrhocorax pyrrhocorax",
        "common_name": "Red-billed Chough",
        "inaturalist_id": 8349,
        "wikidata_id": "Q643836",
        "identifiers": {
            "P2026": "18D2C42A9AD30CD8",
            "P3444": "rebcho1",
            "P846": "2482552",
            "P3151": "8349",
        },
    }
    base.update(overrides)
    return TaxonIdentifiers(**base)


@pytest.fixture
def wikidata(monkeypatch):
    state = {"qid": "Q643836", "ids": {"P2026": "18D2C42A9AD30CD8"}, "fail": None}

    def fake_find(inaturalist_id, *, timeout, user_agent=None):
        if state["fail"]:
            raise state["fail"]
        return state["qid"]

    def fake_read(qid, *, timeout, user_agent=None):
        return dict(state["ids"])

    monkeypatch.setattr(taxonids, "find_wikidata_item", fake_find)
    monkeypatch.setattr(taxonids, "read_identifiers", fake_read)
    return state


def test_links_use_the_curated_set_and_always_add_wikidata():
    labels = [label for label, _ in ids().links()]
    assert labels[-1] == "Wikidata"
    assert "Avibase" in labels and "eBird" in labels and "GBIF" in labels
    # ITIS and IUCN are absent from this fixture and must not appear.
    assert "ITIS" not in labels and "IUCN" not in labels


def test_link_urls_are_built_from_the_templates():
    urls = dict(ids().links())
    assert urls["Avibase"].endswith("avibaseid=18D2C42A9AD30CD8")
    assert urls["eBird"] == "https://ebird.org/species/rebcho1"
    assert urls["GBIF"] == "https://www.gbif.org/species/2482552"
    assert urls["Wikidata"] == "https://www.wikidata.org/wiki/Q643836"


def test_every_display_template_has_a_placeholder():
    for prop, label, template in DISPLAY_IDENTIFIERS:
        assert "{id}" in template, f"{label} template cannot be filled"
        assert prop.startswith("P")


def test_collect_uses_a_known_inaturalist_id_without_a_name_search(wikidata, monkeypatch):
    def explode(*a, **k):
        raise AssertionError("must not search iNaturalist by name")

    monkeypatch.setattr(taxonids, "resolve_taxon", explode)
    resolved, skipped = collect_taxon_identifiers(
        [("Pyrrhocorax pyrrhocorax", "Red-billed Chough", 8349)], delay=0
    )
    assert skipped == []
    assert resolved[0].wikidata_id == "Q643836"
    assert resolved[0].inaturalist_id == 8349


def test_collect_resolves_a_missing_id_by_exact_name(wikidata, monkeypatch):
    monkeypatch.setattr(
        taxonids,
        "resolve_taxon",
        lambda name, **k: {"id": 8349, "preferred_common_name": "Red-billed Chough"},
    )
    resolved, _ = collect_taxon_identifiers([("Pyrrhocorax pyrrhocorax", None, None)], delay=0)
    assert resolved[0].inaturalist_id == 8349
    assert resolved[0].common_name == "Red-billed Chough"


def test_an_unmatched_name_is_skipped(wikidata, monkeypatch):
    monkeypatch.setattr(taxonids, "resolve_taxon", lambda name, **k: None)
    resolved, skipped = collect_taxon_identifiers([("Ghost bird", None, None)], delay=0)
    assert resolved == []
    assert "no exact name match" in skipped[0][1]


def test_a_species_absent_from_wikidata_is_skipped(wikidata):
    wikidata["qid"] = None
    resolved, skipped = collect_taxon_identifiers([("X y", None, 999)], delay=0)
    assert resolved == []
    assert "no Wikidata item" in skipped[0][1]


def test_a_network_failure_is_a_skip_not_an_error(wikidata):
    import urllib.error

    wikidata["fail"] = urllib.error.URLError("offline")
    resolved, skipped = collect_taxon_identifiers([("X y", None, 1)], delay=0)
    assert resolved == []
    assert "lookup failed" in skipped[0][1]


def test_written_json_records_how_matching_was_done(tmp_path):
    path = tmp_path / "taxon-ids.json"
    write_taxon_ids(path, [ids()], [("Ghost bird", "not found")])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["source"] == "Wikidata"
    assert "never on name" in payload["matched_on"]
    assert payload["species"][0]["wikidata_id"] == "Q643836"
    # The full identifier set is kept, not just what is displayed.
    assert payload["species"][0]["identifiers"]["P3444"] == "rebcho1"
    assert payload["skipped"][0]["scientific_name"] == "Ghost bird"


def test_iucn_uses_the_redirecting_form():
    """/species/<id>/0 is a 404; /details/<id>/0 redirects to the assessment."""
    template = dict((p, t) for p, _, t in DISPLAY_IDENTIFIERS)["P627"]
    assert "/details/" in template
    assert "/species/" not in template
