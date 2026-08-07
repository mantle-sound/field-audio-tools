import json
import urllib.error

import pytest

from field_audio_tools import inaturalist
from field_audio_tools.common import ToolError
from field_audio_tools.inaturalist import (
    DEFAULT_LICENSES,
    collect_species_photos,
    resolve_taxon,
    select_photo,
    write_credits,
)


def taxon(name, license_code="cc-by-nc", taxon_id=1):
    return {
        "id": taxon_id,
        "name": name,
        "preferred_common_name": name.split()[-1].title(),
        "wikipedia_url": None,
        "default_photo": {
            "id": 99,
            "license_code": license_code,
            "attribution": f"(c) Someone ({license_code})",
            "attribution_name": "Someone",
            "square_url": "https://example.invalid/square.jpeg",
            "medium_url": "https://example.invalid/medium.jpeg",
        },
    }


@pytest.fixture
def api(monkeypatch):
    """Stub the two network calls; nothing in these tests touches the wire."""
    state = {"results": [], "downloads": [], "fail_lookup": None, "fail_download": None}

    def fake_get_json(url, *, timeout, user_agent):
        if state["fail_lookup"]:
            raise state["fail_lookup"]
        return {"results": state["results"]}

    def fake_download(url, destination_dir, stem, *, timeout, user_agent):
        if state["fail_download"]:
            raise state["fail_download"]
        destination_dir.mkdir(parents=True, exist_ok=True)
        name = f"{stem}.jpg"
        (destination_dir / name).write_bytes(b"\xff\xd8stub")
        state["downloads"].append(url)
        return name

    monkeypatch.setattr(inaturalist, "_get_json", fake_get_json)
    monkeypatch.setattr(inaturalist, "download_photo", fake_download)
    return state


def test_resolve_taxon_requires_an_exact_name(api):
    # iNaturalist ranks a congener first for this query in reality.
    api["results"] = [taxon("Pyrrhocorax graculus"), taxon("Pyrrhocorax pyrrhocorax")]
    found = resolve_taxon("Pyrrhocorax pyrrhocorax", timeout=1)
    assert found["name"] == "Pyrrhocorax pyrrhocorax"


def test_resolve_taxon_returns_none_rather_than_a_near_miss(api):
    api["results"] = [taxon("Pyrrhocorax graculus")]
    assert resolve_taxon("Pyrrhocorax pyrrhocorax", timeout=1) is None


def test_resolve_taxon_ignores_case_and_padding(api):
    api["results"] = [taxon("Corvus macrorhynchos")]
    assert resolve_taxon("  corvus MACRORHYNCHOS ", timeout=1) is not None


def test_select_photo_rejects_all_rights_reserved():
    assert select_photo(taxon("X y", license_code=None), DEFAULT_LICENSES) is None


def test_select_photo_rejects_a_licence_outside_the_allow_list():
    assert select_photo(taxon("X y", license_code="cc-by-nd"), ["cc0", "cc-by"]) is None


def test_select_photo_accepts_an_allowed_licence():
    assert select_photo(taxon("X y", license_code="cc-by"), ["cc0", "cc-by"]) is not None


def test_collect_downloads_and_credits(tmp_path, api):
    api["results"] = [taxon("Corvus macrorhynchos", taxon_id=8026)]
    photos, skipped = collect_species_photos(
        ["Corvus macrorhynchos"], tmp_path / "species-photos", delay=0
    )
    assert skipped == []
    assert len(photos) == 1
    entry = photos[0]
    assert entry.file_name == "corvus-macrorhynchos.jpg"
    assert entry.taxon_url == "https://www.inaturalist.org/taxa/8026"
    assert (tmp_path / "species-photos" / entry.file_name).exists()


def test_collect_uses_the_requested_size(tmp_path, api):
    api["results"] = [taxon("Corvus macrorhynchos")]
    collect_species_photos(
        ["Corvus macrorhynchos"], tmp_path / "p", size="square", delay=0
    )
    assert api["downloads"] == ["https://example.invalid/square.jpeg"]


def test_collect_rejects_an_unknown_size(tmp_path):
    with pytest.raises(ToolError):
        collect_species_photos(["X y"], tmp_path, size="enormous", delay=0)


def test_collect_rejects_an_empty_licence_list(tmp_path):
    with pytest.raises(ToolError):
        collect_species_photos(["X y"], tmp_path, allowed_licenses=[], delay=0)


def test_a_network_failure_is_a_skip_not_an_error(tmp_path, api):
    api["fail_lookup"] = urllib.error.URLError("no route to host")
    photos, skipped = collect_species_photos(["X y"], tmp_path, delay=0)
    assert photos == []
    assert len(skipped) == 1
    assert "lookup failed" in skipped[0][1]


def test_a_failed_download_is_a_skip_not_an_error(tmp_path, api):
    api["results"] = [taxon("X y")]
    api["fail_download"] = urllib.error.URLError("connection reset")
    photos, skipped = collect_species_photos(["X y"], tmp_path, delay=0)
    assert photos == []
    assert "download failed" in skipped[0][1]


def test_no_match_is_reported_with_a_reason(tmp_path, api):
    api["results"] = [taxon("Other species")]
    _, skipped = collect_species_photos(["X y"], tmp_path, delay=0)
    assert skipped == [("X y", "no exact name match on iNaturalist")]


def test_reserved_photo_is_reported_with_a_reason(tmp_path, api):
    api["results"] = [taxon("X y", license_code=None)]
    _, skipped = collect_species_photos(["X y"], tmp_path, delay=0)
    assert "all rights reserved" in skipped[0][1]


def test_one_failure_does_not_stop_the_rest(tmp_path, api):
    calls = {"n": 0}
    original = inaturalist._get_json

    def flaky(url, *, timeout, user_agent):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("flaky")
        return {"results": [taxon("Corvus macrorhynchos")]}

    inaturalist._get_json = flaky
    try:
        photos, skipped = collect_species_photos(
            ["Bad name", "Corvus macrorhynchos"], tmp_path, delay=0
        )
    finally:
        inaturalist._get_json = original
    assert len(photos) == 1 and len(skipped) == 1


def test_credits_record_licence_and_provenance(tmp_path, api):
    api["results"] = [taxon("Corvus macrorhynchos", taxon_id=8026)]
    photos, skipped = collect_species_photos(
        ["Corvus macrorhynchos"], tmp_path, delay=0
    )
    skipped.append(("Ghost bird", "no exact name match on iNaturalist"))
    path = tmp_path / "credits.json"
    write_credits(path, photos, skipped, photo_dir="species-photos")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["source"] == "iNaturalist"
    assert payload["photos"][0]["license_code"] == "cc-by-nc"
    assert payload["photos"][0]["photographer"] == "Someone"
    assert payload["skipped"][0]["scientific_name"] == "Ghost bird"
