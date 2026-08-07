import pytest

from field_audio_tools.birdidpv import (
    Detection,
    Interval,
    _report_html,
    _report_multimedia_html,
    build_parser,
    merge_track_detections,
    read_intervals_csv,
    read_windows_csv,
    summarize_species,
)
from field_audio_tools.common import ToolError


INTERVALS_CSV = """start_seconds,end_seconds,start_timecode,end_timecode
0.0,10.0,00:00:00.000,00:00:10.000
20.0,50.0,00:00:20.000,00:00:50.000
"""

WINDOWS_CSV = """start_seconds,end_seconds,start_timecode,end_timecode,score,flagged
0.0,10.0,00:00:00.000,00:00:10.000,0.10,False
10.0,20.0,00:00:10.000,00:00:20.000,0.20,False
20.0,30.0,00:00:20.000,00:00:30.000,0.99,True
30.0,40.0,00:00:30.000,00:00:40.000,0.30,False
"""


def detection(start, name, confidence, track=1):
    return Detection(
        start_seconds=start,
        end_seconds=start + 3.0,
        scientific_name=f"Genus {name.lower()}",
        common_name=name,
        confidence=confidence,
        track=track,
    )


def test_read_intervals_csv(tmp_path):
    path = tmp_path / "candidate-clean-intervals.csv"
    path.write_text(INTERVALS_CSV, encoding="utf-8")
    assert read_intervals_csv(path) == [Interval(0.0, 10.0), Interval(20.0, 50.0)]


def test_read_intervals_csv_rejects_a_foreign_file(tmp_path):
    path = tmp_path / "other.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ToolError):
        read_intervals_csv(path)


def test_read_windows_csv_merges_below_threshold_runs(tmp_path):
    path = tmp_path / "windows.csv"
    path.write_text(WINDOWS_CSV, encoding="utf-8")
    assert read_windows_csv(path, 0.95) == [Interval(0.0, 20.0), Interval(30.0, 40.0)]


def test_read_windows_csv_errors_when_everything_is_flagged(tmp_path):
    path = tmp_path / "windows.csv"
    path.write_text(
        "start_seconds,end_seconds,score\n0.0,10.0,0.99\n", encoding="utf-8"
    )
    with pytest.raises(ToolError):
        read_windows_csv(path, 0.95)


def test_merge_track_detections_keeps_the_louder_track():
    merged = merge_track_detections(
        [
            detection(0.0, "Chough", 0.4, track=1),
            detection(0.0, "Chough", 0.9, track=2),
            detection(3.0, "Crow", 0.5, track=1),
        ]
    )
    assert [(item.common_name, item.confidence, item.track) for item in merged] == [
        ("Chough", 0.9, 2),
        ("Crow", 0.5, 1),
    ]


def test_merge_track_detections_keeps_distinct_species_in_one_frame():
    merged = merge_track_detections(
        [
            detection(0.0, "Chough", 0.4, track=1),
            detection(0.0, "Crow", 0.6, track=1),
        ]
    )
    assert len(merged) == 2


def test_summarize_species_ranks_by_confidence():
    species = summarize_species(
        [
            detection(0.0, "Crow", 0.5),
            detection(30.0, "Chough", 0.9),
            detection(60.0, "Chough", 0.7),
        ]
    )
    assert [row["common_name"] for row in species] == ["Chough", "Crow"]
    assert species[0]["detection_count"] == 2
    assert species[0]["max_confidence"] == 0.9
    assert species[0]["first_start_timecode"] == "00:00:30.000"
    assert species[0]["max_confidence_timecode"] == "00:00:30.000"


def test_defaults_match_the_documented_behaviour():
    args = build_parser().parse_args(["recording.wav", "--output", "report"])
    assert args.min_conf == 0.25
    assert args.week == -1
    assert args.report == "bare"
    assert args.intervals is None


def summary_fixture():
    return {
        "sources": [{"name": "Tr1.WAV"}],
        "range_source": "lowdom intervals CSV: candidate-clean-intervals.csv",
        "parameters": {
            "min_confidence": 0.25,
            "latitude": 29.898,
            "longitude": 102.03,
            "week_48": 3,
        },
        "duration_seconds": 120.0,
        "duration_timecode": "00:02:00.000",
        "analysed_range_count": 2,
        "analysed_timecode": "00:00:40.000",
        "analysed_percent": 33.3,
        "detection_count": 2,
        "species_count": 2,
    }


def test_report_html_is_bare_html4():
    detections = [detection(0.0, "Chough", 0.9), detection(30.0, "Crow", 0.4)]
    page = _report_html(
        summary_fixture(),
        detections,
        summarize_species(detections),
        [Interval(0.0, 10.0), Interval(20.0, 50.0)],
    )
    assert page.startswith('<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN">')
    assert "<style" not in page
    assert 'border="1" cellpadding="4" cellspacing="0"' in page
    assert "not a verified record" in page
    assert "species-row" not in page


def test_report_html_states_when_nothing_was_detected():
    page = _report_html(summary_fixture(), [], [], [Interval(0.0, 10.0)])
    assert "No detections reached the minimum confidence" in page


def test_report_html_reports_an_unfiltered_run():
    summary = summary_fixture()
    summary["parameters"]["latitude"] = None
    summary["parameters"]["longitude"] = None
    assert "full 6,522-species model" in _report_html(
        summary, [], [], [Interval(0.0, 10.0)]
    )


def test_report_multimedia_html_wires_up_playback_and_filtering():
    detections = [detection(0.0, "Chough", 0.9), detection(30.0, "Crow", 0.4)]
    page = _report_multimedia_html(
        summary_fixture(),
        detections,
        summarize_species(detections),
        [Interval(0.0, 10.0), Interval(20.0, 50.0)],
        preview_dir="recording-preview-48k",
        chunk_seconds=300.0,
        bare_report_href="report.html",
        lowdom_report_href="../260115-002-lowdom/report-js.html",
    )
    assert page.startswith('<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN">')
    assert 'id="enable"' in page
    assert "chunkSeconds = 300" in page
    assert 'data-species="Genus chough"' in page
    assert 'href="report.html"' in page
    assert 'href="../260115-002-lowdom/report-js.html"' in page


def test_report_multimedia_html_omits_absent_links():
    page = _report_multimedia_html(
        summary_fixture(),
        [],
        [],
        [Interval(0.0, 10.0)],
        preview_dir="preview",
        chunk_seconds=60.0,
        bare_report_href=None,
        lowdom_report_href=None,
    )
    assert "See also:" not in page
    assert "chunkSeconds = 60" in page


def test_report_escapes_species_names():
    detections = [
        Detection(0.0, 3.0, 'X <script>"', "Y & Z", 0.5, 1),
    ]
    page = _report_multimedia_html(
        summary_fixture(),
        detections,
        summarize_species(detections),
        [Interval(0.0, 10.0)],
        preview_dir="preview",
        chunk_seconds=60.0,
        bare_report_href=None,
        lowdom_report_href=None,
    )
    assert "<script>\"" not in page.split("<script>\nconst data")[0]
    assert "Y &amp; Z" in page


# --- reference photographs -------------------------------------------------


def photo(scientific_name="Genus chough", **overrides):
    from field_audio_tools.inaturalist import SpeciesPhoto

    fields = {
        "scientific_name": scientific_name,
        "taxon_id": 8350,
        "common_name": "Chough",
        "file_name": "genus-chough.jpg",
        "photo_id": 167222978,
        "license_code": "cc-by-nc",
        "attribution": "(c) A Photographer, some rights reserved (CC BY-NC)",
        "photographer": "A Photographer",
        "source_url": "https://example.invalid/medium.jpeg",
        "taxon_url": "https://www.inaturalist.org/taxa/8350",
        "wikipedia_url": None,
    }
    fields.update(overrides)
    return SpeciesPhoto(**fields)


def test_species_table_without_photos_has_no_photo_column():
    detections = [detection(0.0, "Chough", 0.9)]
    page = _report_html(
        summary_fixture(), detections, summarize_species(detections), [Interval(0.0, 10.0)]
    )
    assert "Reference photograph" not in page
    assert "<img" not in page


def test_species_table_renders_photo_and_credit():
    detections = [detection(0.0, "Chough", 0.9)]
    page = _report_html(
        summary_fixture(),
        detections,
        summarize_species(detections),
        [Interval(0.0, 10.0)],
        {"Genus chough": photo()},
        "species-photos",
    )
    assert "Reference photograph" in page
    assert 'src="species-photos/genus-chough.jpg"' in page
    assert "(c) A Photographer, some rights reserved (CC BY-NC)" in page
    assert 'href="https://www.inaturalist.org/taxa/8350"' in page
    assert "not evidence that it was present" in page


def test_species_without_a_photo_still_gets_a_row():
    detections = [detection(0.0, "Chough", 0.9), detection(30.0, "Crow", 0.4)]
    page = _report_html(
        summary_fixture(),
        detections,
        summarize_species(detections),
        [Interval(0.0, 10.0)],
        {"Genus chough": photo()},
        "species-photos",
    )
    assert page.count("<img") == 1
    assert "Crow" in page


def test_photo_credit_is_escaped():
    detections = [detection(0.0, "Chough", 0.9)]
    page = _report_html(
        summary_fixture(),
        detections,
        summarize_species(detections),
        [Interval(0.0, 10.0)],
        {"Genus chough": photo(attribution='<script>"x"')},
        "species-photos",
    )
    assert "<script>" not in page.split("<script>\nconst data")[0]
    assert "&lt;script&gt;" in page


def test_photo_defaults_are_opt_in():
    args = build_parser().parse_args(["recording.wav", "--output", "report"])
    assert args.photos is False
    assert args.photo_dir == "species-photos"
    assert args.photo_size == "medium"
    assert "cc-by-nc" in args.photo_licenses


def test_photo_licenses_parse_from_a_comma_list():
    args = build_parser().parse_args(
        ["recording.wav", "--output", "report", "--photo-licenses", "cc0, CC-BY "]
    )
    assert args.photo_licenses == ["cc0", "cc-by"]


def test_photo_links_do_not_toggle_the_row_filter():
    detections = [detection(0.0, "Chough", 0.9)]
    page = _report_multimedia_html(
        summary_fixture(),
        detections,
        summarize_species(detections),
        [Interval(0.0, 10.0)],
        preview_dir="preview",
        chunk_seconds=300.0,
        bare_report_href=None,
        lowdom_report_href=None,
        photos={"Genus chough": photo()},
        photo_dir="species-photos",
    )
    assert "stopPropagation" in page
    assert "<img" in page


def test_locked_playback_marker_is_orange():
    detections = [detection(0.0, "Chough", 0.9)]
    page = _report_multimedia_html(
        summary_fixture(),
        detections,
        summarize_species(detections),
        [Interval(0.0, 10.0)],
        preview_dir="preview",
        chunk_seconds=300.0,
        bare_report_href=None,
        lowdom_report_href=None,
    )
    assert 'LOCKED_COLOUR = "#d85b00"' in page
    assert "marker(scale(data.detections[playingIndex].s), LOCKED_COLOUR, 3)" in page
    # The hover marker stays black and still yields to the locked one.
    assert 'marker(scale(data.detections[hoverIndex].s), "#000", 2)' in page
    assert "hoverIndex !== playingIndex" in page


# --- detection spectrograms ------------------------------------------------


def test_spectrogram_panel_is_absent_without_images():
    detections = [detection(0.0, "Chough", 0.9)]
    page = _report_multimedia_html(
        summary_fixture(),
        detections,
        summarize_species(detections),
        [Interval(0.0, 10.0)],
        preview_dir="preview",
        chunk_seconds=300.0,
        bare_report_href=None,
        lowdom_report_href=None,
    )
    assert 'id="spectrogram"' not in page
    assert '"g":' not in page


def test_spectrogram_panel_and_per_frame_file_names():
    detections = [detection(0.0, "Chough", 0.9), detection(30.0, "Crow", 0.4)]
    page = _report_multimedia_html(
        summary_fixture(),
        detections,
        summarize_species(detections),
        [Interval(0.0, 10.0), Interval(20.0, 50.0)],
        preview_dir="preview",
        chunk_seconds=300.0,
        bare_report_href=None,
        lowdom_report_href=None,
        spectrograms={0: "frame-000000000.webp", 30000: "frame-000030000.webp"},
        spectrogram_dir="detection-spectrograms",
    )
    assert 'id="spectrogram"' in page
    assert 'src="detection-spectrograms/frame-000000000.webp"' in page
    assert '"g":"frame-000030000.webp"' in page
    assert 'spectrogramDir = "detection-spectrograms"' in page
    assert "showSpectrogram(index)" in page


def test_a_frame_without_an_image_is_handled_in_the_page():
    detections = [detection(0.0, "Chough", 0.9), detection(30.0, "Crow", 0.4)]
    page = _report_multimedia_html(
        summary_fixture(),
        detections,
        summarize_species(detections),
        [Interval(0.0, 10.0)],
        preview_dir="preview",
        chunk_seconds=300.0,
        bare_report_href=None,
        lowdom_report_href=None,
        spectrograms={0: "frame-000000000.webp"},
        spectrogram_dir="detection-spectrograms",
    )
    # Only the first detection has one; the page must cope with the other.
    assert page.count('"g":') == 1
    assert "No spectrogram was rendered for this frame." in page


def test_spectrogram_defaults_are_opt_in():
    args = build_parser().parse_args(["recording.wav", "--output", "report"])
    assert args.spectrograms is False
    assert args.spectrogram_dir == "detection-spectrograms"
    assert args.spectrogram_format == "webp"
    assert args.spectrogram_min_conf is None


# --- preview audio ---------------------------------------------------------


def test_multimedia_report_offers_playback_when_a_preview_exists():
    detections = [detection(0.0, "Chough", 0.9)]
    page = _report_multimedia_html(
        summary_fixture(),
        detections,
        summarize_species(detections),
        [Interval(0.0, 10.0)],
        preview_dir="recording-preview-48k",
        chunk_seconds=300.0,
        bare_report_href=None,
        lowdom_report_href=None,
        has_preview=True,
    )
    assert "<audio" in page
    assert 'id="enable"' in page
    assert 'src="recording-preview-48k/segment-000.ogg"' in page


def test_multimedia_report_drops_the_player_when_there_is_no_preview():
    detections = [detection(0.0, "Chough", 0.9)]
    page = _report_multimedia_html(
        summary_fixture(),
        detections,
        summarize_species(detections),
        [Interval(0.0, 10.0)],
        preview_dir="recording-preview-48k",
        chunk_seconds=300.0,
        bare_report_href=None,
        lowdom_report_href=None,
        has_preview=False,
    )
    assert "<audio" not in page
    assert 'id="enable"' not in page
    # Filtering and the timeline must survive.
    assert 'id="clear"' in page
    assert 'id="timeline"' in page
    assert "No preview audio in this package" in page
    # The script must not assume the missing elements exist.
    assert "if (!audio) return;" in page
    assert "if (enableButton) enableButton.addEventListener" in page
    assert "if (audio) audio.addEventListener" in page


def test_preview_defaults_generate_rather_than_assume():
    args = build_parser().parse_args(["recording.wav", "--output", "report"])
    assert args.no_preview is False
    assert args.preview_dir == "recording-preview-48k"
    assert args.preview_bitrate == "48k"
    assert args.preview_sample_rate == 48000


# --- BirdNET attribution ---------------------------------------------------
#
# The models are CC BY-NC-SA 4.0, so the credit is a licence condition. These
# tests exist so it cannot be dropped by accident.


def test_bare_report_carries_the_birdnet_credit():
    detections = [detection(0.0, "Chough", 0.9)]
    page = _report_html(
        summary_fixture(), detections, summarize_species(detections), [Interval(0.0, 10.0)]
    )
    assert "BirdNET" in page
    assert "Cornell Lab of Ornithology" in page
    assert "Chemnitz University of Technology" in page
    assert "CC BY-NC-SA 4.0" in page
    assert "Kahl" in page and "101236" in page


def test_multimedia_report_carries_the_birdnet_credit():
    detections = [detection(0.0, "Chough", 0.9)]
    page = _report_multimedia_html(
        summary_fixture(),
        detections,
        summarize_species(detections),
        [Interval(0.0, 10.0)],
        preview_dir="preview",
        chunk_seconds=300.0,
        bare_report_href=None,
        lowdom_report_href=None,
    )
    assert "Cornell Lab of Ornithology" in page
    assert "CC BY-NC-SA 4.0" in page
    assert "Kahl" in page and "101236" in page


def test_attribution_block_is_complete():
    from field_audio_tools.birdidpv import BIRDNET_ATTRIBUTION

    for key in (
        "developers",
        "homepage",
        "model_license",
        "model_license_url",
        "source_license",
        "citation",
        "note",
    ):
        assert BIRDNET_ATTRIBUTION.get(key), f"attribution is missing {key}"
    assert BIRDNET_ATTRIBUTION["model_license"] == "CC BY-NC-SA 4.0"
    assert "non-commercial" in BIRDNET_ATTRIBUTION["note"]


def test_analyzer_versions_reads_what_the_model_reports():
    from field_audio_tools.birdidpv import analyzer_versions

    class FakeAnalyzer:
        model_name = "BirdNET-Analyzer"
        version = "2.4"
        labels = ["a"] * 6522

    versions = analyzer_versions(FakeAnalyzer())
    assert versions["model_name"] == "BirdNET-Analyzer"
    assert versions["model_version"] == "2.4"
    assert versions["label_count"] == 6522
    assert versions["runner"] == "birdnetlib"


def test_bare_report_names_the_taxon_id_file_only_when_there_is_one():
    from field_audio_tools.taxonids import TaxonIdentifiers

    detections = [detection(0.0, "Chough", 0.9)]
    species = summarize_species(detections)
    without = _report_html(summary_fixture(), detections, species, [Interval(0.0, 10.0)])
    assert "taxon-ids.json" not in without
    assert "Taxon identifiers" not in without

    ids = [
        TaxonIdentifiers(
            scientific_name="Genus chough",
            common_name="Chough",
            inaturalist_id=8349,
            wikidata_id="Q643836",
            identifiers={"P2026": "18D2C42A9AD30CD8", "P3444": "rebcho1"},
        )
    ]
    with_ids = _report_html(
        summary_fixture(), detections, species, [Interval(0.0, 10.0)], None, "", ids
    )
    assert 'href="taxon-ids.json"' in with_ids
    assert "Taxon identifiers" in with_ids
    assert "avibaseid=18D2C42A9AD30CD8" in with_ids
    assert "https://www.wikidata.org/wiki/Q643836" in with_ids
