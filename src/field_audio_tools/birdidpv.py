from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from . import __version__
from .common import ToolError, format_time, prepare_output, probe_audio, require_program
from .inaturalist import (
    DEFAULT_LICENSES,
    PHOTO_SIZES,
    USER_AGENT,
    SpeciesPhoto,
    collect_species_photos,
    write_credits,
)
from .preview import (
    DEFAULT_BITRATE,
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_SAMPLE_RATE,
    has_segments,
    write_preview_segments,
)
from .taxonids import (
    TaxonIdentifiers,
    collect_taxon_identifiers,
    write_taxon_ids,
)
from .spectrograms import (
    DEFAULT_SIZE,
    FORMATS,
    index_by_start,
    render_detections,
)

BIRDNET_SAMPLE_RATE = 48000

# Everything identifying here is BirdNET's work, not this project's. The models
# are CC BY-NC-SA 4.0, so attribution is a licence condition rather than a
# courtesy, and it is carried into summary.json and both reports. The models are
# never redistributed with this package: birdnetlib fetches them on first run.
BIRDNET_ATTRIBUTION = {
    "name": "BirdNET",
    "developers": [
        "K. Lisa Yang Center for Conservation Bioacoustics, Cornell Lab of "
        "Ornithology",
        "Chemnitz University of Technology",
    ],
    "homepage": "https://birdnet.cornell.edu/",
    "analyzer": "https://birdnet-team.github.io/BirdNET-Analyzer/",
    "model_license": "CC BY-NC-SA 4.0",
    "model_license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "source_license": "MIT",
    "citation": (
        "Kahl, S., Wood, C. M., Eibl, M., & Klinck, H. (2021). BirdNET: A deep "
        "learning solution for avian diversity monitoring. Ecological "
        "Informatics, 61, 101236."
    ),
    "note": (
        "This tool redistributes no model file. The models ship inside the "
        "birdnetlib package and are present once the birdnet extra is "
        "installed, so analysis needs no network. Installing therefore places "
        "a CC BY-NC-SA 4.0 copy on disk, and passing that environment on is "
        "redistribution. Those terms also apply to how results are used, "
        "including the non-commercial condition. The BirdNET authors state "
        "that educational and research purposes count as non-commercial."
    ),
}

BIRDNET_CREDIT_HTML = (
    'Identification by <a href="https://birdnet.cornell.edu/">BirdNET</a>, '
    "developed by the K. Lisa Yang Center for Conservation Bioacoustics at the "
    "Cornell Lab of Ornithology with Chemnitz University of Technology. The "
    'BirdNET models are licensed <a href="https://creativecommons.org/licenses/'
    'by-nc-sa/4.0/">CC BY-NC-SA 4.0</a>; the non-commercial condition applies '
    "to these results. Cite: Kahl, S., Wood, C. M., Eibl, M., &amp; Klinck, H. "
    "(2021). BirdNET: A deep learning solution for avian diversity monitoring. "
    "<em>Ecological Informatics</em>, 61, 101236."
)

# BirdNET scores fixed 3-second frames. Ranges shorter than one frame cannot be
# classified at all, so they are reported as skipped rather than silently
# dropped.
BIRDNET_FRAME_SECONDS = 3.0

_MISSING_EXTRA = (
    "birdidpv needs the optional BirdNET dependencies.\n"
    "Install them with:  pip install 'field-audio-tools[birdnet]'\n"
    "That pulls TensorFlow and about 65 MB of model files. Analysis then "
    "runs offline."
)


@dataclass(frozen=True)
class Interval:
    start_seconds: float
    end_seconds: float

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True)
class Detection:
    start_seconds: float
    end_seconds: float
    scientific_name: str
    common_name: str
    confidence: float
    track: int


def analyzer_versions(analyzer: Any) -> dict[str, Any]:
    """Record exactly which model and runner produced these detections."""
    import importlib.metadata as metadata

    try:
        runner_version = metadata.version("birdnetlib")
    except metadata.PackageNotFoundError:  # pragma: no cover
        runner_version = None
    return {
        "model_name": getattr(analyzer, "model_name", None),
        "model_version": getattr(analyzer, "version", None),
        "label_count": len(getattr(analyzer, "labels", []) or []) or None,
        "runner": "birdnetlib",
        "runner_version": runner_version,
    }


def load_analyzer(species_list_path: str | None = None) -> Any:
    """Import birdnetlib lazily so the core tools stay dependency-light."""
    try:
        from birdnetlib.analyzer import Analyzer
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ToolError(_MISSING_EXTRA) from exc
    try:
        if species_list_path:
            return Analyzer(custom_species_list_path=species_list_path)
        return Analyzer()
    except Exception as exc:  # pragma: no cover - model download or disk errors
        raise ToolError(f"BirdNET model could not be loaded: {exc}") from exc


def _recording_buffer_class() -> Any:
    try:
        from birdnetlib import RecordingBuffer
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ToolError(_MISSING_EXTRA) from exc
    return RecordingBuffer


def read_intervals_csv(path: Path) -> list[Interval]:
    """Read a lowdom candidate-clean-intervals.csv."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "start_seconds" not in reader.fieldnames:
            raise ToolError(f"{path} does not look like a lowdom intervals CSV")
        intervals = []
        for line, row in enumerate(reader, start=2):
            try:
                start = float(row["start_seconds"])
                end = float(row["end_seconds"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ToolError(f"{path}: invalid interval on line {line}") from exc
            if end <= start:
                raise ToolError(f"{path}: interval on line {line} is not positive")
            intervals.append(Interval(start, end))
    if not intervals:
        raise ToolError(f"{path} contains no intervals")
    return intervals


def read_windows_csv(path: Path, threshold: float) -> list[Interval]:
    """Derive contiguous below-threshold ranges from a lowdom windows.csv."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "score" not in reader.fieldnames:
            raise ToolError(f"{path} does not look like a lowdom windows CSV")
        intervals: list[Interval] = []
        start: float | None = None
        end = 0.0
        for line, row in enumerate(reader, start=2):
            try:
                window_start = float(row["start_seconds"])
                window_end = float(row["end_seconds"])
                score = float(row["score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ToolError(f"{path}: invalid window on line {line}") from exc
            flagged = score >= threshold
            if not flagged:
                if start is None:
                    start = window_start
                end = window_end
            elif start is not None:
                intervals.append(Interval(start, end))
                start = None
        if start is not None:
            intervals.append(Interval(start, end))
    if not intervals:
        raise ToolError(
            f"{path} has no windows below the threshold {threshold}; "
            "nothing would be analysed"
        )
    return intervals


def decode_interval(
    source: Path,
    interval: Interval,
    *,
    sample_rate: int = BIRDNET_SAMPLE_RATE,
) -> NDArray[np.float32]:
    ffmpeg = require_program("ffmpeg")
    command = [
        ffmpeg,
        "-v",
        "error",
        "-ss",
        f"{interval.start_seconds:.6f}",
        "-t",
        f"{interval.duration:.6f}",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    process = subprocess.run(command, capture_output=True)
    if process.returncode:
        detail = process.stderr.decode(errors="replace").strip()
        raise ToolError(f"ffmpeg could not decode {source}: {detail}")
    usable = len(process.stdout) - (len(process.stdout) % 4)
    return np.frombuffer(process.stdout[:usable], dtype="<f4").copy()


def analyze_interval(
    analyzer: Any,
    samples: NDArray[np.float32],
    interval: Interval,
    track: int,
    *,
    lat: float | None,
    lon: float | None,
    week: int,
    min_conf: float,
    sensitivity: float,
    overlap: float,
    filter_threshold: float,
) -> list[Detection]:
    recording_buffer = _recording_buffer_class()
    recording = recording_buffer(
        analyzer,
        samples,
        BIRDNET_SAMPLE_RATE,
        lat=lat,
        lon=lon,
        week_48=week,
        min_conf=min_conf,
        sensitivity=sensitivity,
        overlap=overlap,
        filter_threshold=filter_threshold,
    )
    try:
        recording.analyze()
    except Exception as exc:  # pragma: no cover - model runtime errors
        raise ToolError(
            f"BirdNET failed on the range starting at "
            f"{format_time(interval.start_seconds)}: {exc}"
        ) from exc
    detections = []
    for item in recording.detections:
        # BirdNET reports frame times relative to the buffer it was given.
        start = interval.start_seconds + float(item["start_time"])
        end = interval.start_seconds + float(item["end_time"])
        detections.append(
            Detection(
                start_seconds=start,
                end_seconds=min(end, interval.end_seconds),
                scientific_name=str(item["scientific_name"]),
                common_name=str(item["common_name"]),
                confidence=float(item["confidence"]),
                track=track,
            )
        )
    return detections


def merge_track_detections(detections: list[Detection]) -> list[Detection]:
    """Keep the highest-confidence track for each frame and species."""
    best: dict[tuple[float, str], Detection] = {}
    for detection in detections:
        key = (round(detection.start_seconds, 3), detection.scientific_name)
        current = best.get(key)
        if current is None or detection.confidence > current.confidence:
            best[key] = detection
    return sorted(
        best.values(), key=lambda item: (item.start_seconds, -item.confidence)
    )


def summarize_species(detections: list[Detection]) -> list[dict[str, object]]:
    grouped: dict[str, list[Detection]] = {}
    for detection in detections:
        grouped.setdefault(detection.scientific_name, []).append(detection)
    rows = []
    for scientific_name, items in grouped.items():
        best = max(items, key=lambda item: item.confidence)
        first = min(items, key=lambda item: item.start_seconds)
        rows.append(
            {
                "scientific_name": scientific_name,
                "common_name": items[0].common_name,
                "detection_count": len(items),
                "max_confidence": round(best.confidence, 4),
                "max_confidence_start_seconds": round(best.start_seconds, 3),
                "max_confidence_timecode": format_time(best.start_seconds),
                "first_start_seconds": round(first.start_seconds, 3),
                "first_start_timecode": format_time(first.start_seconds),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["max_confidence"]),  # type: ignore[arg-type]
            -int(row["detection_count"]),  # type: ignore[arg-type]
        )
    )
    return rows


def _write_detections_csv(path: Path, detections: list[Detection]) -> None:
    fields = [
        "start_seconds",
        "end_seconds",
        "start_timecode",
        "end_timecode",
        "scientific_name",
        "common_name",
        "confidence",
        "track",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for detection in detections:
            writer.writerow(
                {
                    "start_seconds": round(detection.start_seconds, 3),
                    "end_seconds": round(detection.end_seconds, 3),
                    "start_timecode": format_time(detection.start_seconds),
                    "end_timecode": format_time(detection.end_seconds),
                    "scientific_name": detection.scientific_name,
                    "common_name": detection.common_name,
                    "confidence": round(detection.confidence, 4),
                    "track": detection.track,
                }
            )


def _write_species_csv(path: Path, species: list[dict[str, object]]) -> None:
    fields = [
        "scientific_name",
        "common_name",
        "detection_count",
        "max_confidence",
        "max_confidence_timecode",
        "first_start_timecode",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(species)


def _metadata_table_rows(summary: dict[str, Any]) -> str:
    sources = ", ".join(
        html.escape(str(item["name"])) for item in summary["sources"]
    )
    parameters = summary["parameters"]
    position = (
        f'{parameters["latitude"]}, {parameters["longitude"]} · week '
        f'{parameters["week_48"]} of 48'
        if parameters["latitude"] is not None
        else "not applied (full 6,522-species model)"
    )
    analysed = (
        f'{summary["analysed_range_count"]} ranges · '
        f'{summary["analysed_timecode"]} of {summary["duration_timecode"]} '
        f'({summary["analysed_percent"]:.1f}%)'
    )
    return "\n".join(
        [
            f"<tr><th>Sources</th><td>{sources}</td></tr>",
            f'<tr><th>Ranges analysed</th><td>{html.escape(analysed)}</td></tr>',
            f'<tr><th>Range source</th><td>{html.escape(str(summary["range_source"]))}</td></tr>',
            f"<tr><th>Species filter</th><td>{html.escape(position)}</td></tr>",
            f'<tr><th>Minimum confidence</th><td>{parameters["min_confidence"]}</td></tr>',
            f'<tr><th>Detections</th><td>{summary["detection_count"]} '
            f'in {summary["species_count"]} species</td></tr>',
        ]
    )


def _photo_cell(
    scientific_name: str,
    photos: dict[str, SpeciesPhoto],
    photo_dir: str,
) -> str:
    """A thumbnail plus the credit the photographer's licence requires."""
    photo = photos.get(scientific_name)
    if photo is None:
        return "<td></td>"
    source = html.escape(f"{photo_dir}/{photo.file_name}", quote=True)
    alt = html.escape(
        f"{photo.common_name or scientific_name}, reference photograph", quote=True
    )
    credit = html.escape(photo.attribution or photo.license_code)
    link = html.escape(photo.taxon_url, quote=True)
    return (
        f'<td><a href="{link}"><img src="{source}" alt="{alt}" width="120"></a>'
        f"<br><small>{credit}</small></td>"
    )


def _species_table_html(
    species: list[dict[str, object]],
    interactive: bool,
    photos: dict[str, SpeciesPhoto] | None = None,
    photo_dir: str = "",
) -> str:
    if not species:
        return (
            "<p>No detections reached the minimum confidence in the analysed "
            "ranges.</p>"
        )
    photos = photos or {}
    photo_header = "<th>Reference photograph</th>" if photos else ""
    lines = [
        '<table border="1" cellpadding="4" cellspacing="0">',
        f"<tr>{photo_header}<th>Common name</th><th>Scientific name</th>"
        "<th>Detections</th>"
        "<th>Highest confidence</th><th>At</th><th>First heard</th></tr>",
    ]
    for row in species:
        scientific_name = str(row["scientific_name"])
        attributes = (
            f' class="species-row" data-species="'
            f'{html.escape(scientific_name, quote=True)}" tabindex="0"'
            if interactive
            else ""
        )
        photo_cell = (
            _photo_cell(scientific_name, photos, photo_dir) if photos else ""
        )
        lines.append(
            f"<tr{attributes}>"
            f"{photo_cell}"
            f'<td>{html.escape(str(row["common_name"]))}</td>'
            f"<td><em>{html.escape(scientific_name)}</em></td>"
            f'<td>{row["detection_count"]}</td>'
            f'<td>{float(row["max_confidence"]):.3f}</td>'  # type: ignore[arg-type]
            f'<td>{html.escape(str(row["max_confidence_timecode"])[:8])}</td>'
            f'<td>{html.escape(str(row["first_start_timecode"])[:8])}</td>'
            "</tr>"
        )
    lines.append("</table>")
    if photos:
        lines.append(
            "<p>Reference photographs come from iNaturalist and are stored "
            "verbatim in this package; each is licensed by its photographer as "
            'credited above. Full details are in <a href="'
            f'{html.escape(photo_dir, quote=True)}/credits.json">credits.json</a>. '
            "They show what the species looks like. They are not evidence that "
            "it was present.</p>"
        )
    return "\n".join(lines)


_CAVEAT = """<p><strong>Machine identification, not a verified record:</strong>
BirdNET assigns a confidence to each 3-second frame; it does not confirm that a
species was present. Confidence is not a probability, and a high score on one
frame is not a record. Geophony, distant human sound, and species outside the
filtered list can all produce confident errors. Listen to every detection before
citing it.</p>"""


def _embedded_data(
    detections: list[Detection],
    intervals: list[Interval],
    duration: float,
    spectrograms: dict[int, str] | None = None,
) -> str:
    spectrograms = spectrograms or {}
    items = []
    for item in detections:
        entry: dict[str, Any] = {
            "s": round(item.start_seconds, 3),
            "e": round(item.end_seconds, 3),
            "n": item.common_name,
            "t": item.scientific_name,
            "c": round(item.confidence, 4),
        }
        # The file name travels with the detection so the page never has to
        # reconstruct it from a naming convention.
        name = spectrograms.get(round(item.start_seconds * 1000))
        if name:
            entry["g"] = name
        items.append(entry)
    payload = {
        "duration": round(duration, 3),
        "ranges": [
            [round(item.start_seconds, 3), round(item.end_seconds, 3)]
            for item in intervals
        ],
        "detections": items,
    }
    return json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")


def _taxon_ids_file_clause(taxon_ids: list[TaxonIdentifiers] | None) -> str:
    """Only name the file when the run actually wrote one."""
    if not taxon_ids:
        return ""
    return (
        ' Cross-references to the other databases are in '
        '<a href="taxon-ids.json">taxon-ids.json</a>.'
    )


def _taxon_ids_html(taxon_ids: list[TaxonIdentifiers]) -> str:
    """One row of outbound identifiers per species."""
    if not taxon_ids:
        return ""
    rows = []
    for item in taxon_ids:
        links = " · ".join(
            f'<a href="{html.escape(url, quote=True)}">{html.escape(label)}</a>'
            for label, url in item.links()
        )
        name = item.common_name or item.scientific_name
        rows.append(
            f"<tr><td>{html.escape(name)}<br>"
            f"<em>{html.escape(item.scientific_name)}</em></td>"
            f"<td>{links}</td></tr>"
        )
    return (
        "<h2>Taxon identifiers</h2>\n"
        "<p>The same species in the other databases, matched on iNaturalist "
        "taxon ID rather than on name. Machine-readable in "
        '<a href="taxon-ids.json">taxon-ids.json</a>, which also keeps the '
        "identifiers not shown here.</p>\n"
        '<table border="1" cellpadding="4" cellspacing="0">\n'
        "<tr><th>Species</th><th>Identifiers</th></tr>\n"
        + "\n".join(rows)
        + "\n</table>\n"
    )


def _report_html(
    summary: dict[str, Any],
    detections: list[Detection],
    species: list[dict[str, object]],
    intervals: list[Interval],
    photos: dict[str, SpeciesPhoto] | None = None,
    photo_dir: str = "",
    taxon_ids: list[TaxonIdentifiers] | None = None,
) -> str:
    duration = float(summary["duration_seconds"])
    taxon_ids_file = _taxon_ids_file_clause(taxon_ids)
    return f"""<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN">
<html lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<title>birdidpv report</title>
</head>
<body>
<h1>birdidpv identification report</h1>
{_CAVEAT}
<h2>Identification metadata</h2>
<table border="1" cellpadding="4" cellspacing="0">
{_metadata_table_rows(summary)}
</table>
<p><canvas id="timeline" width="1400" height="240"
aria-label="Timeline of BirdNET detections inside the analysed ranges"></canvas></p>
<p>The pale band marks ranges that were analysed; unanalysed audio is left
blank. Each mark is one detection, taller and darker with higher confidence.</p>
<h2>Species</h2>
{_species_table_html(species, False, photos, photo_dir)}
{_taxon_ids_html(taxon_ids or [])}
<p>Every detection is in <a href="detections.csv">detections.csv</a>;
the species roll-up is in <a href="species.csv">species.csv</a>;
parameters are in <a href="summary.json">summary.json</a>.{taxon_ids_file}</p>
<hr>
<p><small>{BIRDNET_CREDIT_HTML}</small></p>
<script>
const data = {_embedded_data(detections, intervals, duration)};
const canvas = document.getElementById("timeline");
const ctx = canvas.getContext("2d");
const width = canvas.width, height = canvas.height;
const scale = seconds => seconds * width / data.duration;
ctx.clearRect(0, 0, width, height);
ctx.fillStyle = "#dfeee7";
data.ranges.forEach(range => {{
  const x = scale(range[0]);
  ctx.fillRect(x, 0, Math.max(1, scale(range[1]) - x), height);
}});
data.detections.forEach(det => {{
  const x = scale(det.s);
  const bar = Math.max(3, det.c * height);
  ctx.fillStyle = `rgba(20, 78, 58, ${{0.25 + 0.75 * det.c}})`;
  ctx.fillRect(x, height - bar, Math.max(2, scale(det.e) - x), bar);
}});
</script>
</body>
</html>
"""


def _report_multimedia_html(
    summary: dict[str, Any],
    detections: list[Detection],
    species: list[dict[str, object]],
    intervals: list[Interval],
    *,
    preview_dir: str,
    chunk_seconds: float,
    bare_report_href: str | None,
    lowdom_report_href: str | None,
    photos: dict[str, SpeciesPhoto] | None = None,
    photo_dir: str = "",
    spectrograms: dict[int, str] | None = None,
    spectrogram_dir: str = "",
    has_preview: bool = True,
    taxon_ids: list[TaxonIdentifiers] | None = None,
) -> str:
    duration = float(summary["duration_seconds"])
    preview_href = html.escape(preview_dir, quote=True)
    spectrograms = spectrograms or {}
    first_spectrogram = ""
    for item in detections:
        name = spectrograms.get(round(item.start_seconds * 1000))
        if name:
            first_spectrogram = name
            break
    links = []
    if bare_report_href:
        links.append(
            f'<a href="{html.escape(bare_report_href)}">bare identification report</a>'
        )
    if lowdom_report_href:
        links.append(
            f'<a href="{html.escape(lowdom_report_href)}">lowdom screening report</a>'
        )
    taxon_ids_file = _taxon_ids_file_clause(taxon_ids)
    related = f"<p>See also: {' · '.join(links)}.</p>\n" if links else ""
    if has_preview:
        audio_controls = (
            '<p><button id="enable" type="button" aria-pressed="false">'
            "Enable hover playback</button>\n"
            '<button id="clear" type="button">Show all species</button></p>\n'
            '<p id="status" role="status">Audio is off. Enable it once, then '
            "hover over the timeline.</p>\n"
            '<p><audio id="audio" controls preload="metadata"\n'
            f'src="{preview_href}/segment-000.ogg"></audio></p>\n'
        )
    else:
        # No preview audio exists, so no player is offered. The timeline, the
        # spectrograms, and the species filter all still work.
        audio_controls = (
            '<p><button id="clear" type="button">Show all species</button></p>\n'
            '<p id="status" role="status">No preview audio in this package; '
            "select a detection to see its frame.</p>\n"
        )
    spectrogram_block = (
        "<p id=\"spectrogram-panel\">\n"
        f'<img id="spectrogram" src="{html.escape(spectrogram_dir, quote=True)}/'
        f'{html.escape(first_spectrogram, quote=True)}"\n'
        'alt="Spectrogram of the selected detection frame">\n'
        '<br><span id="spectrogram-caption">Select a detection to see its '
        "spectrogram.</span></p>\n"
        if first_spectrogram
        else ""
    )
    return f"""<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN">
<html lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<title>birdidpv multimedia report</title>
<style>
tr.species-row {{ cursor: pointer; }}
tr.species-row.selected td {{ background: #dfeee7; font-weight: bold; }}
</style>
</head>
<body>
<h1>birdidpv multimedia identification report</h1>
{_CAVEAT}
<h2>Identification metadata</h2>
<table border="1" cellpadding="4" cellspacing="0">
{_metadata_table_rows(summary)}
</table>
{audio_controls}<p><canvas id="timeline" width="1400" height="240" tabindex="0"
aria-label="Interactive timeline of BirdNET detections"></canvas></p>
{spectrogram_block}<p>The pale band marks the ranges that were analysed. Each mark is one
3-second detection, taller and darker with higher confidence. Hover to select
the nearest detection; playback starts after a short pause and stops at the end
of that frame. Click to play immediately, which also copies the start timecode;
the locked detection keeps an orange marker so you can see where playback sits
after the pointer moves on. Select a species in the table to show only its
detections. The preview is lossy
Opus at a reduced bitrate; use the source files for verification.</p>
<h2>Species</h2>
{_species_table_html(species, True, photos, photo_dir)}
{_taxon_ids_html(taxon_ids or [])}
<p>Every detection is in <a href="detections.csv">detections.csv</a>;
the species roll-up is in <a href="species.csv">species.csv</a>;
parameters are in <a href="summary.json">summary.json</a>.{taxon_ids_file}</p>
<hr>
<p><small>{BIRDNET_CREDIT_HTML}</small></p>
{related}<script>
const data = {_embedded_data(detections, intervals, duration, spectrograms)};
const chunkSeconds = {chunk_seconds};
const previewDir = {json.dumps(preview_dir)};
const spectrogramDir = {json.dumps(spectrogram_dir)};
// Matches the orange lowdom uses for flagged windows.
const LOCKED_COLOUR = "#d85b00";
const canvas = document.getElementById("timeline");
const spectrogram = document.getElementById("spectrogram");
const spectrogramCaption = document.getElementById("spectrogram-caption");
const ctx = canvas.getContext("2d");
const audio = document.getElementById("audio");
const enableButton = document.getElementById("enable");
const clearButton = document.getElementById("clear");
const status = document.getElementById("status");
const rows = Array.from(document.querySelectorAll("tr.species-row"));
let enabled = false;
let filter = null;
let hoverIndex = -1;
let playingIndex = -1;
let playingEnd = 0;
let loadedChunk = 0;
let hoverTimer = 0;
let playRequest = 0;

function timecode(seconds) {{
  const whole = Math.floor(seconds);
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const secs = whole % 60;
  return [hours, minutes, secs].map(value => String(value).padStart(2, "0")).join(":");
}}

function visible() {{
  return filter === null
    ? data.detections.map((det, index) => index)
    : data.detections.reduce((acc, det, index) => {{
        if (det.t === filter) acc.push(index);
        return acc;
      }}, []);
}}

function scale(seconds) {{
  return seconds * canvas.width / data.duration;
}}

function describe(index, prefix) {{
  const det = data.detections[index];
  status.textContent = `${{prefix}}${{timecode(det.s)}} · ${{det.n}} (${{det.t}}) · confidence ${{det.c.toFixed(3)}}`;
  showSpectrogram(index);
}}

// Every frame's spectrogram was rendered ahead of time, so switching is just
// an src swap.
function showSpectrogram(index) {{
  if (!spectrogram) return;
  const det = data.detections[index];
  if (!det || !det.g) {{
    spectrogramCaption.textContent = "No spectrogram was rendered for this frame.";
    return;
  }}
  spectrogram.src = `${{spectrogramDir}}/${{det.g}}`;
  spectrogram.alt = `Spectrogram of the frame at ${{timecode(det.s)}}, detected as ${{det.n}}`;
  spectrogramCaption.textContent =
    `${{timecode(det.s)}}–${{timecode(det.e)}} · ${{det.n}} · confidence ${{det.c.toFixed(3)}}`;
}}

function draw() {{
  const width = canvas.width;
  const height = canvas.height;
  const shown = new Set(visible());
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#dfeee7";
  data.ranges.forEach(range => {{
    const x = scale(range[0]);
    ctx.fillRect(x, 0, Math.max(1, scale(range[1]) - x), height);
  }});
  data.detections.forEach((det, index) => {{
    if (!shown.has(index)) return;
    const x = scale(det.s);
    const bar = Math.max(3, det.c * height);
    const w = Math.max(2, scale(det.e) - x);
    ctx.fillStyle = index === playingIndex
      ? "#000000"
      : `rgba(20, 78, 58, ${{0.25 + 0.75 * det.c}})`;
    ctx.fillRect(x, height - bar, w, bar);
  }});
  // The locked detection keeps an orange marker of its own, so it stays
  // visible after the pointer moves on. Hover stays black.
  if (playingIndex >= 0 && shown.has(playingIndex)) {{
    marker(scale(data.detections[playingIndex].s), LOCKED_COLOUR, 3);
  }}
  if (hoverIndex >= 0 && hoverIndex !== playingIndex && shown.has(hoverIndex)) {{
    marker(scale(data.detections[hoverIndex].s), "#000", 2);
  }}
}}

function marker(x, colour, lineWidth) {{
  ctx.strokeStyle = colour;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  ctx.moveTo(x, 0);
  ctx.lineTo(x, canvas.height);
  ctx.stroke();
}}

function nearestAt(clientX) {{
  const candidates = visible();
  if (!candidates.length) return -1;
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
  const seconds = ratio * data.duration;
  let best = candidates[0];
  let bestDistance = Infinity;
  candidates.forEach(index => {{
    const det = data.detections[index];
    const distance = seconds < det.s
      ? det.s - seconds
      : (seconds > det.e ? seconds - det.e : 0);
    if (distance < bestDistance) {{
      bestDistance = distance;
      best = index;
    }}
  }});
  return best;
}}

async function copyTextToClipboard(text) {{
  if (navigator.clipboard && window.isSecureContext) {{
    try {{
      await navigator.clipboard.writeText(text);
      return true;
    }} catch (error) {{
      return false;
    }}
  }}
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.position = "fixed";
  area.style.left = "-9999px";
  document.body.appendChild(area);
  area.select();
  let copied = false;
  try {{
    copied = document.execCommand("copy");
  }} catch (error) {{
    copied = false;
  }}
  document.body.removeChild(area);
  return copied;
}}

async function playDetection(index, options = {{}}) {{
  const force = options.force === true;
  const request = ++playRequest;
  if (index < 0) return;
  // Built without preview audio: selection and spectrograms still work.
  if (!audio) return;
  if (!enabled && !force) {{
    describe(index, "Audio is off · ");
    return;
  }}
  const det = data.detections[index];
  try {{
    if (!enabled && force) {{
      audio.muted = true;
      await audio.play();
      audio.pause();
      audio.muted = false;
    }}
    const chunk = Math.floor(det.s / chunkSeconds);
    const localStart = det.s - chunk * chunkSeconds;
    if (chunk !== loadedChunk) {{
      audio.pause();
      loadedChunk = chunk;
      audio.src = `${{previewDir}}/segment-${{String(chunk).padStart(3, "0")}}.ogg`;
      audio.load();
      if (audio.readyState < 1) {{
        await new Promise((resolve, reject) => {{
          const ready = () => {{
            audio.removeEventListener("error", failed);
            resolve();
          }};
          const failed = () => {{
            audio.removeEventListener("loadedmetadata", ready);
            reject(new Error("Audio chunk could not be loaded"));
          }};
          audio.addEventListener("loadedmetadata", ready, {{ once: true }});
          audio.addEventListener("error", failed, {{ once: true }});
        }});
      }}
    }}
    if (request !== playRequest) return;
    playingIndex = index;
    playingEnd = localStart + (det.e - det.s);
    draw();
    audio.currentTime = localStart;
    await audio.play();
    const copied = await copyTextToClipboard(timecode(det.s));
    describe(index, copied ? "Playing · copied · " : "Playing · ");
  }} catch (error) {{
    playingIndex = -1;
    draw();
    if (enabled) setHoverPlayback(false);
    status.textContent = "Playback was blocked. Click Enable hover playback and try again.";
  }}
}}

function setHoverPlayback(on) {{
  enabled = on;
  if (!enableButton) return;
  enableButton.textContent = on ? "Disable hover playback" : "Enable hover playback";
  enableButton.setAttribute("aria-pressed", on ? "true" : "false");
  if (!on) clearTimeout(hoverTimer);
}}

async function toggleHoverPlayback() {{
  if (enabled) {{
    setHoverPlayback(false);
    status.textContent = "Hover playback off. Click a detection to play without hover.";
    return;
  }}
  try {{
    audio.muted = true;
    await audio.play();
    audio.pause();
    audio.muted = false;
    setHoverPlayback(true);
    status.textContent = "Ready. Hover over the timeline or click a detection.";
  }} catch (error) {{
    audio.muted = false;
    status.textContent = "The browser could not enable playback. Use the audio controls once, then try again.";
  }}
}}

function setFilter(species) {{
  filter = species;
  hoverIndex = -1;
  rows.forEach(row => {{
    row.classList.toggle("selected", species !== null && row.dataset.species === species);
  }});
  draw();
  if (species === null) {{
    status.textContent = `Showing all ${{data.detections.length}} detections.`;
  }} else {{
    const count = visible().length;
    status.textContent = `Showing ${{count}} detection${{count === 1 ? "" : "s"}} of ${{species}}.`;
  }}
}}

if (enableButton) enableButton.addEventListener("click", toggleHoverPlayback);
clearButton.addEventListener("click", () => setFilter(null));
rows.forEach(row => {{
  const select = () => setFilter(
    filter === row.dataset.species ? null : row.dataset.species
  );
  // A reference photograph links out to iNaturalist. Without this the same
  // click would also toggle the row filter on the way up.
  row.querySelectorAll("a").forEach(link => {{
    link.addEventListener("click", event => event.stopPropagation());
  }});
  row.addEventListener("click", select);
  row.addEventListener("keydown", event => {{
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    select();
  }});
}});
if (audio) audio.addEventListener("timeupdate", () => {{
  if (playingIndex >= 0 && audio.currentTime >= playingEnd) {{
    audio.pause();
    describe(playingIndex, "Finished · ");
    playingIndex = -1;
    draw();
  }}
}});
canvas.addEventListener("pointermove", event => {{
  const index = nearestAt(event.clientX);
  if (index === hoverIndex || index < 0) return;
  hoverIndex = index;
  draw();
  describe(index, audio ? (enabled ? "Selected · " : "Audio is off · ") : "Selected · ");
  clearTimeout(hoverTimer);
  hoverTimer = setTimeout(() => playDetection(index), 180);
}});
canvas.addEventListener("pointerleave", () => clearTimeout(hoverTimer));
canvas.addEventListener("pointerdown", event => {{
  clearTimeout(hoverTimer);
  hoverIndex = nearestAt(event.clientX);
  draw();
  playDetection(hoverIndex, {{ force: true }});
}});
canvas.addEventListener("keydown", event => {{
  if (!["ArrowLeft", "ArrowRight", "Enter", " "].includes(event.key)) return;
  event.preventDefault();
  const candidates = visible();
  if (!candidates.length) return;
  let position = candidates.indexOf(hoverIndex);
  if (position < 0) position = 0;
  else if (event.key === "ArrowLeft") position = Math.max(0, position - 1);
  else if (event.key === "ArrowRight") position = Math.min(candidates.length - 1, position + 1);
  hoverIndex = candidates[position];
  draw();
  describe(hoverIndex, "Selected · ");
  if (event.key === "Enter" || event.key === " ") {{
    playDetection(hoverIndex, {{ force: true }});
  }}
}});
draw();
</script>
</body>
</html>
"""


def _source_duration(source: Path) -> float:
    probe = probe_audio(source)
    for candidate in (
        probe.get("format", {}).get("duration"),
        probe.get("streams", [{}])[0].get("duration"),
    ):
        if candidate:
            return float(candidate)
    raise ToolError(f"ffprobe did not report a duration for {source}")


def identify(args: argparse.Namespace, *, progress: bool = True) -> Path:
    sources = [Path(value).expanduser() for value in args.audio]
    for source in sources:
        if not source.is_file():
            raise ToolError(f"Audio file not found: {source}")
    if len(sources) > 2:
        raise ToolError("birdidpv currently supports one file or two tracks")
    if args.intervals and args.windows:
        raise ToolError("Use either --intervals or --windows, not both")
    if not 0 < args.min_conf < 1:
        raise ToolError("--min-conf must be between zero and one")
    if not 0 <= args.overlap < BIRDNET_FRAME_SECONDS:
        raise ToolError("--overlap must be at least zero and below 3 seconds")
    if not 1 <= args.week <= 48 and args.week != -1:
        raise ToolError("--week must be between 1 and 48, or -1 to disable")
    if (args.lat is None) != (args.lon is None):
        raise ToolError("--lat and --lon must be given together")
    if args.preview_chunk_seconds <= 0:
        raise ToolError("--preview-chunk-seconds must be greater than zero")

    duration = _source_duration(sources[0])
    if args.intervals:
        intervals = read_intervals_csv(Path(args.intervals).expanduser())
        range_source = f"lowdom intervals CSV: {Path(args.intervals).name}"
    elif args.windows:
        intervals = read_windows_csv(
            Path(args.windows).expanduser(), args.window_threshold
        )
        range_source = (
            f"lowdom windows CSV below {args.window_threshold}: "
            f"{Path(args.windows).name}"
        )
    else:
        intervals = [Interval(0.0, duration)]
        range_source = "whole recording"

    usable = [item for item in intervals if item.duration >= BIRDNET_FRAME_SECONDS]
    skipped = len(intervals) - len(usable)
    if not usable:
        raise ToolError(
            "Every range is shorter than the 3-second BirdNET frame; "
            "nothing can be analysed"
        )

    output = prepare_output(Path(args.output), args.force)
    analyzer = load_analyzer(args.species_list)

    raw: list[Detection] = []
    total = len(usable) * len(sources)
    step = 0
    for track, source in enumerate(sources, start=1):
        for interval in usable:
            step += 1
            if progress:
                print(
                    f"[{step}/{total}] track {track} · "
                    f"{format_time(interval.start_seconds)}"
                    f"–{format_time(interval.end_seconds)}",
                    file=sys.stderr,
                    flush=True,
                )
            samples = decode_interval(source, interval)
            if samples.size < BIRDNET_FRAME_SECONDS * BIRDNET_SAMPLE_RATE:
                continue
            raw.extend(
                analyze_interval(
                    analyzer,
                    samples,
                    interval,
                    track,
                    lat=args.lat,
                    lon=args.lon,
                    week=args.week,
                    min_conf=args.min_conf,
                    sensitivity=args.sensitivity,
                    overlap=args.overlap,
                    filter_threshold=args.filter_threshold,
                )
            )

    detections = merge_track_detections(raw) if len(sources) > 1 else sorted(
        raw, key=lambda item: (item.start_seconds, -item.confidence)
    )
    species = summarize_species(detections)

    analysed_seconds = sum(item.duration for item in usable)
    source_summaries = []
    for source in sources:
        probe = probe_audio(source)
        stream = probe.get("streams", [{}])[0]
        source_summaries.append(
            {
                "name": source.name,
                "path_as_provided": str(source),
                "sample_rate_hz": int(stream["sample_rate"])
                if stream.get("sample_rate")
                else None,
                "channels": stream.get("channels"),
                "codec": stream.get("codec_name"),
            }
        )
    summary: dict[str, Any] = {
        "schema": "https://mantle-sound.org/schemas/birdidpv/0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": {"tool": "birdidpv", "version": __version__},
        "model": {
            **BIRDNET_ATTRIBUTION,
            **analyzer_versions(analyzer),
            "frame_seconds": BIRDNET_FRAME_SECONDS,
            "analysis_rate_hz": BIRDNET_SAMPLE_RATE,
        },
        "sources": source_summaries,
        "range_source": range_source,
        "parameters": {
            "min_confidence": args.min_conf,
            "sensitivity": args.sensitivity,
            "overlap_seconds": args.overlap,
            "latitude": args.lat,
            "longitude": args.lon,
            "week_48": args.week,
            "location_filter_threshold": args.filter_threshold,
            "custom_species_list": args.species_list,
            "track_merge": "maximum confidence per frame and species",
        },
        "interpretation": (
            "BirdNET confidence is a model score for a 3-second frame, not a "
            "probability and not a verified record. Listen before citing."
        ),
        "duration_seconds": round(duration, 3),
        "duration_timecode": format_time(duration),
        "analysed_range_count": len(usable),
        "skipped_short_range_count": skipped,
        "analysed_seconds": round(analysed_seconds, 3),
        "analysed_timecode": format_time(analysed_seconds),
        "analysed_percent": 100 * analysed_seconds / duration if duration else 0.0,
        "detection_count": len(detections),
        "species_count": len(species),
    }

    spectrogram_index: dict[int, str] = {}
    if args.spectrograms and detections:
        floor = (
            args.min_conf
            if args.spectrogram_min_conf is None
            else args.spectrogram_min_conf
        )
        frames = [
            (item.start_seconds, item.end_seconds)
            for item in detections
            if item.confidence >= floor
        ]
        if progress:
            print(
                f"Rendering spectrograms for {len(frames)} detected frames",
                file=sys.stderr,
                flush=True,
            )
        rendered, failed = render_detections(
            sources[0],
            frames,
            output / args.spectrogram_dir,
            size=args.spectrogram_size,
            image_format=args.spectrogram_format,
        )
        spectrogram_index = index_by_start(rendered)
        summary["spectrograms"] = {
            "directory": args.spectrogram_dir,
            "source": sources[0].name,
            "size": args.spectrogram_size,
            "format": args.spectrogram_format,
            "minimum_confidence": floor,
            "rendered": len(rendered),
            "failed": [
                {"start_timecode": format_time(start), "reason": reason}
                for start, reason in failed
            ],
        }
        if progress:
            for start, reason in failed:
                print(
                    f"  no spectrogram at {format_time(start)}: {reason}",
                    file=sys.stderr,
                )

    taxon_ids: list[TaxonIdentifiers] = []
    if args.taxon_ids and species:
        if progress:
            print(
                f"Resolving taxon identifiers for {len(species)} species",
                file=sys.stderr,
                flush=True,
            )
        known = {p.scientific_name: p.taxon_id for p in photos.values()}
        resolved, unresolved = collect_taxon_identifiers(
            [
                (
                    str(row["scientific_name"]),
                    str(row["common_name"]),
                    known.get(str(row["scientific_name"])),
                )
                for row in species
            ],
            timeout=args.taxon_id_timeout,
            delay=args.taxon_id_delay,
        )
        taxon_ids = resolved
        write_taxon_ids(output / "taxon-ids.json", resolved, unresolved)
        summary["taxon_identifiers"] = {
            "source": "Wikidata",
            "file": "taxon-ids.json",
            "resolved": len(resolved),
            "skipped": [
                {"scientific_name": n, "reason": r} for n, r in unresolved
            ],
        }
        if progress:
            for name, reason in unresolved:
                print(f"  no identifiers for {name}: {reason}", file=sys.stderr)

    photos: dict[str, SpeciesPhoto] = {}
    if args.photos and species:
        if progress:
            print(
                f"Fetching {len(species)} reference photographs from iNaturalist",
                file=sys.stderr,
                flush=True,
            )
        collected, skipped = collect_species_photos(
            [str(row["scientific_name"]) for row in species],
            output / args.photo_dir,
            size=args.photo_size,
            allowed_licenses=args.photo_licenses,
            timeout=args.photo_timeout,
            delay=args.photo_delay,
            user_agent=args.photo_user_agent,
        )
        photos = {photo.scientific_name: photo for photo in collected}
        write_credits(
            output / args.photo_dir / "credits.json",
            collected,
            skipped,
            photo_dir=args.photo_dir,
        )
        summary["photographs"] = {
            "source": "iNaturalist",
            "directory": args.photo_dir,
            "size": args.photo_size,
            "allowed_licenses": list(args.photo_licenses),
            "collected": len(collected),
            "skipped": [
                {"scientific_name": name, "reason": reason} for name, reason in skipped
            ],
        }
        if progress:
            for name, reason in skipped:
                print(f"  no photograph for {name}: {reason}", file=sys.stderr)

    _write_detections_csv(output / "detections.csv", detections)
    _write_species_csv(output / "species.csv", species)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.report in ("bare", "both"):
        (output / "report.html").write_text(
            _report_html(
                summary,
                detections,
                species,
                usable,
                photos,
                args.photo_dir,
                taxon_ids,
            ),
            encoding="utf-8",
        )
    if args.report in ("multimedia", "both"):
        # A hover-playback report needs audio it can actually load. Reuse a
        # preview that is already there — a lowdom package next door, say — and
        # otherwise write one, so a person holding only a recording still gets
        # a page that works.
        preview_dir = (output / args.preview_dir).resolve()
        has_preview = has_segments(preview_dir)
        if not has_preview and not args.no_preview:
            if progress:
                print(
                    f"Writing preview audio into {args.preview_dir}",
                    file=sys.stderr,
                    flush=True,
                )
            count = write_preview_segments(
                sources,
                preview_dir,
                duration,
                chunk_seconds=args.preview_chunk_seconds,
                sample_rate=args.preview_sample_rate,
                bitrate=args.preview_bitrate,
            )
            has_preview = count > 0
            summary["preview"] = {
                "directory": args.preview_dir,
                "generated": True,
                "segments": count,
                "chunk_seconds": args.preview_chunk_seconds,
                "sample_rate_hz": args.preview_sample_rate,
                "bitrate": args.preview_bitrate,
            }
        elif has_preview:
            summary["preview"] = {
                "directory": args.preview_dir,
                "generated": False,
                "reused": True,
            }
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "report-multimedia.html").write_text(
            _report_multimedia_html(
                summary,
                detections,
                species,
                usable,
                preview_dir=args.preview_dir,
                chunk_seconds=args.preview_chunk_seconds,
                bare_report_href="report.html" if args.report == "both" else None,
                lowdom_report_href=args.lowdom_report,
                photos=photos,
                photo_dir=args.photo_dir,
                spectrograms=spectrogram_index,
                spectrogram_dir=args.spectrogram_dir,
                has_preview=has_preview,
                taxon_ids=taxon_ids,
            ),
            encoding="utf-8",
        )
    return output


def _license_list(value: str) -> list[str]:
    codes = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not codes:
        raise argparse.ArgumentTypeError("give at least one licence code")
    return codes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="birdidpv",
        description=(
            "Run BirdNET over a field recording, optionally restricted to the "
            "ranges a lowdom screening left unflagged."
        ),
        epilog=(
            "birdidpv does not generate preview audio. The multimedia report "
            "links to preview segments produced by lowdom --report multimedia."
        ),
    )
    parser.add_argument(
        "audio",
        nargs="+",
        metavar="AUDIO",
        help="one audio file, or two synchronized mono track files",
    )
    parser.add_argument("-o", "--output", required=True, help="output directory")
    parser.add_argument(
        "--intervals",
        help="lowdom candidate-clean-intervals.csv restricting what is analysed",
    )
    parser.add_argument(
        "--windows",
        help="lowdom windows.csv; below-threshold windows are analysed",
    )
    parser.add_argument(
        "--window-threshold",
        type=float,
        default=0.95,
        help="threshold applied to --windows scores (default: 0.95)",
    )
    parser.add_argument(
        "--lat", type=float, help="recording latitude for the species filter"
    )
    parser.add_argument(
        "--lon", type=float, help="recording longitude for the species filter"
    )
    parser.add_argument(
        "--week",
        type=int,
        default=-1,
        help="week of 48 for the species filter, or -1 for no seasonal weighting",
    )
    parser.add_argument(
        "--filter-threshold",
        type=float,
        default=0.03,
        help="location filter cutoff; higher keeps fewer species (default: 0.03)",
    )
    parser.add_argument(
        "--species-list",
        help="path to a custom species list, used instead of the location filter",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.25,
        help="minimum BirdNET confidence to report (default: 0.25)",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=1.0,
        help="BirdNET detection sensitivity (default: 1.0)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="overlap in seconds between 3-second frames (default: 0)",
    )
    parser.add_argument(
        "--force", action="store_true", help="allow writing into a non-empty directory"
    )
    parser.add_argument(
        "--report",
        choices=("bare", "multimedia", "both", "none"),
        default="bare",
        help=(
            "HTML output: bare timeline (default), multimedia hover playback, "
            "both, or none"
        ),
    )
    parser.add_argument(
        "--preview-dir",
        default="recording-preview-48k",
        help=(
            "where the multimedia report looks for preview audio, relative to "
            "the output (default: recording-preview-48k). If segments are "
            "already there they are reused; otherwise they are written."
        ),
    )
    parser.add_argument(
        "--preview-chunk-seconds",
        type=float,
        default=DEFAULT_CHUNK_SECONDS,
        help=f"seconds per preview segment (default: {DEFAULT_CHUNK_SECONDS:g})",
    )
    parser.add_argument(
        "--preview-sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"sample rate for generated preview audio (default: {DEFAULT_SAMPLE_RATE})",
    )
    parser.add_argument(
        "--preview-bitrate",
        default=DEFAULT_BITRATE,
        help=f"Opus bitrate for generated preview audio (default: {DEFAULT_BITRATE})",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help=(
            "never write preview audio; if none is present the multimedia "
            "report is built without playback"
        ),
    )
    parser.add_argument(
        "--lowdom-report",
        help="relative link to the matching lowdom report (multimedia report)",
    )
    photos = parser.add_argument_group(
        "reference photographs",
        "Opt-in. This is the only part of the toolkit that uses the network.",
    )
    photos.add_argument(
        "--photos",
        action="store_true",
        help="download one reference photograph per species from iNaturalist",
    )
    photos.add_argument(
        "--photo-dir",
        default="species-photos",
        help="subdirectory for downloaded photographs (default: species-photos)",
    )
    photos.add_argument(
        "--photo-size",
        choices=PHOTO_SIZES,
        default="medium",
        help="iNaturalist image size to store (default: medium)",
    )
    photos.add_argument(
        "--photo-licenses",
        type=_license_list,
        default=list(DEFAULT_LICENSES),
        help=(
            "comma-separated licence codes to accept (default: "
            f"{','.join(DEFAULT_LICENSES)}). All-rights-reserved photographs "
            "are never downloaded."
        ),
    )
    photos.add_argument(
        "--photo-timeout",
        type=float,
        default=20.0,
        help="seconds to wait on each iNaturalist request (default: 20)",
    )
    photos.add_argument(
        "--photo-delay",
        type=float,
        default=1.0,
        help="seconds between iNaturalist requests (default: 1)",
    )
    photos.add_argument(
        "--photo-user-agent",
        default=USER_AGENT,
        help="User-Agent sent to iNaturalist; identify yourself if you fork this",
    )
    spectrograms = parser.add_argument_group(
        "detection spectrograms",
        "Opt-in. Rendered ahead of time from the source audio, not the preview.",
    )
    spectrograms.add_argument(
        "--spectrograms",
        action="store_true",
        help="render one spectrogram per detected frame for the report",
    )
    spectrograms.add_argument(
        "--spectrogram-dir",
        default="detection-spectrograms",
        help="subdirectory for rendered spectrograms (default: detection-spectrograms)",
    )
    spectrograms.add_argument(
        "--spectrogram-min-conf",
        type=float,
        help=(
            "only render frames at or above this confidence "
            "(default: every detection, i.e. --min-conf)"
        ),
    )
    spectrograms.add_argument(
        "--spectrogram-size",
        default=DEFAULT_SIZE,
        help=(
            f"size of the spectrum area as WIDTHxHEIGHT (default: {DEFAULT_SIZE}); "
            "the axis legend adds roughly 280x130 around it"
        ),
    )
    spectrograms.add_argument(
        "--spectrogram-format",
        choices=FORMATS,
        default="webp",
        help="image format (default: webp, roughly a tenth the size of png)",
    )
    identifiers = parser.add_argument_group(
        "taxon identifiers",
        "Opt-in. Cross-references each species to the taxonomic databases.",
    )
    identifiers.add_argument(
        "--taxon-ids",
        action="store_true",
        help="look up Avibase, eBird, GBIF, ITIS, IUCN and more via Wikidata",
    )
    identifiers.add_argument(
        "--taxon-id-timeout",
        type=float,
        default=20.0,
        help="seconds to wait on each Wikidata request (default: 20)",
    )
    identifiers.add_argument(
        "--taxon-id-delay",
        type=float,
        default=0.5,
        help="seconds between Wikidata requests (default: 0.5)",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = identify(args)
    except (ToolError, ValueError) as exc:
        parser.exit(2, f"birdidpv: error: {exc}\n")
    for name in ("report.html", "report-multimedia.html"):
        path = output / name
        if path.is_file():
            print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
