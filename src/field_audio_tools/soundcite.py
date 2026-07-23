from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .common import (
    ToolError,
    format_time,
    parse_time,
    prepare_output,
    probe_audio,
    require_program,
    run,
    sha256_file,
    write_json,
)


def _stream(probe: dict[str, Any]) -> dict[str, Any]:
    streams = probe.get("streams", [])
    if not streams:
        raise ToolError("No audio stream found")
    return streams[0]


def _duration(probe: dict[str, Any]) -> float:
    value = probe.get("format", {}).get("duration") or _stream(probe).get("duration")
    if value is None:
        raise ToolError("Audio duration is unavailable")
    return float(value)


def _public_probe(probe: dict[str, Any]) -> dict[str, Any]:
    stream = _stream(probe)
    file_format = probe.get("format", {})
    return {
        "codec": stream.get("codec_name"),
        "sample_format": stream.get("sample_fmt"),
        "sample_rate_hz": int(stream["sample_rate"])
        if stream.get("sample_rate")
        else None,
        "channels": stream.get("channels"),
        "bits_per_sample": stream.get("bits_per_sample"),
        "duration_seconds": float(file_format["duration"])
        if file_format.get("duration")
        else None,
        "container": file_format.get("format_name"),
        "embedded_tags": file_format.get("tags", {}),
    }


def _extract(
    ffmpeg: str,
    source: Path,
    destination: Path,
    start: float,
    duration: float,
) -> None:
    run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-ss",
            f"{start:.9f}",
            "-t",
            f"{duration:.9f}",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_f32le",
            str(destination),
        ]
    )


def _make_preview(ffmpeg: str, excerpts: list[Path], destination: Path) -> None:
    command = [ffmpeg, "-v", "error", "-y"]
    for excerpt in excerpts:
        command.extend(["-i", str(excerpt)])
    if len(excerpts) == 1:
        command.extend(
            [
                "-af",
                "alimiter=limit=0.95:attack=5:release=50:level=false",
                "-c:a",
                "libopus",
                "-b:a",
                "160k",
                str(destination),
            ]
        )
    else:
        labels = "".join(f"[{index}:a]" for index in range(len(excerpts)))
        filter_graph = (
            f"{labels}amerge=inputs={len(excerpts)},"
            "alimiter=limit=0.95:attack=5:release=50:level=false[a]"
        )
        command.extend(
            [
                "-filter_complex",
                filter_graph,
                "-map",
                "[a]",
                "-c:a",
                "libopus",
                "-b:a",
                "192k",
                str(destination),
            ]
        )
    run(command)


def _make_spectrogram(ffmpeg: str, preview: Path, destination: Path) -> None:
    run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            str(preview),
            "-lavfi",
            (
                "showspectrumpic=s=1600x900:legend=1:mode=separate:"
                "color=viridis:scale=log:fscale=log"
            ),
            "-frames:v",
            "1",
            str(destination),
        ]
    )


def _html_page(manifest: dict[str, Any]) -> str:
    citation = manifest["citation"]
    title = html.escape(citation.get("title") or "Field recording excerpt")
    rows = [
        ("Start", manifest["clip"]["start_timecode"]),
        ("Duration", f'{manifest["clip"]["duration_seconds"]:.3f} seconds'),
        ("Recorded", citation.get("recorded_at")),
        ("Location", citation.get("location")),
        ("Recordist", citation.get("recordist")),
        ("License", citation.get("license")),
    ]
    metadata_rows = "\n".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(value))}</td></tr>"
        for label, value in rows
        if value not in (None, "")
    )
    notes = citation.get("notes")
    notes_block = (
        f"<section><h2>Notes</h2><p>{html.escape(notes)}</p></section>" if notes else ""
    )
    source_items = "\n".join(
        f"<li><code>{html.escape(source['name'])}</code>"
        f" — {source['technical']['sample_rate_hz']} Hz,"
        f" {source['technical']['codec']}</li>"
        for source in manifest["sources"]
    )
    machine_manifest = html.escape(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ max-width: 72rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
    audio, img {{ display: block; width: 100%; margin: 1rem 0; }}
    img {{ height: auto; }}
    table {{ border-collapse: collapse; }}
    th, td {{ padding: .35rem .8rem .35rem 0; text-align: left; vertical-align: top; }}
    code {{ overflow-wrap: anywhere; }}
    .notice {{ border-left: .25rem solid #d58b00; padding-left: .8rem; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p class="notice">The browser preview is lossy and peak-limited.
    The WAV excerpt files preserve 32-bit floating-point samples.</p>
  </header>
  <audio controls preload="metadata" src="preview.ogg"></audio>
  <img src="spectrogram.png" alt="Spectrogram of the cited excerpt">
  <section>
    <h2>Citation metadata</h2>
    <table>{metadata_rows}</table>
  </section>
  {notes_block}
  <section>
    <h2>Sources</h2>
    <ul>{source_items}</ul>
    <p>Checksums and complete technical metadata are in
    <a href="manifest.json"><code>manifest.json</code></a>.</p>
  </section>
  <script type="application/json" id="soundcite-manifest">{machine_manifest}</script>
</body>
</html>
"""


def create_package(args: argparse.Namespace) -> Path:
    ffmpeg = require_program("ffmpeg")
    sources = [Path(value).expanduser() for value in args.audio]
    for source in sources:
        if not source.is_file():
            raise ToolError(f"Audio file not found: {source}")
    if len(sources) > 2:
        raise ToolError("soundcite currently supports one mono/stereo file or two tracks")

    start = parse_time(args.start)
    duration = parse_time(args.duration)
    if duration <= 0:
        raise ToolError("Duration must be greater than zero")
    probes = [probe_audio(source) for source in sources]
    for source, probe in zip(sources, probes, strict=True):
        if start >= _duration(probe):
            raise ToolError(f"Start time is beyond the end of {source}")
        if start + duration > _duration(probe) + 0.001:
            raise ToolError(f"Requested excerpt extends beyond the end of {source}")

    output = prepare_output(Path(args.output), args.force)
    excerpts: list[Path] = []
    for index, source in enumerate(sources, start=1):
        name = "excerpt.wav" if len(sources) == 1 else f"excerpt-track{index}.wav"
        excerpt = output / name
        _extract(ffmpeg, source, excerpt, start, duration)
        excerpts.append(excerpt)

    preview = output / "preview.ogg"
    spectrogram = output / "spectrogram.png"
    _make_preview(ffmpeg, excerpts, preview)
    _make_spectrogram(ffmpeg, preview, spectrogram)

    source_records = []
    for source, probe in zip(sources, probes, strict=True):
        stat = source.stat()
        record = {
            "name": source.name,
            "path_as_provided": str(source),
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
            "technical": _public_probe(probe),
        }
        if args.hash_source:
            record["sha256"] = sha256_file(source)
        source_records.append(record)

    excerpt_records = [
        {
            "name": excerpt.name,
            "sha256": sha256_file(excerpt),
            "size_bytes": excerpt.stat().st_size,
        }
        for excerpt in excerpts
    ]
    manifest = {
        "schema": "https://mantle-sound.org/schemas/soundcite/0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": {"tool": "soundcite", "version": __version__},
        "citation": {
            "title": args.title,
            "recordist": args.recordist,
            "recorded_at": args.recorded_at,
            "location": args.location,
            "license": args.license,
            "notes": args.notes,
        },
        "clip": {
            "start_seconds": start,
            "start_timecode": format_time(start),
            "duration_seconds": duration,
            "end_seconds": start + duration,
            "end_timecode": format_time(start + duration),
            "track_count": len(sources),
        },
        "sources": source_records,
        "artifacts": {
            "lossless_excerpts": excerpt_records,
            "preview": {
                "name": preview.name,
                "sha256": sha256_file(preview),
                "lossy": True,
                "processing": "Opus encoding with a 0.95 peak limiter",
            },
            "spectrogram": {
                "name": spectrogram.name,
                "sha256": sha256_file(spectrogram),
                "source": preview.name,
                "frequency_scale": "logarithmic",
            },
        },
    }
    write_json(output / "manifest.json", manifest)
    (output / "index.html").write_text(_html_page(manifest), encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundcite",
        description="Create a portable evidence package from a field recording excerpt.",
    )
    parser.add_argument(
        "audio",
        nargs="+",
        metavar="AUDIO",
        help="one audio file, or two synchronized mono track files",
    )
    parser.add_argument("--start", required=True, help="seconds, MM:SS, or HH:MM:SS")
    parser.add_argument(
        "--duration", default="20", help="excerpt length (default: 20 seconds)"
    )
    parser.add_argument("-o", "--output", required=True, help="output directory")
    parser.add_argument("--title", default="Field recording excerpt")
    parser.add_argument("--recordist")
    parser.add_argument("--recorded-at", help="ISO 8601 date or date-time")
    parser.add_argument("--location", help="public-safe location description")
    parser.add_argument("--license", default="All rights reserved")
    parser.add_argument("--notes")
    parser.add_argument(
        "--hash-source",
        action="store_true",
        help="hash entire source files (slow for long recordings)",
    )
    parser.add_argument(
        "--force", action="store_true", help="allow writing into a non-empty directory"
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = create_package(args)
    except (ToolError, ValueError) as exc:
        parser.exit(2, f"soundcite: error: {exc}\n")
    print(output / "index.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
