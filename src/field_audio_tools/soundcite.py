from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timedelta, timezone
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


def _sample_rate(probe: dict[str, Any]) -> int:
    value = _stream(probe).get("sample_rate")
    if value is None:
        raise ToolError("Audio sample rate is unavailable")
    return int(value)


def _tags(probe: dict[str, Any]) -> dict[str, Any]:
    return probe.get("format", {}).get("tags", {})


def _validate_sources(
    sources: list[Path], probes: list[dict[str, Any]]
) -> int:
    sample_rates = [_sample_rate(probe) for probe in probes]
    if len(set(sample_rates)) != 1:
        raise ToolError("Source tracks have different sample rates")

    if len(sources) == 2:
        channels = [_stream(probe).get("channels") for probe in probes]
        if channels != [1, 1]:
            raise ToolError("Two-file input requires two mono source tracks")

        tolerance = 1 / sample_rates[0]
        durations = [_duration(probe) for probe in probes]
        if abs(durations[0] - durations[1]) > tolerance:
            raise ToolError("Source tracks have different durations")

        for field in ("date", "creation_time", "time_reference"):
            values = [_tags(probe).get(field) for probe in probes]
            present = [value for value in values if value not in (None, "")]
            if len(present) == 2 and present[0] != present[1]:
                raise ToolError(
                    f"Source tracks have different embedded {field} values"
                )

    return sample_rates[0]


def _embedded_excerpt_time(
    probes: list[dict[str, Any]], start_seconds: float
) -> str | None:
    tags = _tags(probes[0])
    date = tags.get("date")
    creation_time = tags.get("creation_time")
    if not date or not creation_time:
        return None
    try:
        recording_start = datetime.fromisoformat(f"{date}T{creation_time}")
    except ValueError:
        return None
    return (recording_start + timedelta(seconds=start_seconds)).isoformat()


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
    start_sample: int,
    duration_samples: int,
    sample_rate: int,
) -> None:
    start = start_sample / sample_rate
    duration = duration_samples / sample_rate
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
            "-af",
            f"atrim=start_sample=0:end_sample={duration_samples},asetpts=PTS-STARTPTS",
            "-c:a",
            "pcm_f32le",
            str(destination),
        ]
    )


def _make_preview(ffmpeg: str, excerpts: list[Path], destination: Path) -> None:
    # -bitexact must sit with the output options, not before -i: as a global
    # option it never reaches the muxer. It stops the Ogg muxer generating a
    # random stream serial number, which is what kept two identical runs from
    # producing identical files. It also drops FFmpeg's version from the
    # vendor string, and that version is already in the manifest where it can
    # be read. The encoded audio is unaffected.
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
                "-bitexact",
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
                "-bitexact",
                str(destination),
            ]
        )
    run(command)


def _make_spectrogram(
    ffmpeg: str, excerpts: list[Path], destination: Path
) -> None:
    command = [ffmpeg, "-v", "error", "-y"]
    for excerpt in excerpts:
        command.extend(["-i", str(excerpt)])
    if len(excerpts) == 1:
        source = "[0:a]"
        merge = ""
    else:
        source = "[merged]"
        labels = "".join(f"[{index}:a]" for index in range(len(excerpts)))
        merge = f"{labels}amerge=inputs={len(excerpts)}[merged];"
    filter_graph = (
        f"{merge}{source}showspectrumpic=s=1600x900:legend=1:mode=separate:"
        "color=viridis:scale=log:fscale=log[out]"
    )
    command.extend(
        [
            "-filter_complex",
            filter_graph,
            "-map",
            "[out]",
            "-frames:v",
            "1",
            str(destination),
        ]
    )
    run(command)


def _write_checksums(output: Path, names: list[str]) -> None:
    lines = [f"{sha256_file(output / name)}  {name}" for name in names]
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _padding(args: argparse.Namespace) -> tuple[float, float]:
    """Resolve --pad, --pad-start and --pad-end into head and tail seconds.

    `--pad` sets both ends. The specific flags override it, so an asymmetric
    request only has to name the end that differs.
    """
    def seconds(flag: str, value: str | None, fallback: float) -> float:
        if value is None:
            return fallback
        try:
            # parse_time already refuses negative values; naming the flag is
            # the only thing missing from the message it raises.
            return parse_time(value)
        except ValueError as exc:
            raise ToolError(f"{flag}: {exc}") from exc

    base = seconds("--pad", args.pad, 0.0)
    return (
        seconds("--pad-start", args.pad_start, base),
        seconds("--pad-end", args.pad_end, base),
    )


def _html_page(
    manifest: dict[str, Any], *, site_package_href: str | None = None
) -> str:
    citation = manifest["citation"]
    title = html.escape(citation.get("title") or "Field recording excerpt")
    clip = manifest["clip"]
    cited = clip.get("cited")
    padding = clip.get("padding")
    rows = [
        ("Start", clip["start_timecode"]),
        ("Duration", f'{clip["duration_seconds"]:.3f} seconds'),
        (
            "Cited range",
            f'{cited["start_timecode"]} for {cited["duration_seconds"]:.3f} seconds'
            if cited
            else None,
        ),
        (
            "Padding",
            f'{padding["start_seconds"]:.3f} seconds before, '
            f'{padding["end_seconds"]:.3f} seconds after'
            + (" (limited by the ends of the file)"
               if padding["start_clamped"] or padding["end_clamped"] else "")
            if padding
            else None,
        ),
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
        f"<h2>Notes</h2>\n<p>{html.escape(notes)}</p>\n" if notes else ""
    )
    source_items = "\n".join(
        f"<li><code>{html.escape(source['name'])}</code>"
        f" - {source['technical']['sample_rate_hz']} Hz,"
        f" {source['technical']['codec']}</li>"
        for source in manifest["sources"]
    )
    machine_manifest = (
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    excerpt_links = "\n".join(
        f'<li><a href="{html.escape(item["name"])}">'
        f'<code>{html.escape(item["name"])}</code></a> '
        f'({item["size_bytes"]:,} bytes)</li>'
        for item in manifest["artifacts"]["lossless_excerpts"]
    )
    site_link_block = (
        f'<p><a href="{html.escape(site_package_href)}">'
        "Open the site package page</a></p>\n"
        if site_package_href
        else ""
    )
    return f"""<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN">
<html lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<title>{title}</title>
</head>
<body>
<h1>{title}</h1>
<p><audio controls preload="metadata" src="preview.ogg"></audio></p>
<p><img src="spectrogram.png" alt="Spectrogram of the cited excerpt"></p>
<h2>Citation metadata</h2>
<table border="1" cellpadding="4" cellspacing="0">
{metadata_rows}
</table>
{notes_block}
<h2>Sources</h2>
<ul>
{source_items}
</ul>
<h2>Package files</h2>
<ul>
{excerpt_links}
<li><a href="preview.ogg"><code>preview.ogg</code></a> (lossy listening copy)</li>
<li><a href="spectrogram.png"><code>spectrogram.png</code></a></li>
<li><a href="manifest.json"><code>manifest.json</code></a> (complete provenance and technical metadata)</li>
<li><a href="SHA256SUMS"><code>SHA256SUMS</code></a> (package integrity checks)</li>
</ul>
{site_link_block}
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
    pad_start, pad_end = _padding(args)
    probes = [probe_audio(source) for source in sources]
    sample_rate = _validate_sources(sources, probes)
    for source, probe in zip(sources, probes, strict=True):
        if start >= _duration(probe):
            raise ToolError(f"Start time is beyond the end of {source}")
        if start + duration > _duration(probe) + 0.001:
            raise ToolError(f"Requested excerpt extends beyond the end of {source}")

    cited_start_sample = round(start * sample_rate)
    cited_duration_samples = round(duration * sample_rate)

    # Padding runs out at the ends of the file. Trim it to what is available
    # rather than refusing the run, and record how much survived: an excerpt
    # that quietly carried less context than asked for would misdescribe
    # itself, and this tool exists to describe itself accurately.
    available = min(round(_duration(probe) * sample_rate) for probe in probes)
    head = min(round(pad_start * sample_rate), cited_start_sample)
    tail = min(
        round(pad_end * sample_rate),
        max(available - cited_start_sample - cited_duration_samples, 0),
    )

    start_sample = cited_start_sample - head
    duration_samples = cited_duration_samples + head + tail
    exact_start = start_sample / sample_rate
    exact_duration = duration_samples / sample_rate

    output = prepare_output(Path(args.output), args.force)
    excerpts: list[Path] = []
    for index, source in enumerate(sources, start=1):
        name = "excerpt.wav" if len(sources) == 1 else f"excerpt-track{index}.wav"
        excerpt = output / name
        _extract(
            ffmpeg,
            source,
            excerpt,
            start_sample,
            duration_samples,
            sample_rate,
        )
        excerpts.append(excerpt)

    preview = output / "preview.ogg"
    spectrogram = output / "spectrogram.png"
    _make_preview(ffmpeg, excerpts, preview)
    _make_spectrogram(ffmpeg, excerpts, spectrogram)

    source_records = []
    for source, probe in zip(sources, probes, strict=True):
        stat = source.stat()
        record = {
            "name": source.name,
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
            "technical": _public_probe(probe),
        }
        if args.include_source_path:
            record["path_as_provided"] = str(source)
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
        "created_by": {
            "tool": "soundcite",
            "version": __version__,
            "ffmpeg": run([ffmpeg, "-version"]).stdout.splitlines()[0],
        },
        "citation": {
            "title": args.title,
            "recordist": args.recordist,
            "recorded_at": args.recorded_at
            or _embedded_excerpt_time(probes, exact_start),
            "location": args.location,
            "license": args.license,
            "notes": args.notes,
        },
        "clip": {
            "start_seconds": exact_start,
            "start_timecode": format_time(exact_start),
            "start_sample": start_sample,
            "duration_seconds": exact_duration,
            "duration_samples": duration_samples,
            "end_seconds": exact_start + exact_duration,
            "end_timecode": format_time(exact_start + exact_duration),
            "end_sample": start_sample + duration_samples,
            "sample_rate_hz": sample_rate,
            "track_count": len(sources),
        },
        "sources": source_records,
        "artifacts": {
            "lossless_excerpts": excerpt_records,
            "preview": {
                "name": preview.name,
                "sha256": sha256_file(preview),
                "lossy": True,
                "codec": "Opus",
                "bitrate": "160 kbit/s" if len(excerpts) == 1 else "192 kbit/s",
                "channel_mapping": "source channels in input order",
                "processing": "Opus encoding with a 0.95 peak limiter",
            },
            "spectrogram": {
                "name": spectrogram.name,
                "sha256": sha256_file(spectrogram),
                "source": [excerpt.name for excerpt in excerpts],
                "frequency_scale": "logarithmic",
            },
            "checksums": {"name": "SHA256SUMS", "algorithm": "SHA-256"},
        },
    }
    # The fields above describe the audio that was written, because that is
    # what the checksums cover. When padding widened it, the range actually
    # being cited is recorded separately, along with how much padding survived
    # the ends of the file.
    if pad_start or pad_end:
        cited_start = cited_start_sample / sample_rate
        cited_duration = cited_duration_samples / sample_rate
        manifest["clip"]["cited"] = {
            "start_seconds": cited_start,
            "start_timecode": format_time(cited_start),
            "start_sample": cited_start_sample,
            "duration_seconds": cited_duration,
            "duration_samples": cited_duration_samples,
            "end_seconds": cited_start + cited_duration,
            "end_timecode": format_time(cited_start + cited_duration),
            "end_sample": cited_start_sample + cited_duration_samples,
        }
        manifest["clip"]["padding"] = {
            "start_seconds": head / sample_rate,
            "end_seconds": tail / sample_rate,
            "start_requested_seconds": pad_start,
            "end_requested_seconds": pad_end,
            "start_clamped": head < round(pad_start * sample_rate),
            "end_clamped": tail < round(pad_end * sample_rate),
        }
    write_json(output / "manifest.json", manifest)
    bare_html = _html_page(manifest)
    (output / "index-bare.html").write_text(bare_html, encoding="utf-8")
    (output / "index.html").write_text(bare_html, encoding="utf-8")
    _write_checksums(
        output,
        [
            *(excerpt.name for excerpt in excerpts),
            preview.name,
            spectrogram.name,
            "manifest.json",
            "index-bare.html",
            "index.html",
        ],
    )
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
    parser.add_argument(
        "--pad",
        help="extra seconds kept at both ends, outside the cited range",
    )
    parser.add_argument(
        "--pad-start", help="extra seconds before the cited range (overrides --pad)"
    )
    parser.add_argument(
        "--pad-end", help="extra seconds after the cited range (overrides --pad)"
    )
    parser.add_argument("-o", "--output", required=True, help="output directory")
    parser.add_argument("--title", default="Field recording excerpt")
    parser.add_argument("--recordist")
    parser.add_argument(
        "--recorded-at",
        help="ISO 8601 date-time at the excerpt start (derived from tags if omitted)",
    )
    parser.add_argument("--location", help="public-safe location description")
    parser.add_argument("--license", default="CC BY-NC-SA 4.0")
    parser.add_argument("--notes")
    parser.add_argument(
        "--hash-source",
        action="store_true",
        help="hash entire source files (slow for long recordings)",
    )
    parser.add_argument(
        "--include-source-path",
        action="store_true",
        help="include input paths in the manifest (may expose private local paths)",
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
