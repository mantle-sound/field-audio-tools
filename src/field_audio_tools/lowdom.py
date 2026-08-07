from __future__ import annotations

import argparse
import csv
import html
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from . import __version__
from .common import ToolError, format_time, prepare_output, probe_audio, require_program, run
from .preview import write_preview_segments


@dataclass(frozen=True)
class WindowMetrics:
    start_seconds: float
    end_seconds: float
    rms_dbfs: float
    peak: float
    low_frequency_ratio: float
    spectral_flatness: float
    score: float


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def measure_window(
    samples: NDArray[np.floating],
    sample_rate: int,
    *,
    low_min_hz: float = 10.0,
    low_max_hz: float = 120.0,
    total_max_hz: float = 1000.0,
) -> tuple[float, float, float, float, float]:
    """Return RMS dBFS, peak, low-band ratio, flatness, and heuristic score."""
    if samples.size < 2:
        raise ValueError("A window needs at least two samples")
    values = np.asarray(samples, dtype=np.float64)
    values = values - np.mean(values)
    rms = float(np.sqrt(np.mean(np.square(values))))
    rms_dbfs = 20 * math.log10(max(rms, 1e-12))
    peak = float(np.max(np.abs(values)))

    tapered = values * np.hanning(values.size)
    power = np.square(np.abs(np.fft.rfft(tapered)))
    frequencies = np.fft.rfftfreq(values.size, d=1 / sample_rate)
    total_mask = (frequencies >= low_min_hz) & (frequencies <= total_max_hz)
    low_mask = (frequencies >= low_min_hz) & (frequencies <= low_max_hz)
    total_power = float(np.sum(power[total_mask]))
    low_power = float(np.sum(power[low_mask]))
    low_ratio = low_power / max(total_power, 1e-30)

    # Flatness is measured inside the candidate low band. Measuring it over the
    # full comparison band would make genuinely low-frequency broadband noise
    # look artificially tonal merely because little energy exists above 120 Hz.
    selected = power[low_mask]
    if selected.size:
        epsilon = max(float(np.mean(selected)) * 1e-12, 1e-30)
        flatness = float(
            np.exp(np.mean(np.log(selected + epsilon)))
            / max(float(np.mean(selected + epsilon)), 1e-30)
        )
    else:
        flatness = 0.0

    # This is deliberately a transparent screening heuristic, not a trained
    # wind classifier. Broadband low-frequency energy drives the score;
    # level and flatness gates reduce silent-window and pure-tone false alarms.
    prominence = _sigmoid((low_ratio - 0.55) / 0.08)
    level_gate = _sigmoid((rms_dbfs + 55.0) / 6.0)
    noise_gate = _sigmoid((flatness - 0.015) / 0.015)
    score = prominence * (0.35 + 0.65 * level_gate) * (0.5 + 0.5 * noise_gate)
    return rms_dbfs, peak, low_ratio, flatness, float(score)


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def decode_windows(
    source: Path,
    *,
    sample_rate: int,
    window_seconds: float,
) -> Iterator[NDArray[np.float32]]:
    ffmpeg = require_program("ffmpeg")
    command = [
        ffmpeg,
        "-v",
        "error",
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
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    bytes_per_window = round(sample_rate * window_seconds) * 4
    minimum_bytes = bytes_per_window // 2
    try:
        while data := _read_exact(process.stdout, bytes_per_window):
            if len(data) < minimum_bytes:
                break
            usable = len(data) - (len(data) % 4)
            yield np.frombuffer(data[:usable], dtype="<f4").copy()
    finally:
        process.stdout.close()
    stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code:
        raise ToolError(f"ffmpeg could not decode {source}: {stderr.strip()}")


def analyze_track(
    source: Path,
    *,
    sample_rate: int,
    window_seconds: float,
    low_min_hz: float,
    low_max_hz: float,
    total_max_hz: float,
) -> list[WindowMetrics]:
    rows = []
    for index, samples in enumerate(
        decode_windows(
            source,
            sample_rate=sample_rate,
            window_seconds=window_seconds,
        )
    ):
        rms, peak, ratio, flatness, score = measure_window(
            samples,
            sample_rate,
            low_min_hz=low_min_hz,
            low_max_hz=low_max_hz,
            total_max_hz=total_max_hz,
        )
        start = index * window_seconds
        actual_duration = samples.size / sample_rate
        rows.append(
            WindowMetrics(
                start_seconds=start,
                end_seconds=start + actual_duration,
                rms_dbfs=rms,
                peak=peak,
                low_frequency_ratio=ratio,
                spectral_flatness=flatness,
                score=score,
            )
        )
    return rows


def _clean_intervals(
    rows: list[dict[str, object]], threshold: float
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    start: float | None = None
    end = 0.0
    for row in rows:
        flagged = float(row["score"]) >= threshold
        if not flagged and start is None:
            start = float(row["start_seconds"])
        if not flagged:
            end = float(row["end_seconds"])
        if flagged and start is not None:
            intervals.append((start, end))
            start = None
    if start is not None:
        intervals.append((start, end))
    return intervals


def _write_windows_csv(
    path: Path,
    merged: list[dict[str, object]],
    track_count: int,
) -> None:
    fields = [
        "start_seconds",
        "end_seconds",
        "start_timecode",
        "end_timecode",
        "score",
        "flagged",
    ]
    for index in range(1, track_count + 1):
        fields.extend(
            [
                f"track_{index}_score",
                f"track_{index}_rms_dbfs",
                f"track_{index}_low_frequency_ratio",
                f"track_{index}_spectral_flatness",
                f"track_{index}_peak",
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)


def _write_intervals_csv(path: Path, intervals: list[tuple[float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["start_seconds", "end_seconds", "start_timecode", "end_timecode"]
        )
        for start, end in intervals:
            writer.writerow([start, end, format_time(start), format_time(end)])


def _embedded_row_data(rows: list[dict[str, object]]) -> str:
    return json.dumps(
        [
            {
                "start": row["start_seconds"],
                "end": row["end_seconds"],
                "score": row["score"],
                "flagged": row["flagged"],
            }
            for row in rows
        ],
        separators=(",", ":"),
    ).replace("</", "<\\/")


def _metadata_table_rows(summary: dict[str, object]) -> str:
    sources = ", ".join(
        html.escape(str(item["name"])) for item in summary["sources"]  # type: ignore[index]
    )
    flagged = (
        f'{summary["flagged_windows"]} of {summary["window_count"]} '
        f'windows ({summary["flagged_percent"]:.1f}%)'
    )
    return "\n".join(
        [
            f"<tr><th>Sources</th><td>{sources}</td></tr>",
            f'<tr><th>Analysed</th><td>{html.escape(str(summary["duration_timecode"]))}</td></tr>',
            f'<tr><th>Window</th><td>{summary["parameters"]["window_seconds"]} seconds</td></tr>',
            f'<tr><th>Threshold</th><td>{summary["parameters"]["threshold"]}</td></tr>',
            f"<tr><th>Flagged</th><td>{flagged}</td></tr>",
        ]
    )


def _report_html(summary: dict[str, object], rows: list[dict[str, object]]) -> str:
    embedded_rows = _embedded_row_data(rows)
    metadata_rows = _metadata_table_rows(summary)
    return f"""<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN">
<html lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<title>lowdom report</title>
</head>
<body>
<h1>lowdom screening report</h1>
<p><strong>Not a wind detector:</strong> highlighted windows contain broadband
low-frequency energy consistent with wind buffeting. Water, vehicles, handling,
and other geophony can produce similar scores. Listen before excluding data.</p>
<h2>Screening metadata</h2>
<table border="1" cellpadding="4" cellspacing="0">
{metadata_rows}
</table>
<p><canvas id="timeline" width="1400" height="320"
aria-label="Timeline of low-frequency contamination scores"></canvas></p>
<p>Darker orange means a higher score. The horizontal line is the selected
threshold. Exact values are in <a href="windows.csv">windows.csv</a>;
contiguous unflagged ranges are in
<a href="candidate-clean-intervals.csv">candidate-clean-intervals.csv</a>.</p>
<script>
const rows = {embedded_rows};
const threshold = {summary["parameters"]["threshold"]};
const canvas = document.getElementById("timeline");
const ctx = canvas.getContext("2d");
const width = canvas.width, height = canvas.height;
ctx.clearRect(0, 0, width, height);
rows.forEach((row, index) => {{
  const x = index * width / rows.length;
  const next = (index + 1) * width / rows.length;
  const bar = row.score * height;
  ctx.fillStyle = row.flagged ? "#d85b00" : "#3f8f72";
  ctx.fillRect(x, height - bar, Math.max(1, next - x), bar);
}});
ctx.strokeStyle = "#e3a600";
ctx.lineWidth = 2;
ctx.beginPath();
ctx.moveTo(0, height * (1 - threshold));
ctx.lineTo(width, height * (1 - threshold));
ctx.stroke();
</script>
</body>
</html>
"""


def _report_multimedia_html(
    summary: dict[str, object],
    rows: list[dict[str, object]],
    *,
    preview_dir: str,
    chunk_seconds: float,
    bare_timeline_href: str | None = None,
) -> str:
    embedded_rows = _embedded_row_data(rows)
    metadata_rows = _metadata_table_rows(summary)
    preview_href = html.escape(preview_dir, quote=True)
    bare_link = (
        f'<p><a href="{html.escape(bare_timeline_href)}">Open the bare timeline report</a></p>\n'
        if bare_timeline_href
        else ""
    )
    threshold = summary["parameters"]["threshold"]
    return f"""<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN">
<html lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<title>lowdom multimedia report</title>
</head>
<body>
<h1>lowdom multimedia screening report</h1>
<p><strong>Listening aid, not a wind detector:</strong> orange windows score at
or above the selected threshold; green windows fall below it. Wind, water,
vehicles, handling, and other low-frequency sounds can produce similar scores.
Listen before excluding data.</p>
<h2>Screening metadata</h2>
<table border="1" cellpadding="4" cellspacing="0">
{metadata_rows}
</table>
<p><button id="enable" type="button" aria-pressed="false">Enable hover playback</button></p>
<p id="status" role="status">Audio is off. Enable it once, then hover over the timeline.</p>
<p><audio id="audio" controls preload="metadata"
src="{preview_href}/segment-000.ogg"></audio></p>
<p><canvas id="timeline" width="1400" height="320" tabindex="0"
aria-label="Interactive timeline of low-frequency screening scores"></canvas></p>
<p>Move across the timeline to select a window. Playback changes after a short
pause in pointer movement and stops at the end of that window. Click the
  timeline to play immediately. While a window plays, its start timecode
  (HH:MM:SS) is copied to the clipboard. The preview is lossy Opus at a reduced bitrate
with a 0.95 peak limiter; use the source files for analysis.</p>
{bare_link}<p>Exact values are in <a href="windows.csv">windows.csv</a>;
contiguous unflagged ranges are in
<a href="candidate-clean-intervals.csv">candidate-clean-intervals.csv</a>;
parameters are in <a href="summary.json">summary.json</a>.</p>
<script>
const rows = {embedded_rows};
const threshold = {threshold};
const chunkSeconds = {chunk_seconds};
const previewDir = {json.dumps(preview_dir)};
const canvas = document.getElementById("timeline");
const ctx = canvas.getContext("2d");
const audio = document.getElementById("audio");
const enableButton = document.getElementById("enable");
const status = document.getElementById("status");
let enabled = false;
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

function describe(index, prefix) {{
  const row = rows[index];
  const state = row.flagged ? "orange · at or above threshold" : "green · below threshold";
  status.textContent = `${{prefix}}${{timecode(row.start)}}–${{timecode(row.end)}} · score ${{row.score.toFixed(3)}} · ${{state}}`;
}}

function draw() {{
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  rows.forEach((row, index) => {{
    const x = index * width / rows.length;
    const next = (index + 1) * width / rows.length;
    const bar = row.score * height;
    ctx.fillStyle = row.flagged ? "#d85b00" : "#3f8f72";
    ctx.fillRect(x, height - bar, Math.max(1, next - x), bar);
  }});
  ctx.strokeStyle = "#e3a600";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, height * (1 - threshold));
  ctx.lineTo(width, height * (1 - threshold));
  ctx.stroke();
  if (playingIndex >= 0) {{
    const px = playingIndex * width / rows.length;
    const pw = Math.max(1, (playingIndex + 1) * width / rows.length - px);
    ctx.fillStyle = "rgba(0, 0, 0, 0.14)";
    ctx.fillRect(px, 0, pw, height);
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 3;
    ctx.strokeRect(px + 1.5, 1.5, Math.max(0, pw - 3), height - 3);
  }}
  if (hoverIndex >= 0 && hoverIndex !== playingIndex) {{
    const x = (hoverIndex + 0.5) * width / rows.length;
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }}
}}

function indexAt(clientX) {{
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(0.999999, Math.max(0, (clientX - rect.left) / rect.width));
  return Math.floor(ratio * rows.length);
}}

function windowStartTimecode(index) {{
  return timecode(rows[index].start);
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

async function copyTimestamp(index) {{
  return copyTextToClipboard(windowStartTimecode(index));
}}

async function playWindow(index, options = {{}}) {{
  const force = options.force === true;
  const request = ++playRequest;
  if (!enabled && !force) {{
    describe(index, "Audio is off · ");
    return;
  }}
  const row = rows[index];
  try {{
    if (!enabled && force) {{
      audio.muted = true;
      await audio.play();
      audio.pause();
      audio.muted = false;
    }}
    const chunk = Math.floor(row.start / chunkSeconds);
    const localStart = row.start - chunk * chunkSeconds;
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
    playingEnd = localStart + (row.end - row.start);
    draw();
    audio.currentTime = localStart;
    await audio.play();
    const copied = await copyTimestamp(index);
    const stamp = windowStartTimecode(index);
    const prefix = copied ? `Playing · ${{stamp}} copied · ` : `Playing · ${{stamp}} · `;
    describe(index, prefix);
  }} catch (error) {{
    playingIndex = -1;
    draw();
    if (enabled) {{
      setHoverPlayback(false);
    }}
    status.textContent = "Playback was blocked. Click Enable hover playback and try again.";
  }}
}}

function setHoverPlayback(on) {{
  enabled = on;
  enableButton.disabled = false;
  enableButton.textContent = on ? "Disable hover playback" : "Enable hover playback";
  enableButton.setAttribute("aria-pressed", on ? "true" : "false");
  if (!on) {{
    clearTimeout(hoverTimer);
  }}
}}

async function toggleHoverPlayback() {{
  if (enabled) {{
    setHoverPlayback(false);
    status.textContent = "Hover playback off. Click a window to play without hover.";
    return;
  }}
  try {{
    audio.muted = true;
    await audio.play();
    audio.pause();
    audio.muted = false;
    setHoverPlayback(true);
    status.textContent = "Ready. Hover over the timeline or click a window.";
  }} catch (error) {{
    audio.muted = false;
    status.textContent = "The browser could not enable playback. Use the audio controls once, then try again.";
  }}
}}

enableButton.addEventListener("click", toggleHoverPlayback);
audio.addEventListener("timeupdate", () => {{
  if (playingIndex >= 0 && audio.currentTime >= playingEnd) {{
    audio.pause();
    describe(playingIndex, "Finished · ");
    playingIndex = -1;
    draw();
  }}
}});
canvas.addEventListener("pointermove", event => {{
  const index = indexAt(event.clientX);
  if (index === hoverIndex) return;
  hoverIndex = index;
  draw();
  describe(index, enabled ? "Selected · " : "Audio is off · ");
  clearTimeout(hoverTimer);
  hoverTimer = setTimeout(() => playWindow(index), 180);
}});
canvas.addEventListener("pointerleave", () => clearTimeout(hoverTimer));
canvas.addEventListener("pointerdown", event => {{
  clearTimeout(hoverTimer);
  hoverIndex = indexAt(event.clientX);
  draw();
  playWindow(hoverIndex, {{ force: true }});
}});
canvas.addEventListener("keydown", event => {{
  if (!["ArrowLeft", "ArrowRight", "Enter", " "].includes(event.key)) return;
  event.preventDefault();
  if (hoverIndex < 0) hoverIndex = 0;
  if (event.key === "ArrowLeft") hoverIndex = Math.max(0, hoverIndex - 1);
  if (event.key === "ArrowRight") hoverIndex = Math.min(rows.length - 1, hoverIndex + 1);
  draw();
  describe(hoverIndex, "Selected · ");
  if (event.key === "Enter" || event.key === " ") playWindow(hoverIndex, {{ force: true }});
}});
draw();
</script>
</body>
</html>
"""


def create_report(args: argparse.Namespace) -> Path:
    sources = [Path(value).expanduser() for value in args.audio]
    for source in sources:
        if not source.is_file():
            raise ToolError(f"Audio file not found: {source}")
    if len(sources) > 2:
        raise ToolError("lowdom currently supports one file or two tracks")
    if args.window <= 0:
        raise ToolError("--window must be greater than zero")
    if not 0 < args.threshold < 1:
        raise ToolError("--threshold must be between zero and one")
    if args.analysis_rate < 2 * args.total_max:
        raise ToolError("--analysis-rate must be at least twice --total-max")
    if not 0 <= args.low_min < args.low_max < args.total_max:
        raise ToolError("frequency bands must satisfy low-min < low-max < total-max")
    if args.preview_chunk_seconds <= 0:
        raise ToolError("--preview-chunk-seconds must be greater than zero")
    if args.report in ("multimedia", "both") and args.preview_sample_rate < 8000:
        raise ToolError("--preview-sample-rate must be at least 8000 Hz")

    output = prepare_output(Path(args.output), args.force)
    tracks = [
        analyze_track(
            source,
            sample_rate=args.analysis_rate,
            window_seconds=args.window,
            low_min_hz=args.low_min,
            low_max_hz=args.low_max,
            total_max_hz=args.total_max,
        )
        for source in sources
    ]
    if not tracks or not tracks[0]:
        raise ToolError("No complete analysis windows were decoded")
    minimum_count = min(len(track) for track in tracks)
    if max(len(track) for track in tracks) - minimum_count > 1:
        raise ToolError("Track durations differ by more than one analysis window")

    merged: list[dict[str, object]] = []
    for index in range(minimum_count):
        track_rows = [track[index] for track in tracks]
        score = max(row.score for row in track_rows)
        start = max(row.start_seconds for row in track_rows)
        end = min(row.end_seconds for row in track_rows)
        item: dict[str, object] = {
            "start_seconds": round(start, 6),
            "end_seconds": round(end, 6),
            "start_timecode": format_time(start),
            "end_timecode": format_time(end),
            "score": round(score, 6),
            "flagged": score >= args.threshold,
        }
        for track_index, row in enumerate(track_rows, start=1):
            item.update(
                {
                    f"track_{track_index}_score": round(row.score, 6),
                    f"track_{track_index}_rms_dbfs": round(row.rms_dbfs, 3),
                    f"track_{track_index}_low_frequency_ratio": round(
                        row.low_frequency_ratio, 6
                    ),
                    f"track_{track_index}_spectral_flatness": round(
                        row.spectral_flatness, 6
                    ),
                    f"track_{track_index}_peak": round(row.peak, 6),
                }
            )
        merged.append(item)

    intervals = _clean_intervals(merged, args.threshold)
    flagged_count = sum(bool(row["flagged"]) for row in merged)
    source_summaries = []
    for source in sources:
        probe = probe_audio(source)
        stream = probe.get("streams", [{}])[0]
        tags = probe.get("format", {}).get("tags", {})
        source_summaries.append(
            {
                "name": source.name,
                "path_as_provided": str(source),
                "sample_rate_hz": int(stream["sample_rate"])
                if stream.get("sample_rate")
                else None,
                "channels": stream.get("channels"),
                "codec": stream.get("codec_name"),
                "embedded_tags": tags,
            }
        )
    duration = float(merged[-1]["end_seconds"])
    summary: dict[str, object] = {
        "schema": "https://mantle-sound.org/schemas/lowdom/0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": {"tool": "lowdom", "version": __version__},
        "sources": source_summaries,
        "parameters": {
            "window_seconds": args.window,
            "analysis_rate_hz": args.analysis_rate,
            "low_band_hz": [args.low_min, args.low_max],
            "comparison_band_hz": [args.low_min, args.total_max],
            "threshold": args.threshold,
            "track_merge": "maximum score (conservative)",
        },
        "interpretation": (
            "A screening score for broadband low-frequency dominance. "
            "It is not a trained wind classifier."
        ),
        "window_count": len(merged),
        "flagged_windows": flagged_count,
        "flagged_percent": 100 * flagged_count / len(merged),
        "duration_seconds": duration,
        "duration_timecode": format_time(duration),
        "candidate_clean_interval_count": len(intervals),
    }
    _write_windows_csv(output / "windows.csv", merged, len(sources))
    _write_intervals_csv(output / "candidate-clean-intervals.csv", intervals)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.report in ("bare", "both"):
        (output / "report.html").write_text(
            _report_html(summary, merged),
            encoding="utf-8",
        )
    if args.report in ("multimedia", "both"):
        write_preview_segments(
            sources,
            output / args.preview_dir,
            duration,
            chunk_seconds=args.preview_chunk_seconds,
            sample_rate=args.preview_sample_rate,
            bitrate=args.preview_bitrate,
        )
        (output / "report-multimedia.html").write_text(
            _report_multimedia_html(
                summary,
                merged,
                preview_dir=args.preview_dir,
                chunk_seconds=args.preview_chunk_seconds,
                bare_timeline_href="report.html" if args.report == "both" else None,
            ),
            encoding="utf-8",
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lowdom",
        description=(
            "Screen long field recordings for windows dominated by "
            "broadband low-frequency energy."
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
        "--window", type=float, default=10.0, help="window seconds (default: 10)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="screening threshold from 0 to 1 (default: 0.95)",
    )
    parser.add_argument(
        "--analysis-rate",
        type=int,
        default=4000,
        help="downsampled analysis rate (default: 4000 Hz)",
    )
    parser.add_argument("--low-min", type=float, default=10.0)
    parser.add_argument("--low-max", type=float, default=120.0)
    parser.add_argument("--total-max", type=float, default=1000.0)
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
        help="subdirectory for lossy Opus preview chunks (multimedia report)",
    )
    parser.add_argument(
        "--preview-chunk-seconds",
        type=float,
        default=300.0,
        help="length of each Opus preview segment in seconds (default: 300)",
    )
    parser.add_argument(
        "--preview-sample-rate",
        type=int,
        default=48000,
        help="sample rate for lossy preview audio (default: 48000)",
    )
    parser.add_argument(
        "--preview-bitrate",
        default="48k",
        help="Opus bitrate for preview segments (default: 48k)",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = create_report(args)
    except (ToolError, ValueError) as exc:
        parser.exit(2, f"lowdom: error: {exc}\n")
    for name in ("report.html", "report-multimedia.html"):
        path = output / name
        if path.is_file():
            print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
