"""Render one spectrogram per detection frame with FFmpeg.

Images are generated ahead of time, from the source audio the analysis ran on
rather than from the lossy preview, and written into the output package. The
report then shows the frame under inspection without any work at view time.

A spectrogram is the cheapest way to see whether a detection looks like a call
at all: a corvid shows harmonic sweeps in the kilohertz bands, while broadband
low-frequency energy with nothing above it is geophony, whatever the model
named it.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .common import ToolError, format_time, require_program

FORMATS = ("webp", "png")

# Log frequency and log amplitude. A linear axis spends most of its height on
# empty ultrasound and squashes everything below 500 Hz into a single line,
# which is exactly the band that decides whether a detection is a bird.
DEFAULT_SIZE = "560x260"

# The first audio stream is selected inside the graph rather than with -map.
# A -map of the audio stream would also route it to the image file, and FFmpeg
# would then fail looking for a webp audio encoder.
_FILTER = (
    "[0:a:0]showspectrumpic=s={size}:legend=1:scale=log:fscale=log:color=intensity"
)


@dataclass(frozen=True)
class DetectionSpectrogram:
    start_seconds: float
    end_seconds: float
    file_name: str


def _file_name(start_seconds: float, extension: str) -> str:
    """Name by start time in milliseconds, so the name survives re-ordering."""
    return f"frame-{round(start_seconds * 1000):09d}.{extension}"


def render_one(
    source: Path,
    destination: Path,
    start_seconds: float,
    duration_seconds: float,
    *,
    size: str,
    ffmpeg: str,
) -> None:
    command = [
        ffmpeg,
        "-v",
        "error",
        "-ss",
        f"{start_seconds:.6f}",
        "-t",
        f"{duration_seconds:.6f}",
        "-i",
        str(source),
        "-lavfi",
        _FILTER.format(size=size),
        "-frames:v",
        "1",
        "-y",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode or not destination.exists():
        detail = result.stderr.decode(errors="replace").strip()
        raise ToolError(
            f"FFmpeg could not render the spectrogram at "
            f"{format_time(start_seconds)}: {detail}"
        )


def render_detections(
    source: Path,
    frames: Sequence[tuple[float, float]],
    destination_dir: Path,
    *,
    size: str = DEFAULT_SIZE,
    image_format: str = "webp",
    progress: bool = False,
) -> tuple[list[DetectionSpectrogram], list[tuple[float, str]]]:
    """Render every frame once.

    Frames are deduplicated by start time: two species detected in the same
    three seconds share one image rather than rendering it twice.

    Returns the spectrograms written and, for anything that failed, the start
    time paired with the reason. A failure is a skip, not an error, so a report
    is still produced when the source audio has moved.
    """
    if image_format not in FORMATS:
        raise ToolError(
            f"--spectrogram-format must be one of: {', '.join(FORMATS)}"
        )
    if not _valid_size(size):
        raise ToolError("--spectrogram-size must look like WIDTHxHEIGHT, e.g. 560x260")
    if not source.is_file():
        raise ToolError(f"Audio file not found: {source}")

    ffmpeg = require_program("ffmpeg")
    destination_dir.mkdir(parents=True, exist_ok=True)

    unique: dict[int, tuple[float, float]] = {}
    for start, end in frames:
        unique.setdefault(round(start * 1000), (start, end))

    rendered: list[DetectionSpectrogram] = []
    skipped: list[tuple[float, str]] = []
    ordered = sorted(unique.values())
    for index, (start, end) in enumerate(ordered, start=1):
        if progress:
            print(
                f"  spectrogram {index}/{len(ordered)}: {format_time(start)}",
                flush=True,
            )
        name = _file_name(start, image_format)
        try:
            render_one(
                source,
                destination_dir / name,
                start,
                max(end - start, 0.1),
                size=size,
                ffmpeg=ffmpeg,
            )
        except ToolError as exc:
            skipped.append((start, str(exc)))
            continue
        rendered.append(DetectionSpectrogram(start, end, name))
    return rendered, skipped


def _valid_size(size: str) -> bool:
    parts = size.lower().split("x")
    if len(parts) != 2:
        return False
    try:
        width, height = (int(part) for part in parts)
    except ValueError:
        return False
    return width > 0 and height > 0


def index_by_start(
    spectrograms: Iterable[DetectionSpectrogram],
) -> dict[int, str]:
    """Map start time in milliseconds to file name, for report lookup."""
    return {
        round(item.start_seconds * 1000): item.file_name for item in spectrograms
    }
