import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from field_audio_tools.common import ToolError, probe_audio
from field_audio_tools.soundcite import (
    _padding,
    _embedded_excerpt_time,
    _html_page,
    _validate_sources,
    build_parser,
    create_package,
)


def probe(
    *,
    sample_rate=48000,
    channels=1,
    duration=10.0,
    date="2026-01-15",
    creation_time="12:19:25",
    time_reference="0",
):
    return {
        "streams": [
            {
                "codec_name": "pcm_f32le",
                "sample_fmt": "flt",
                "sample_rate": str(sample_rate),
                "channels": channels,
                "bits_per_sample": 32,
            }
        ],
        "format": {
            "format_name": "wav",
            "duration": str(duration),
            "tags": {
                "date": date,
                "creation_time": creation_time,
                "time_reference": time_reference,
            },
        },
    }


def test_two_track_sources_must_be_synchronized():
    sources = [Path("track1.wav"), Path("track2.wav")]
    assert _validate_sources(sources, [probe(), probe()]) == 48000

    with pytest.raises(ToolError, match="sample rates"):
        _validate_sources(sources, [probe(), probe(sample_rate=44100)])
    with pytest.raises(ToolError, match="mono"):
        _validate_sources(sources, [probe(), probe(channels=2)])
    with pytest.raises(ToolError, match="durations"):
        _validate_sources(sources, [probe(), probe(duration=9.0)])
    with pytest.raises(ToolError, match="creation_time"):
        _validate_sources(
            sources, [probe(), probe(creation_time="12:19:26")]
        )


def test_embedded_excerpt_time_adds_clip_offset():
    assert (
        _embedded_excerpt_time([probe()], 3600)
        == "2026-01-15T13:19:25"
    )


def test_embedded_manifest_is_valid_json():
    manifest = {
        "citation": {
            "title": "Bird <call>",
            "recorded_at": None,
            "location": None,
            "recordist": None,
            "license": "CC BY-NC-SA 4.0",
            "notes": None,
        },
        "clip": {
            "start_timecode": "00:00:00.000",
            "duration_seconds": 1.0,
        },
        "sources": [
            {
                "name": "source.wav",
                "technical": {
                    "sample_rate_hz": 48000,
                    "codec": "pcm_f32le",
                },
            }
        ],
        "artifacts": {
            "lossless_excerpts": [
                {"name": "excerpt.wav", "size_bytes": 192000}
            ]
        },
    }
    page = _html_page(manifest)
    value = re.search(
        r'<script type="application/json" id="soundcite-manifest">(.*?)</script>',
        page,
        re.DOTALL,
    ).group(1)
    assert "<style" not in page.lower()
    assert "stylesheet" not in page.lower()
    assert 'style="' not in page
    assert 'class="' not in page
    assert json.loads(value) == manifest


def test_html_page_can_link_site_package():
    manifest = {
        "citation": {"title": "Example"},
        "clip": {
            "start_timecode": "00:00:00.000",
            "duration_seconds": 1.0,
        },
        "sources": [],
        "artifacts": {"lossless_excerpts": []},
    }
    page = _html_page(manifest, site_package_href="index.html")
    assert '<a href="index.html">Open the site package page</a>' in page


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg is required for the integration test",
)
def test_package_has_sample_ranges_and_complete_checksums(tmp_path):
    sources = [tmp_path / "track1.wav", tmp_path / "track2.wav"]
    for index, source in enumerate(sources, start=1):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={index * 440}:sample_rate=48000:duration=1",
                "-c:a",
                "pcm_f32le",
                str(source),
            ],
            check=True,
        )

    output = tmp_path / "package"
    args = build_parser().parse_args(
        [
            *(str(source) for source in sources),
            "--start",
            "0.25",
            "--duration",
            "0.5",
            "--output",
            str(output),
        ]
    )
    create_package(args)

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["clip"]["start_sample"] == 12000
    assert manifest["clip"]["duration_samples"] == 24000
    assert manifest["clip"]["end_sample"] == 36000
    assert manifest["artifacts"]["spectrogram"]["source"] == [
        "excerpt-track1.wav",
        "excerpt-track2.wav",
    ]
    assert "path_as_provided" not in manifest["sources"][0]

    excerpt_probe = probe_audio(output / "excerpt-track1.wav")
    assert float(excerpt_probe["format"]["duration"]) == pytest.approx(0.5)

    checksum_lines = (output / "SHA256SUMS").read_text().splitlines()
    assert {line.split("  ", 1)[1] for line in checksum_lines} == {
        "excerpt-track1.wav",
        "excerpt-track2.wav",
        "preview.ogg",
        "spectrogram.png",
        "manifest.json",
        "index-bare.html",
        "index.html",
    }
    for line in checksum_lines:
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected


def _pad_args(**overrides):
    """A parsed namespace with the padding flags at their defaults."""
    argv = ["in.wav", "--start", "10", "--duration", "5", "--output", "out"]
    for flag, value in overrides.items():
        argv += [f"--{flag.replace('_', '-')}", value]
    return build_parser().parse_args(argv)


def test_padding_defaults_to_nothing():
    assert _padding(_pad_args()) == (0.0, 0.0)


def test_pad_sets_both_ends():
    assert _padding(_pad_args(pad="3")) == (3.0, 3.0)


def test_specific_flags_override_pad():
    assert _padding(_pad_args(pad="3", pad_end="4")) == (3.0, 4.0)
    assert _padding(_pad_args(pad="3", pad_start="1")) == (1.0, 3.0)


def test_padding_accepts_timecodes():
    assert _padding(_pad_args(pad_start="00:00:02.5")) == (2.5, 0.0)


def test_negative_padding_is_refused():
    with pytest.raises(ToolError):
        _padding(_pad_args(pad="-1"))


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg is required for the integration test",
)
def test_padding_widens_the_excerpt_and_records_the_cited_range(tmp_path):
    source = tmp_path / "track.wav"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi",
            "-i", "sine=frequency=440:sample_rate=48000:duration=3",
            "-c:a", "pcm_f32le", str(source),
        ],
        check=True,
    )
    output = tmp_path / "package"
    create_package(
        build_parser().parse_args(
            [
                str(source), "--start", "1.0", "--duration", "1.0",
                "--pad-start", "0.25", "--pad-end", "0.5",
                "--output", str(output),
            ]
        )
    )
    clip = json.loads((output / "manifest.json").read_text())["clip"]

    # The top-level range is what the WAV holds, because the checksums cover it.
    assert clip["start_sample"] == 36000
    assert clip["duration_samples"] == 84000
    # The cited range is what the user pointed at.
    assert clip["cited"]["start_sample"] == 48000
    assert clip["cited"]["duration_samples"] == 48000
    assert clip["padding"] == {
        "start_seconds": 0.25,
        "end_seconds": 0.5,
        "start_requested_seconds": 0.25,
        "end_requested_seconds": 0.5,
        "start_clamped": False,
        "end_clamped": False,
    }
    assert float(probe_audio(output / "excerpt.wav")["format"]["duration"]) == (
        pytest.approx(1.75)
    )


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg is required for the integration test",
)
def test_padding_is_clamped_at_the_ends_and_says_so(tmp_path):
    source = tmp_path / "track.wav"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi",
            "-i", "sine=frequency=440:sample_rate=48000:duration=2",
            "-c:a", "pcm_f32le", str(source),
        ],
        check=True,
    )
    output = tmp_path / "package"
    # Asking for a second of context at each end of a two-second file, around a
    # range that starts a quarter second in and runs to the last quarter second.
    create_package(
        build_parser().parse_args(
            [
                str(source), "--start", "0.25", "--duration", "1.5",
                "--pad", "1", "--output", str(output),
            ]
        )
    )
    clip = json.loads((output / "manifest.json").read_text())["clip"]
    assert clip["start_sample"] == 0
    assert clip["padding"]["start_seconds"] == 0.25
    assert clip["padding"]["start_clamped"] is True
    assert clip["padding"]["end_seconds"] == 0.25
    assert clip["padding"]["end_clamped"] is True


def test_a_run_without_padding_keeps_the_old_manifest_shape():
    """Existing packages must not gain fields they never had."""
    manifest = {
        "clip": {"start_timecode": "00:00:01.000", "duration_seconds": 2.0},
        "citation": {"title": "t"},
        "sources": [],
        "artifacts": {"lossless_excerpts": [], "checksums": {"name": "SHA256SUMS"}},
    }
    page = _html_page(manifest)
    assert "Cited range" not in page
    assert "Padding" not in page
