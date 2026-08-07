import numpy as np

from field_audio_tools.lowdom import (
    _clean_intervals,
    _report_html,
    _report_multimedia_html,
    build_parser,
    measure_window,
)


def band_noise(
    sample_rate: int,
    seconds: float,
    minimum_hz: float,
    maximum_hz: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = round(sample_rate * seconds)
    frequencies = np.fft.rfftfreq(count, 1 / sample_rate)
    spectrum = np.zeros(frequencies.size, dtype=np.complex128)
    mask = (frequencies >= minimum_hz) & (frequencies <= maximum_hz)
    spectrum[mask] = rng.normal(size=mask.sum()) + 1j * rng.normal(size=mask.sum())
    values = np.fft.irfft(spectrum, n=count)
    return values / np.max(np.abs(values)) * 0.25


def test_low_frequency_noise_scores_above_high_frequency_noise():
    sample_rate = 4000
    low = band_noise(sample_rate, 2, 15, 110, seed=1)
    high = band_noise(sample_rate, 2, 400, 900, seed=2)

    *_, low_score = measure_window(low, sample_rate)
    *_, high_score = measure_window(high, sample_rate)

    assert low_score > 0.45
    assert high_score < 0.05
    assert low_score > high_score


def test_silence_is_not_flagged_by_score():
    *_, score = measure_window(np.zeros(8000), 4000)
    assert score < 0.01


def test_clean_intervals_merge_adjacent_windows():
    rows = [
        {"start_seconds": 0.0, "end_seconds": 10.0, "score": 0.1},
        {"start_seconds": 10.0, "end_seconds": 20.0, "score": 0.2},
        {"start_seconds": 20.0, "end_seconds": 30.0, "score": 0.8},
        {"start_seconds": 30.0, "end_seconds": 40.0, "score": 0.3},
    ]
    assert _clean_intervals(rows, 0.6) == [(0.0, 20.0), (30.0, 40.0)]


def test_default_threshold_is_calibrated_value():
    args = build_parser().parse_args(["recording.wav", "--output", "report"])
    assert args.threshold == 0.95
    assert args.report == "bare"


def test_report_html_is_bare_html4():
    summary = {
        "sources": [{"name": "Tr1.WAV"}, {"name": "Tr2.WAV"}],
        "duration_timecode": "00:01:00.000",
        "parameters": {"window_seconds": 10.0, "threshold": 0.95},
        "flagged_windows": 0,
        "window_count": 6,
        "flagged_percent": 0.0,
    }
    rows = [
        {
            "start_seconds": 0.0,
            "end_seconds": 10.0,
            "score": 0.4,
            "flagged": False,
        }
    ]
    page = _report_html(summary, rows)
    assert page.startswith('<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN">')
    assert "<style" not in page
    assert 'border="1" cellpadding="4" cellspacing="0"' in page
    assert "getElementById(\"timeline\")" in page


def test_report_multimedia_html_is_bare_html4():
    summary = {
        "sources": [{"name": "Tr1.WAV"}],
        "duration_timecode": "00:01:00.000",
        "parameters": {"window_seconds": 10.0, "threshold": 0.95},
        "flagged_windows": 1,
        "window_count": 6,
        "flagged_percent": 16.7,
    }
    rows = [
        {
            "start_seconds": 0.0,
            "end_seconds": 10.0,
            "score": 0.99,
            "flagged": True,
        }
    ]
    page = _report_multimedia_html(
        summary,
        rows,
        preview_dir="recording-preview-48k",
        chunk_seconds=300.0,
        bare_timeline_href="report.html",
    )
    assert page.startswith('<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN">')
    assert 'id="enable"' in page
    assert "chunkSeconds = 300" in page
    assert "copyTimestamp" in page
    assert 'href="report.html"' in page
    page_without = _report_multimedia_html(
        summary,
        rows,
        preview_dir="preview",
        chunk_seconds=60.0,
        bare_timeline_href=None,
    )
    assert 'href="report.html"' not in page_without
