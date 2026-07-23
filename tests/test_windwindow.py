import numpy as np

from field_audio_tools.windwindow import _clean_intervals, measure_window


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
