import pytest

from field_audio_tools.common import format_time, parse_time


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("12.5", 12.5),
        ("01:02.5", 62.5),
        ("01:02:03.25", 3723.25),
    ],
)
def test_parse_time(value, expected):
    assert parse_time(value) == expected


@pytest.mark.parametrize("value", ["", "-1", "1:60", "1:2:60", "a:b"])
def test_parse_time_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_time(value)


def test_format_time():
    assert format_time(3723.25) == "01:02:03.250"
