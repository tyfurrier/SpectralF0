from loudness_calculator import _binary_search_boundaries
import pytest

def test_pitch_conversion():
    # todo
    pass

def test__binary_search_boundaries():
    f = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
         1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500]
    assert _binary_search_boundaries(f, 22.5) == (20, 25)
    assert _binary_search_boundaries(f, 12500) == (12500, 12500)
    # todo: test for 15khz, 12500khz, 20hz, and maybe 10 hz
    with pytest.raises(Exception):
        _binary_search_boundaries(f, 10)
