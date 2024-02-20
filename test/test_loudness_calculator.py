import librosa

import loudness_calculator as lc
import pytest
from matplotlib import pyplot as plt
import plotly.express as px


def test_amplitude_to_db():
    assert round(lc.amplitude_from_db(6), 1) == 2  # doubling amplitude increases db by 6
    assert round(lc.amplitude_from_db(12), 1) == 4
    assert round(lc.amplitude_from_db(-50), 3) == 0.003


def test_dummy():
    value = 10 ** (0.323 * (13.9 + (-10.9) / 10))
    assert value == 1.2499


def test_db_of_f_at_phon():
    assert round(lc.db_from_phon(f=1000, phon=60), 4) == 60
    db_10k_60 = lc.db_from_phon(f=10000, phon=60)
    assert round(db_10k_60, 4) == 73.2297
    assert round(lc.db_from_phon(f=440, phon=40), 4) == 44.2787


@pytest.mark.skip
def test_plot_iso_interpolations():
    FREQUENCIES = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
         1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500]
    for value in ["af", "Lu", "Tf"]:
        for scale in lc.FScale:
            plt.figure()
            plt.scatter(
                [lc.convert_pitch_unit(pitch=f,
                                       from_scale=lc.FScale.HZ,
                                       to_scale=scale)
                 for f in FREQUENCIES],
                [lc._get_iso_value_for_f(f_hz=f, value=value) for f in FREQUENCIES])
            plt.title(f'{value} by {scale.value}')
            plt.ylabel(value)
            plt.xlabel(scale)
            plt.show()


def test_pitch_conversion():
    lc.FScale.HZ.to_hz(value=40)
    expected = {
        lc.FScale.HZ: 440,
        lc.FScale.MIDI: 69,
        lc.FScale.MEL: 6.6,
    }
    print(expected[lc.FScale.MEL])
    for fs in lc.FScale:
        for ts in lc.FScale:
            assert lc.convert_pitch_unit(pitch=expected[fs],
                                         from_scale=fs,
                                         to_scale=ts) == expected[ts]

def test__binary_search_boundaries():
    f = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
         1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500]
    assert lc._binary_search_boundaries(f, 22.5) == (20, 25)
    assert lc._binary_search_boundaries(f, 12500) == (12500, 12500)
    # todo: test for 15khz, 12500khz, 20hz, and maybe 10 hz
    with pytest.raises(Exception):
        lc._binary_search_boundaries(f, 10)
