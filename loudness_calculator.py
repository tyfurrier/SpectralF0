from enum import Enum
from enums import FScale
from typing import List, Tuple, Callable, Dict
import librosa
import math
# citation: https://www.dsprelated.com/showcode/174.php

def convert_pitch_unit(pitch: float, from_scale: FScale, to_scale: FScale) -> float:
    conversion: Dict[FScale, Callable] = {
        FScale.MEL: lambda x, value: x.to_mel(value),
        FScale.HZ: lambda x, value: x.to_hz(value),
        FScale.MIDI: lambda x, value: x.to_midi(value),
    }
    pitch = conversion[to_scale](from_scale, pitch)
    return pitch


FREQUENCIES_FROM_ISO_226 = [
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
         1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500]
ISO226_2023 = {
        20: {"af": 0.635, "Lu": -31.5, "Tf": 78.1},
        25: {"af": 0.602, "Lu": -27.2, "Tf": 68.7},
        31.5: {"af": 0.569, "Lu": -23.1, "Tf": 59.5},
        40: {"af": 0.537, "Lu": -19.3, "Tf": 51.1},
        50: {"af": 0.509, "Lu": -16.1, "Tf": 44.0},
        63: {"af": 0.482, "Lu": -13.1, "Tf": 37.5},
        80: {"af": 0.456, "Lu": -10.4, "Tf": 31.5},
        100: {"af": 0.433, "Lu": -8.2, "Tf": 26.5},
        125: {"af": 0.412, "Lu": -6.3, "Tf": 22.1},
        160: {"af": 0.391, "Lu": -4.6, "Tf": 17.9},
        200: {"af": 0.373, "Lu": -3.2, "Tf": 14.4},
        250: {"af": 0.357, "Lu": -2.1, "Tf": 11.4},
        315: {"af": 0.343, "Lu": -1.2, "Tf": 8.6},
        400: {"af": 0.330, "Lu": -0.5, "Tf": 6.2},
        500: {"af": 0.320, "Lu": 0.0, "Tf": 4.4},
        630: {"af": 0.311, "Lu": 0.4, "Tf": 3.0},
        800: {"af": 0.303, "Lu": 0.5, "Tf": 2.2},
        1000: {"af": 0.300, "Lu": 0.0, "Tf": 2.4},
        1250: {"af": 0.295, "Lu": -2.7, "Tf": 3.5},
        1600: {"af": 0.292, "Lu": -4.2, "Tf": 1.7},
        2000: {"af": 0.290, "Lu": -1.2, "Tf": -1.3},
        2500: {"af": 0.290, "Lu": 1.4, "Tf": -4.2},
        3150: {"af": 0.289, "Lu": 2.3, "Tf": -6.0},
        4000: {"af": 0.289, "Lu": 1.0, "Tf": -5.4},
        5000: {"af": 0.289, "Lu": -2.3, "Tf": -1.5},
        6300: {"af": 0.293, "Lu": -7.2, "Tf": 6.0},
        8000: {"af": 0.303, "Lu": -11.2, "Tf": 12.6},
        10000: {"af": 0.323, "Lu": -10.9, "Tf": 13.9},
        12500: {"af": 0.354, "Lu": -3.5, "Tf": 12.3},
    }

def hearing_threshold(f: float):
    """ Returns the hearing threshold in SPLdB at a given frequency"""
    return _get_iso_value_for_f(f_hz=f, value="Tf")

def estimate_af(f: float):
    scale = FScale.HZ if f >= 4000 else FScale.MIDI
    return _get_iso_value_for_f(f_hz=f,
                         value="af",
                         scale=scale)


def _get_iso_value_for_f(f_hz: float, value: str,
                         scale: FScale = FScale.MEL) -> float:
    """ Returns the inferred value (through interpolation) of a given key for a given
    frequency in the ISO 226-2023 table"""
    frequency_ends = _binary_search_boundaries(FREQUENCIES_FROM_ISO_226, f_hz)
    if frequency_ends[0] == frequency_ends[1]:
        return ISO226_2023[frequency_ends[0]][value]
    bottom_f = convert_pitch_unit(pitch=frequency_ends[0], from_scale=FScale.HZ, to_scale=scale)
    top_f = convert_pitch_unit(pitch=frequency_ends[1], from_scale=FScale.HZ, to_scale=scale)
    target_f = convert_pitch_unit(pitch=f_hz, from_scale=FScale.HZ, to_scale=scale)
    top_val = ISO226_2023[frequency_ends[1]][value]
    bottom_val = ISO226_2023[frequency_ends[0]][value]

    # linearly interpolate the target using the scaled values (y = mx + b)
    m = (top_val - bottom_val)/(top_f - bottom_f)
    interpolated_value = m * (target_f - bottom_f) + bottom_val
    return interpolated_value

def _binary_search_boundaries(values: List[float], target: float) -> Tuple[float, float]:
    """ Returns the two values in an ordered list that encapsulate (min and max of)
    the given target. If the given target is in the list, it will return that value.
    """
    bottom, top = 0, len(values) - 1
    while top - bottom > 1:
        mid = (top + bottom) // 2
        if values[mid] < target:
            bottom = mid
        elif values[mid] > target:
            top = mid
        elif values[mid] == target:
            return values[mid], values[mid]
    return values[bottom], values[top]


def db_from_phon(f: float, phon: float):
    t_f = hearing_threshold(f)  # threshold of hearing in dB at frequency f based on ISO 226:2023
    t_r = hearing_threshold(1000)  # threshold of hearing in dB at 1000 Hz based on ISO 226:2023
    a_f = estimate_af(f=f)  # exponent for loudness perception at frequency f based on ISO 226:2023
    a_r = 0.3  # exponent for loudness perception at 1000 Hz (reference) based on ISO 226:2023
    # magnitude of linear transfer function normalized at 1000 Hz in dB
    l_u = _get_iso_value_for_f(f_hz=f,
                               value="Lu",
                               scale=FScale.HZ)
    # p_0 = 20e-6  # reference sound pressure in Pa
    # p_a = 0  # absolute sound pressure
    static = (a_r * t_r / 10)
    part_1 = (10 / a_f) # 30.9
    part_2 = 4e-10 ** (a_r - a_f)  # 1.6449
    part_3_1 = 10 ** (a_r * phon / 10) #  63.09
    part_3_2 = 10 ** static  # 1.18
    part_4 = 10 ** 0.0969
    part_4 = 10 ** (a_f * (t_f + l_u) / 10)  # 1.2499
    A_f = (part_2) \
          * (
                  part_3_1 - part_3_2
          ) \
          + (part_4)
    db_of_f_at_phon = part_1 * math.log10(
        A_f
    ) - l_u
    return db_of_f_at_phon

def amplitude_from_db(db: float):
    """ Returns the amplitude ratio for an increase/decrease in the given db value"""
    return 10 ** (db / 20)


def centroid_shift(scale: FScale, scale1: List[float], scale2: List[float]):
    """
    Calculate the centroid shift of a complex sound
    ASSUMING all harmonics are of equal PERCIEVED loudness (phon)
    :param scale: scale to calculate the centroid shift
    :return: centroid shift
    """
    return sum([f * scale[f] for f in scale]) / sum(scale.values())