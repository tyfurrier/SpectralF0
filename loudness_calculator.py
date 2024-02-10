from enum import Enum
from typing import List, Tuple
# citation: https://www.dsprelated.com/showcode/174.php

class FScale(Enum):
    MEL = "mel"
    MIDI = "midi"
    RAW = "raw"

def hearing_threshold(f: float):
    """ Returns the hearing threshold in dB at a given frequency"""
    if f == 1000:
        return 2.4  # dB
    else:
        raise NotImplementedError

def _binary_search_boundaries(values: List[float], target: float) -> Tuple[float, float]:
    """ Returns the two values in an ordered list that encapsulate (min and max of)
    the given target. If the given target is in the list, it will return that value.
    """
    bottom, top = 0, len(values)
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
    f = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
         1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500]

    af = [0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349, 0.330, 0.315,
          0.301, 0.288, 0.276, 0.267, 0.259, 0.253, 0.250, 0.246, 0.244, 0.243, 0.243,
          0.243, 0.242, 0.242, 0.245, 0.254, 0.271, 0.301]

    Lu = [-31.6, -27.2, - 23.0 -19.1 -15.9 -13.0 -10.3 -8.1 -6.2 -4.5 -3.1,
          -2.0 -1.1 -0.4, 0.0, 0.3, 0.5, 0.0 -2.7 -4.1 -1.0, 1.7,
          2.5, 1.2 -2.1 -7.1 -11.2 -10.7 -3.1]

    Tf = [78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9, 14.4,
          11.4, 8.6, 6.2, 4.4, 3.0, 2.2, 2.4, 3.5, 1.7 -1.3 -4.2,
          -6.0 -5.4 -1.5, 6.0, 12.6, 13.9, 12.3]
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
    t_f = hearing_threshold(
        f)  # threshold of hearing in dB at frequency f based on ISO 226:2023
    t_r = hearing_threshold(1000)  # threshold of hearing in dB at 1000 Hz based on ISO 226:2023
    a_f = 0  # exponent for loudness perception at frequency f based on ISO 226:2023
    a_r = 0.3  # exponent for loudness perception at 1000 Hz (reference) based on ISO 226:2023
    l_u = 0  # magnitude of linear transfer function normalized at 1000 Hz in dB
    # p_0 = 20e-6  # reference sound pressure in Pa
    # p_a = 0  # absolute sound pressure
    db_of_f_at_phon = (10 / a_f) * math.log10(
        (4e-10 ** 2) ** (a_r - a_f)
        * (10 ** (a_r * phon / 10) - 10 ** (a_r * t_r / 10))
        + 10 ** (a_f * (t_f + l_u / 10))
    ) - l_u


def centroid_shift(scale: FScale, scale1: List[float], scale2: List[float]):
    """
    Calculate the centroid shift of a complex sound
    ASSUMING all harmonics are of equal PERCIEVED loudness (phon)
    :param scale: scale to calculate the centroid shift
    :return: centroid shift
    """
    return sum([f * scale[f] for f in scale]) / sum(scale.values())