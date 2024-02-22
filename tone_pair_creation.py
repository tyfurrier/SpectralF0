from interface import scale_numpy_wave
import loudness_calculator as lc
import numpy as np
from helpers import _prep_directory, _custom_write, _created_samples_folder_path
import os
from typing import Tuple, List
from src.custom_errors import ClippingError
import pandas as pd
import logging


def pure_tone_synthesizer(fundamental: int,
                          harmonic_decibels: list = None,
                          plot: bool = False,
                          bitrate: int = 44100,
                          length: int = 1,
                          normalize: bool = False,
                          amplitude_of_80dB_SPL: float = None,
                          custom_minmax: tuple = None,
                          clipping_tolerance: int = 2) -> Tuple[np.ndarray, int]:
    """ Returns one second of the fundamental and its harmonics at the given decibel SPL levels. The SPL is
    calculated from an amplitude of 0.5 equating to an SPL of 80dB, the amplitude required to reach 80dB can be
    changed by setting the amplitude_of_80dB_SPL parameter.
    The returned wave is a numpy wave that has an amplitude range of -1 to 1 and is written as a
    32-bit floating-point sound file.
    The amplitudes list should include the fundamental and None for -inf decibels.
    Args:
        fundamental (int)
        harmonic_decibels
        plot (bool)
        bitrate (int)
        length (float) The number of seconds to generate
        amplitude_of_80dB_SPL (float): See docstring
        normalize (bool): If True, the wave will be scaled to fit the range of -1 to 1
          custom_minmax (tuple): If Normalize is True, this will be passed to scale_numpy_wave
        clipping_tolerance (int): The number of digits to round each bit before checking for clipping. For example, if
        a sound reaches an amplitude of 1.001 and `normalize` is set to False, then a ClippingError will be raised if
        `clipping_tolerance` is set to 3 or greater and no error will be raised if `clipping_tolerance` is 2.
        """
    max_db = 0.7  # amplitude for 80dB pure tone, so we can combine four without clipping
    if amplitude_of_80dB_SPL is not None:
        max_db = amplitude_of_80dB_SPL
    if harmonic_decibels is None:
        harmonic_decibels = [0]
    amplitudes = harmonic_decibels.copy()
    for i, db in enumerate(harmonic_decibels):
        if db is None:
            continue
        amplitude_ratio = lc.amplitude_from_db(db - 80)
        amplitudes[i] = max_db * amplitude_ratio
    t = np.linspace(0, length, length * bitrate)
    canvas = np.zeros(bitrate * length)
    for i in range(len(canvas)):
        for f, amp in enumerate(amplitudes):
            if amp is None:
                continue
            harmonic = f + 1
            # amplitude = amp * pure_tone[(((i + 1) * harmonic) - 1) % len(pure_tone)]
            amplitude = amp * np.sin(2 * np.pi * fundamental * t[i] * harmonic)
            canvas[i] += amplitude
    full_second = np.array(canvas).astype(np.float32)
    if normalize:
        return scale_numpy_wave(wave=full_second, plot=plot, minmax=custom_minmax), bitrate
    else:
        for bit in full_second:
            if round(abs(bit), 2) > 1:
                raise ClippingError(f"Clipping detected: {bit}\n"
                                    f"Set `normalize` to True or reduce the cummulative amplitude"
                                    f"Arguments: {locals()}")
        return full_second, bitrate


def create_test_tone(hz: float):
    wave, sr = pure_tone_synthesizer(fundamental=hz,
                                     harmonic_decibels=[lc.hearing_threshold(f=hz)],
                                     length=2)
    folder_path = _created_samples_folder_path()
    _prep_directory(folder_path=folder_path,
                    default_path=folder_path,
                    clear_dir=False)
    _custom_write(path=os.path.join(folder_path, f"test_tone_louder_{hz}.wav"), sr=sr, wave=wave,
                  overwrite=True)


def create_tone_pairs(phon_levels: list = None):
    if phon_levels is None:
        phon_levels = [5, 10, 15, 20, 30, 40, 50, 60, 70]
    # read jake_pairs.csv into a pandas dataframe
    jake_pairs: pd.DataFrame = pd.read_csv("all_pairs.csv")
    output_folder_path = os.path.join(_created_samples_folder_path(), "tone_pairs_pilot")
    _prep_directory(folder_path=output_folder_path,
                    clear_dir=True)
    pair_filter = {1, 2, 3, 4, 6, 8, 22, 23, 24, 26}
    for i, row in jake_pairs.iterrows():
        pair_number = int(row["PAIR"])
        if pair_number not in pair_filter:
            continue
        all_root = True
        if not all_root:
            pair_folder = os.path.join(output_folder_path, f'Pair {round(row["PAIR"])}')
            _prep_directory(folder_path=pair_folder,
                            clear_dir=True)
        else:
            pair_folder = output_folder_path
        for phon in phon_levels:
            sounds_in_combination: List[Tuple[np.ndarray, int]] = []
            for part in ["f1", "f2"]:
                max_harmonic = int(row[f"{part} h3"])
                decibel_list = [None for _ in range(max_harmonic)]
                decibel_list[0] = lc.db_from_phon(f=row[part],
                                                  phon=phon)
                logging.info(f'Creating pair {row["PAIR"]} at {decibel_list[0]} for {phon} phon at {row[part]}')
                for h in ["h1", "h2", "h3"]:
                    harmonic = int(row[f"{part} {h}"])
                    hz_of_harmonic = row[part] * harmonic
                    decibel_list[harmonic - 1] = lc.db_from_phon(f=hz_of_harmonic,
                                                                 phon=phon)
                    logging.info(f'{row["PAIR"]} PAIR: \n'
                                 f'{harmonic}th harmonic at {decibel_list[harmonic - 1]} dB for '
                                 f'{phon} phon at {row[part] * harmonic}')
                try:
                    wave, sr = pure_tone_synthesizer(fundamental=row[part],
                                                     harmonic_decibels=decibel_list,
                                                     length=1,
                                                     clipping_tolerance=1)
                except ClippingError as ce:
                    logging.error(ce)
                    break
                sounds_in_combination.append((wave, sr))
            else:
                appended_pair: np.ndarray = np.append(sounds_in_combination[0][0],
                                                      np.append(np.zeros(sr // 2).astype(np.float32),
                                                                sounds_in_combination[1][0]))
                _custom_write(path=os.path.join(pair_folder, f"{pair_number}_{phon}_phon.wav"),
                              wave=appended_pair,
                              sr=sr,
                              overwrite=True)

def annotate_tone_pairs(phon_levels: list = None):

    # read all_pairs.csv into a pandas dataframe
    tone_pair_csv: pd.DataFrame = pd.read_csv("all_pairs.csv")
    output_folder_path = _created_samples_folder_path()  # we will write a new one where we keep sounds
    for i, row in tone_pair_csv.iterrows():
        for part in ["f1", "f2"]:
                # add column for mel, hz, and midi spectroid for each part
                # add column for spectroid shift in mel, hz, and midi
                # add column for spectroid shift when only considering harmonics
                # add column for min and max inter-harmonic distance
                # add column for fundamental distance in mel, hz, and midi
                # add column for common frequency
                # add column for distance from fundamental to common frequency in each scale mel hz midi
                # add column for distance from fundamental to harmonics in each scale mel hz midi
                max_harmonic = int(row[f"{part} h3"])
                max_harmonic = int(row[f"{part} h3"])
                decibel_list = [None for _ in range(max_harmonic)]
                phon = 1
                decibel_list[0] = lc.db_from_phon(f=row[part],
                                                  phon=phon)
                logging.info(
                    f'Creating pair {row["PAIR"]} at {decibel_list[0]} for {phon} phon at {row[part]}')
                for h in ["h1", "h2", "h3"]:
                    harmonic = int(row[f"{part} {h}"])
                    hz_of_harmonic = row[part] * harmonic
                    decibel_list[harmonic - 1] = lc.db_from_phon(f=hz_of_harmonic,
                                                                 phon=phon)
                    logging.info(f'{row["PAIR"]} PAIR: \n'
                                 f'{harmonic}th harmonic at {decibel_list[harmonic - 1]} dB for '
                                 f'{phon} phon at {row[part] * harmonic}')

    
if __name__ == "__main__":
    create_test_tone(1000)
    logging.basicConfig(level=logging.ERROR)
    create_tone_pairs(
        phon_levels=[10, 30, 50])