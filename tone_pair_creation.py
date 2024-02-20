from interface import pure_tone_synthesizer, scale_numpy_wave
import loudness_calculator as lc
import numpy as np
from helpers import _prep_directory, _custom_write
import os
from typing import Tuple, List
import pandas as pd

def pure_tone_synthesizer(fundamental: int,
                          harmonic_decibels: list = None,
                          plot: bool = False,
                          bitrate: int = 44100,
                          length: int = 1,
                          normalize: bool = False,
                          amplitude_of_80dB_SPL: float = None,
                          custom_minmax: tuple = None) -> Tuple[np.ndarray, int]:
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
        custom_minmax (tuple)
        """
    max_db = 0.5  # amplitude for 80dB pure tone, so we can combine four without clipping
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
        return full_second, bitrate

def create_test_tone(hz: float):
    wave, sr = pure_tone_synthesizer(fundamental=hz,
                                     harmonic_decibels=[lc.hearing_threshold(f=hz)],
                                     length=2)
    folder_path = os.path.join(__file__, os.pardir, "created_samples")
    _prep_directory(folder_path=folder_path,
                    default_path=folder_path,
                    clear_dir=False)
    _custom_write(path=os.path.join(folder_path, f"test_tone_{hz}.wav"), sr=sr, wave=wave,
                  overwrite=True)

def create_max_pairs(phon_levels: list = None):
    if phon_levels is None:
        phon_levels = [30, 40, 50, 60]
    # read jake_pairs.csv into a pandas dataframe
    jake_pairs: pd.DataFrame = pd.read_csv("jake_pairs.csv")
    output_folder_path = os.path.join(__file__, os.pardir, "created_samples", "first_max_pairs")
    for i, row in jake_pairs.iterrows():
        for phon in phon_levels:
            sounds_in_combination: List[Tuple[np.ndarray, int]] = []
            for part in ["f1", "f2"]:
                max_harmonic = int(row[f"{part} h3"])
                decibel_list = [None for _ in range(max_harmonic)]
                decibel_list[0] = lc.db_from_phon(f=row[part],
                                                  phon=phon)
                for h in ["h1", "h2", "h3"]:
                    harmonic = int(row[f"{part} {h}"])
                    hz_of_harmonic = row[part] * harmonic
                    decibel_list[harmonic - 1] = lc.db_from_phon(f=hz_of_harmonic,
                                                                 phon=phon)
                wave, sr = pure_tone_synthesizer(fundamental=row[part],
                                                 harmonic_decibels=decibel_list,
                                                 length=1)
                sounds_in_combination.append((wave, sr))
            np.append(sounds_in_combination[0][0],
                      np.append(np.zeros(sr // 2), sounds_in_combination[1][0]))
            pair_folder = os.path.join(output_folder_path, row["Pair #"])
            _prep_directory(default_path=pair_folder,
                            clear_dir=True)
            _custom_write(path=os.path.join(__file__, os.pardir, "created_samples",
                                           f"{row['f1']}_{row['f2']}_{phon}phon.wav"),
                          wave=sounds_in_combination[0][0],
                          sr=sr,
                          overwrite=True)

    
if __name__ == "__main__":
    create_test_tone(1000)