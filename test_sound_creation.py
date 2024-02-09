import librosa
import os
import numpy as np


def test_scaling_consistency():
    full_wave, sr = librosa.load(os.path.join("created_samples", "trumpet_pure_440.wav"))
    oct_wave, sr = librosa.load(os.path.join("created_samples", "trumpet_pure_octave_440.wav"))
    oct_complement_wave, sr = librosa.load(os.path.join("created_samples", "trumpet_pure_missing_octave_440.wav"))
    check_min_max = lambda x, m: print(f"{m} Min: {np.min(x)}, {m} Max: {np.max(x)}")
    reconstructed_wave = oct_wave + oct_complement_wave
    check_min_max(full_wave, "Full")
    check_min_max(oct_wave, "Octave")
    check_min_max(oct_complement_wave, "Octave Complement")
    check_min_max(reconstructed_wave, "Reconstructed")
    print(len(reconstructed_wave))
    print(len(full_wave))
    assert np.allclose(reconstructed_wave, full_wave)