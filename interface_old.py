from interface import scale_numpy_wave
import math
import tkinter as tk

import matplotlib.pyplot
import scipy.fft
import librosa
import sklearn.preprocessing
from playsound import playsound
import numpy as np
from scipy.io.wavfile import write
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import os, shutil


def fft_of_file(fname: str):
    """ Plots the fft of the given file.
    Args:
        fname (str): The file name.
    """
    # spectrogram = scipy.signal.spectrogram(data, bitrate)
    import librosa

    # Load the audio file
    audio, sr = librosa.load(fname)

    # Create the spectrogram
    spectrogram = librosa.stft(audio)
    plot_fft(data=audio, sr=sr)
    # # Plot the STFT
    # D = np.abs(librosa.stft(audio))
    # # fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    # fig, ax = plt.subplots()
    # img = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), sr=bitrate, y_axis='log', x_axis='time',
    #                          ax=ax)
    # fig.colorbar(img, ax=ax,
    #              # format="%+2.f dB"
    #              )

def autocorrelation(data, sr):
    ac = librosa.autocorrelate(data)[1:]
    x_ticks = [sr // (i + 1) for i in range(len(ac))]
    # for x, y in zip(x_ticks, ac):
    #     plt.plot([x], [y])
    # plt.xscale('log')
    x_ticks = [(512**i) for i in range(5)]
    plt.xticks(x_ticks, labels=x_ticks)
    print(x_ticks)
    y_ticks = [ac[sr // x] for x in x_ticks]
    data = {'x': x_ticks, 'y': ac}
    # plt.plot(x_ticks, y_ticks)
    example_data = [4096, 512, 1024, 2048]
    plt.plot(example_data, [200, -700, 700, -700])
    plt.show()


def plot_fft(data, sr):
    data = np.array(data).astype(float)
    D = np.abs(librosa.stft(data))
    plt.figure()
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), sr=sr, y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.show()
    return D



def violin_sound(window: int,
                 formant_width: int = 100,
                 fundamental: int = 256):
    """ Plays a violin sound with the duration of the given window size in milliseconds.
    Args:
        window (int): The duration in milliseconds.
        formant_width (int): The width of the formant in cents. The percieved magnitude at the formant will be the same
        but the actual magnitude at fine points in the formant will be different. A formant width of 0 will result in a
        pure tone at the fundamental. A formant width of 100 should result in frequencies at the f plus and minus 1 and
        in between.
        fundamental (int): The fundamental frequency in Hz. Defaults to 440.
        """
    db_levels = [60, 50, 45, 50, 43, 48, 37, 48, 30, 20]
    equation = """
    db = 20log(y)
    db = log(y^20)
    10^db = y^20
    10^(db/20) = y"""
    to_magnitude = lambda db: 10 ** (db / 20)
    partials_magnitude = [to_magnitude(db) for db in db_levels]
    partials_magnitude = db_levels
    bitrate = 44100
    master_vol = 0.0000001
    freq = 440
    length = window // 1000

    # create time values
    t = np.linspace(0, length, length * bitrate, dtype=np.float32)
    # generate y values for signal
    y = partials_magnitude[0] * master_vol * np.sin(2 * np.pi * fundamental * t)
    for harmonic, level in enumerate(partials_magnitude[1:]):
        harmonic += 1
        frequency = fundamental * harmonic + fundamental
        y += level * master_vol * np.sin(2 * np.pi * frequency * t)
        print(f"Harmonic {harmonic} at {frequency} Hz with magnitude {level}")

    # save to wave file
    write("violin.wav", bitrate, y)
    return y, bitrate

def sound_oscillators(frequencies: list, magnitudes: list, length: int, bitrate: int = 44100):
    pass


def violin_sound_v2(window: int,
                 formant_width: int = 100,
                 fundamental: int = 256,
                    plot=True):
    """ Plays a violin sound with the duration of the given window size in milliseconds.
    Args:
        window (int): The duration in milliseconds.
        formant_width (int): The width of the formant in cents. The percieved magnitude at the formant will be the same
        but the actual magnitude at fine points in the formant will be different. A formant width of 0 will result in a
        pure tone at the fundamental. A formant width of 100 should result in frequencies at the f plus and minus 1 and
        in between.
        fundamental (int): The fundamental frequency in Hz. Defaults to 440.
        """

    bitrate = 44100
    length = window // 1000

    # create time values
    t = np.linspace(0, length, length * bitrate, dtype=np.float32)
    magnitudes = []
    sample_data, sample_sr = librosa.load(os.path.join('external_samples', 'violin_solo.wav'))
    stft = np.abs(librosa.stft(sample_data, n_fft=bitrate, win_length=bitrate))
    # stft = plot_fft(data=sample_data, sr=sample_sr)
    for f in range(1, bitrate//2//fundamental - 1):
        frequency = fundamental * f
        semi_down = math.floor(librosa.midi_to_hz(librosa.hz_to_midi(frequency) - 1))
        semi_up = math.ceil(librosa.midi_to_hz(librosa.hz_to_midi(frequency) + 1))
        magnitude_data = []
        for hz in range(semi_down, semi_up):
            if hz >= stft.shape[0]:
                break
            magnitude_data.append(np.sum(stft[hz]))  # todo: this hz isn't doing anything, try dividing by 4. need to map to the right frequency
        magnitudes.append(np.average(magnitude_data))
    components = []
    for f, level in enumerate(magnitudes):
        frequency = fundamental * (f + 1)
        components.append(level * np.sin(2 * np.pi * frequency * t))
        print(f"Harmonic {f + 1} at {frequency} Hz with magnitude {level}")
    final_wave = np.sum(components, axis=0).astype(np.float32)
    min = np.min(final_wave)
    max = np.max(final_wave)
    scaled_wave = []
    for bit in final_wave:
        scaled_wave.append((((bit - min) / (max - min)) * 2 - 1) * 2147483647)
    if plot:
        plt.plot(final_wave[:1000])
        plt.show()
        plt.plot(scaled_wave[:1000])
        plt.show()
    scaled_wave = np.array(scaled_wave).astype(np.int32)

    # save to wave file
    if os.path.exists("created_samples"):
        shutil.rmtree("created_samples")
    os.mkdir("created_samples")
    # write("violin.wav", bitrate, final_wave)
    from sklearn.preprocessing import normalize

    write(os.path.join("created_samples", "new2_violin.wav"), bitrate, scaled_wave)
    ret_data, ret_sr = librosa.load("new2_violin.wav")
    plt.plot(ret_data[:1000])
    plt.show()
    return ret_data, ret_sr


def saw_wave(frequency: int, length: int, bitrate: int = 44100):
    """ Returns a list of amplitudes for one period of a saw wave"""
    t = np.linspace(0, length, length * bitrate, dtype=np.float32)
    return np.sin(2 * np.pi * frequency * t)


def trumpet_sound_mk1(frequency: int,
                  bitrate: int = 44100,
                  plot: bool = False,
                  start: int = 0,
                  skip: int = 1):
    """ Returns a list of amplitudes for one second of a trumpet playing the given frequency"""
    amplitudes = [1.0,
                  0.8, 0.22, 0.4,
                  0.55
                  ]
    amplitudes = [0.36, 0.6, 0.23, 1, 0.7, 0.55, 0.18, 0.3, 0.05, 0.06]
    decibels = [-22, -19,  -21.5, -26,
                -25, -28, -27, -27,
                -28, -32.5, -38] \
               + [(-65 - -38)/10 * (i + 1) -38 for i in range(10)] \
                + [(-92 - -65)/6 * (i + 1) - 65 for i in range(6)]# -65 by 10k | -92 by 13k
    amplitudes = [librosa.db_to_amplitude(d) for d in decibels]
    length = 1  # seconds of pure tone to generate
    t = np.linspace(0, length, length * bitrate)
    canvas = np.zeros(bitrate)  # one second since we are using integer hz values
    pure_tone = np.sin(2 * np.pi * frequency * t)
    for i in range(start, len(canvas), skip):
        for f, amp in enumerate(amplitudes):
            harmonic = f + 1
            amplitude = amp * pure_tone[(i * harmonic) % len(pure_tone)]
            canvas[i] += amplitude
    full_second = np.array(canvas).astype(np.float32)
    return scale_numpy_wave(wave=full_second, plot=plot), bitrate



def violin_sound_v3(window: int,
                 formant_width: int = 100,
                 fundamental: int = 256,
                    fname="violin.wav",
                    plot=True):
    """ Plays a violin sound with the duration of the given window size in milliseconds.
    Args:
        window (int): The duration in milliseconds.
        formant_width (int): The width of the formant in cents. The percieved magnitude at the formant will be the same
        but the actual magnitude at fine points in the formant will be different. A formant width of 0 will result in a
        pure tone at the fundamental. A formant width of 100 should result in frequencies at the f plus and minus 1 and
        in between.
        fundamental (int): The fundamental frequency in Hz. Defaults to 440.
        """

    bitrate = 44100
    length = window // 1000

    # create time values
    t = np.linspace(0, length, length * bitrate, dtype=np.float32)
    magnitudes = []
    sample_data, sample_sr = librosa.load(os.path.join('external_samples', 'violin_solo.wav'))
    stft = librosa.stft(sample_data, n_fft=bitrate, win_length=bitrate)
    # stft = plot_fft(data=sample_data, sr=sample_sr)
    collection_radius = 0.5
    collection_frequency_up = librosa.midi_to_hz(librosa.hz_to_midi(fundamental) + collection_radius) - fundamental
    collection_frequency_down = fundamental - librosa.midi_to_hz(librosa.hz_to_midi(fundamental) - collection_radius)

    components = []
    for f in range(1, bitrate//2//fundamental - 1):
    # for f in range(1, 2):
        frequency = fundamental * f
        upper_bound = math.ceil(frequency + collection_frequency_up * f)
        lower_bound = math.floor(frequency - collection_frequency_down * f) # todo: multiply collection_freq_down by f
        if f > 20:
            if frequency > 8300:
                drop_off = 60
            else:
                drop_off = (frequency - 4200) / 4100 * 60 + magnitudes[19]
            base_db = magnitudes[0]
            magnitudes.append(base_db - drop_off)
            continue
        semi_down = math.floor(librosa.midi_to_hz(librosa.hz_to_midi(frequency) - 1))
        semi_up = math.ceil(librosa.midi_to_hz(librosa.hz_to_midi(frequency) + 1))
        magnitude_data = []
        for hz in range(lower_bound, upper_bound):
            if hz >= stft.shape[0]:
                break
            magnitude_data.append(np.average(stft[hz]))  # todo: this hz isn't doing anything, try dividing by 4. need to map to the right frequency
        magnitudes.append(np.average(magnitude_data))  # switch sum/average
    for f, level in enumerate(magnitudes):
        frequency = fundamental * (f + 1)
        db_level = level
        level = librosa.db_to_amplitude(level)
        #print(f"Harmonic {f + 1} at {frequency} Hz with amplitude {level} based on db of {db_level}")

        if formant_width != 0:  # this is where we widen partials
            components_of_partial =[]
            local_width = formant_width * f
            local_level = level/(local_width*2)
            for hz in range(frequency - local_width, frequency + local_width):
                components_of_partial.append(local_level * np.sin(2 * np.pi * frequency * t))
            components.append(np.sum(components_of_partial, axis=0))
        else:
            components.append(level * np.sin(2 * np.pi * frequency * t))

    final_wave = np.sum(components, axis=0).astype(np.float32)
    scaled_wave = scale_numpy_wave(final_wave, plot=plot)

    # save to wave file
    # if os.path.exists("created_samples"):
    #     shutil.rmtree("created_samples")
    # os.mkdir("created_samples")

    write(os.path.join("created_samples", fname), bitrate, scaled_wave)
    if plot:
        ret_data, ret_sr = librosa.load(fname)
        plt.plot(ret_data[:1000])
        plt.show()
    return scaled_wave, bitrate

def play_trumpet_sound(frequency: int = 440, bitrate: int = 44100):
    import simpleaudio as sa
    trumpet_second, bitrate = trumpet_sound(frequency=frequency, bitrate=bitrate, plot=False,
                                            skip=2)
    fname = f"trumpet_pure_missing_octave_{frequency}.wav"
    write(os.path.join("created_samples", fname), bitrate, trumpet_second)

def old_generation():
    import simpleaudio as sa
    versions = []
    for i in range(1, 8, 2):
        our_version, our_bitrate = violin_sound_v3(window=1000,
                                                   formant_width=i * 5,
                                                   plot=False,
                                                   fname=f'violin_width_{i * 5}.wav')
        versions.append((our_version, our_bitrate))
        plot_fft(data=our_version, sr=our_bitrate)
    while input("Play again?") != "n":
        for sound, sr in versions:
            play_obj = sa.play_buffer(our_version, 1, 4, our_bitrate)
            play_obj.wait_done()

    audio, sr = librosa.load(os.path.join("external_samples", "violin_solo.wav"))
    sample_fft = plot_fft(data=audio, sr=sr)
    inverse = librosa.istft(sample_fft)
    play_obj = sa.play_buffer(inverse, 1, 4, sr)
    play_obj.wait_done()

if __name__ == "__main__":
    # fft_of_file(os.path.join(
    #     # os.path.pardir,
    #     "external_samples",
    #     "violin_solo.wav"))
    for f in range(409, 471):
        play_trumpet_sound(frequency=f, bitrate=44100)
    # old_generation()
    # D = np.abs(librosa.stft(audio, n_fft))
    # plt.figure()
    # librosa.display.specshow(D, sr=sr, y_axis='log', x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    # plt.show()
    # playsound(os.path.join(
    #     # os.path.pardir,
    #     "external_samples",
    #     "violin_solo.wav"))