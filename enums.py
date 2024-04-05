from enum import Enum
import librosa
from typing import Dict, Callable


class Decomposition(Enum):
    FULL = ""
    OCTAVE = "octave_"
    HOLLOW = "missing_octave_"
    RECONSTRUCTED = "reconstructed_"

    def file_naming(self):
        if self.value == "":
            return "full"
        else:
            return self.value[:-1]



class SoundFileType(Enum):
    WAV = "wav"


class FScale(Enum):
    MEL = "mel"
    MIDI = "midi"
    HZ = "hz"

    def to_hz(self, value: float):
        to_hz_function: Dict[FScale: Callable] = {
            FScale.MEL: librosa.mel_to_hz,
            FScale.MIDI: librosa.midi_to_hz,
            FScale.HZ: lambda x: x
        }
        return to_hz_function[self]([value])[0]

    def to_mel(self, value: float):
        hz = self.to_hz(value=value)
        return librosa.hz_to_mel([hz])[0]

    def to_midi(self, value: float):
        hz = self.to_hz(value=value)
        return librosa.hz_to_midi([hz])[0]