
from enum import Enum



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
