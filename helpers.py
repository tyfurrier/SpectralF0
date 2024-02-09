import numpy as np
from scipy.io.wavfile import write
from enums import SoundFileType
from typing import Optional
import os
import shutil


def _custom_write(path: str,
                  wave: np.ndarray,
                  sr: int,
                  overwrite: bool,
                  file_type: str = SoundFileType.WAV
                  ):
    """ Tries to write the audio using scipy.io.wavfile.write and overwrites it if overwrite
    is True otherwise raises FileExistsError"""
    if file_type != SoundFileType.WAV:
        raise NotImplementedError(f"file_type {file_type} not implemented")
    if os.path.exists(path):
        if overwrite:
            os.remove(path)
        else:
            raise FileExistsError(f"{path} exists")
    write(path, sr, wave)


def _prep_directory(folder_path: Optional[str],
                    default_path: str,
                    clear_dir: bool = True,
                    ):
    """ Creates a directory at folder_path if it doesn't exist and clears it if clear_dir is True"""
    if folder_path is None:
        folder_path = default_path
    if os.path.exists(folder_path):
        if clear_dir:
            shutil.rmtree(folder_path)
            os.mkdir(folder_path)
    else:
        os.mkdir(folder_path)