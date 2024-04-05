import pytest
import pandas as pd
from typing import List
from enums import FScale
from tone_pair_creation import TonePairPart, TonePair, get_tone_pair_list

@pytest.fixture
def full_thirty_tone_pairs():
    tone_pairs: List[TonePair] = get_tone_pair_list(only_used=True)
    yield tone_pairs

@pytest.fixture
def first_pair(full_thirty_tone_pairs):
    yield full_thirty_tone_pairs[0]

def test_all_tones_direction(full_thirty_tone_pairs):
    for pair in full_thirty_tone_pairs:
        print(f'f1: {pair.first.f}')
        print(f'f2: {pair.second.f}')
        print(pair.first.f < pair.second.f)


def test_spectroid(first_pair):
    assert first_pair.second.spectroid(fscale=FScale.HZ) == 500
    assert first_pair.second.spectroid(fscale=FScale.MEL) == FScale.HZ.to_mel(value=500)