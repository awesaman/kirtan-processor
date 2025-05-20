import numpy as np
from pydub import AudioSegment
from .filters_fast import fast_high_pass, fast_low_pass

def apply_low_pass(audio_track, cutoff_freq=8000):
    """Apply fast low-pass filter to remove high frequencies"""
    return fast_low_pass(audio_track, cutoff=cutoff_freq)

def apply_high_pass(audio_track, cutoff_freq=200):
    """Apply fast high-pass filter to remove low frequencies"""
    return fast_high_pass(audio_track, cutoff=cutoff_freq)