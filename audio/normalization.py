import numpy as np
from pydub import AudioSegment

def normalize_peak(audio_track, target_level=-1.0, headroom=2.0, log_callback=print):
    """Normalize audio to target peak level with headroom"""
    # Get audio samples
    samples = np.array(audio_track.get_array_of_samples()).astype(np.float32)
    
    # Calculate current peak level in dB
    max_possible = float(2 ** (audio_track.sample_width * 8 - 1))
    peak = np.max(np.abs(samples)) / max_possible if max_possible > 0 else 0
    peak_db = 20 * np.log10(peak) if peak > 0 else -80
    
    # Calculate required gain adjustment
    gain_db = target_level - peak_db
    
    # Log normalization info
    log_callback(f"Peak normalization: {peak_db:.2f}dB -> {target_level:.2f}dB (gain: {gain_db:.2f}dB)")
    
    # Apply gain
    return audio_track + gain_db

def normalize_rms(audio_track, target_level=-18.0, headroom=2.0, log_callback=print):
    """Normalize audio to target RMS level with headroom"""
    # Get current RMS level
    current_rms = audio_track.rms
    if current_rms == 0:
        log_callback("[DEBUG] RMS normalization: silent audio (RMS=0), skipping gain adjustment.")
        return audio_track
    # Determine the correct reference for dBFS based on sample width
    sample_width = audio_track.sample_width  # bytes per sample
    ref = float(2 ** (8 * sample_width - 1))
    current_db = 20 * np.log10(current_rms / ref) if current_rms > 0 else -80
    log_callback(f"[DEBUG] RMS normalization: raw RMS={current_rms}, ref={ref}, computed dBFS={current_db:.2f}")
    # Calculate gain needed
    gain_db = target_level - current_db
    # Apply gain
    log_callback(f"RMS normalization: {current_db:.2f}dBFS -> {target_level:.2f}dBFS (gain: {gain_db:.2f}dB)")
    return audio_track + gain_db

def normalize_lufs(audio_track, target_lufs=-16.0, log_callback=print):
    """Normalize audio to target LUFS level"""
    try:
        import pyloudnorm as pyln
        
        # Convert audio to numpy array and scale to -1.0 ... 1.0 for pyloudnorm
        samples = np.array(audio_track.get_array_of_samples()).astype(np.float32)
        max_possible = float(2 ** (8 * audio_track.sample_width - 1))
        samples /= max_possible
        if audio_track.channels == 2:
            samples = samples.reshape((-1, 2))
        # Create meter and measure loudness
        meter = pyln.Meter(audio_track.frame_rate)
        loudness = meter.integrated_loudness(samples)
        
        # Calculate gain needed
        gain_db = target_lufs - loudness
        
        # Apply gain
        log_callback(f"LUFS normalization: {loudness:.2f} LUFS -> {target_lufs:.2f} LUFS (gain: {gain_db:.2f}dB)")
        return audio_track + gain_db
        
    except ImportError:
        log_callback("pyloudnorm not available, falling back to RMS normalization")
        return normalize_rms(audio_track, target_level=target_lufs, log_callback=log_callback)