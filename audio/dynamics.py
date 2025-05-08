import numpy as np
from pydub import AudioSegment

def apply_compressor(audio_track, threshold=-18.0, ratio=2.0, attack=20.0, release=250.0, log_callback=print):
    """Apply a simple compressor to the audio track"""
    # Convert to numpy array
    samples = np.array(audio_track.get_array_of_samples()).astype(np.float32)
    sample_width = audio_track.sample_width
    max_value = float(2 ** (8 * sample_width - 1))
    
    # Calculate threshold in linear scale
    threshold_linear = 10 ** (threshold / 20.0)
    
    # Apply compression
    log_callback(f"Applying compression: threshold={threshold}dB, ratio={ratio}:1")
    
    # Calculate gain reduction
    # This is a simplified compressor - a full implementation would use attack/release
    abs_samples = np.abs(samples) / max_value
    gain_reduction = np.ones_like(abs_samples, dtype=float)
    
    # Apply gain reduction where samples exceed threshold
    mask = abs_samples > threshold_linear
    gain_reduction[mask] = threshold_linear + (abs_samples[mask] - threshold_linear) / ratio
    gain_reduction[mask] /= abs_samples[mask]
    
    # Apply gain reduction to samples
    compressed_samples = samples * gain_reduction
    
    # Create new AudioSegment
    result = audio_track._spawn(compressed_samples.tobytes())
    return result

def apply_limiter(audio_track, threshold=-1.0, release=50.0, log_callback=print):
    """Apply a limiter to prevent clipping"""
    # This is a simplified limiter - just hard limits at threshold
    log_callback(f"Applying limiter: threshold={threshold}dB")
    
    # Get samples
    samples = np.array(audio_track.get_array_of_samples()).astype(np.float32)
    sample_width = audio_track.sample_width
    max_value = float(2 ** (8 * sample_width - 1))
    
    # Calculate threshold in linear scale
    threshold_linear = 10 ** (threshold / 20.0) * max_value
    
    # Limit to threshold
    np.clip(samples, -threshold_linear, threshold_linear, out=samples)
    
    # Create new AudioSegment
    result = audio_track._spawn(samples.tobytes())
    return result