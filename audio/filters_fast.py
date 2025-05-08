import numpy as np
from pydub import AudioSegment

def _audiosegment_to_np(audio_segment):
    """Convert pydub AudioSegment to numpy array (mono or stereo) and return, along with sample rate and sample width."""
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2))
    return samples.astype(np.float32), audio_segment.frame_rate, audio_segment.sample_width

def _np_to_audiosegment(samples, frame_rate, sample_width, channels):
    """Convert numpy array back to AudioSegment."""
    # Ensure correct shape for mono/stereo
    if channels == 2 and samples.ndim == 1:
        samples = samples.reshape((-1, 2))
    # Clip to int range
    max_val = float(2 ** (8 * sample_width - 1) - 1)
    min_val = -max_val - 1
    samples = np.clip(samples, min_val, max_val)
    samples = samples.astype(np.int16 if sample_width == 2 else np.int32)
    return AudioSegment(
        samples.tobytes(),
        frame_rate=frame_rate,
        sample_width=sample_width,
        channels=channels
    )

def fast_high_pass(audio_segment, cutoff=200, order=4):
    from scipy.signal import butter, sosfilt
    samples, sr, sample_width = _audiosegment_to_np(audio_segment)
    sos = butter(order, cutoff, btype='highpass', fs=sr, output='sos')
    filtered = sosfilt(sos, samples, axis=0)
    return _np_to_audiosegment(filtered, sr, sample_width, audio_segment.channels)

def fast_low_pass(audio_segment, cutoff=8000, order=4):
    from scipy.signal import butter, sosfilt
    samples, sr, sample_width = _audiosegment_to_np(audio_segment)
    sos = butter(order, cutoff, btype='lowpass', fs=sr, output='sos')
    filtered = sosfilt(sos, samples, axis=0)
    return _np_to_audiosegment(filtered, sr, sample_width, audio_segment.channels)
