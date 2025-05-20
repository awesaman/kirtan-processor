import numpy as np
from pydub import AudioSegment
import sys
from audio.normalization import normalize_lufs

def debug_lufs(input_path):
    audio = AudioSegment.from_file(input_path)
    print(f"Channels: {audio.channels}, Frame rate: {audio.frame_rate}, Duration: {len(audio)/1000:.2f}s, RMS: {audio.rms}, dBFS: {audio.dBFS}")
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    print(f"Samples shape: {samples.shape}, Min: {samples.min()}, Max: {samples.max()}, Mean: {samples.mean()}, Std: {samples.std()}")
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(audio.frame_rate)
        loudness = meter.integrated_loudness(samples)
        print(f"LUFS loudness: {loudness}")
    except Exception as e:
        print(f"LUFS calculation failed: {e}")
    normed = normalize_lufs(audio, target_lufs=-14.0)
    print(f"After normalization: dBFS: {normed.dBFS}, RMS: {normed.rms}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python normalization_debug.py <audiofile>")
    else:
        debug_lufs(sys.argv[1])
