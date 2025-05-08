import numpy as np
from pydub import AudioSegment

def detect_silence_efficiently(audio_segment, min_silence_len=1000, silence_thresh=-50, seek_step=10):
    """Detect silence in audio clip efficiently with better error handling and debug logging"""
    print("[DEBUG][Segmentation] detect_silence_efficiently CALLED")
    try:
        # Prevent common errors by checking the audio segment
        if not isinstance(audio_segment, AudioSegment):
            print(f"Error: Expected AudioSegment, got {type(audio_segment)}")
            return [(0, 1)]
        # Make sure audio segment has data
        if len(audio_segment) == 0:
            print("Warning: Audio segment is empty")
            return [(0, 1)]
        # --- DEBUG: Log segmentation parameters ---
        msg1 = f"[DEBUG][Segmentation] silence_thresh={silence_thresh}, min_silence_len={min_silence_len}, seek_step={seek_step}"
        msg2 = f"[DEBUG][Segmentation] audio length: {len(audio_segment)} ms, dBFS: {getattr(audio_segment, 'dBFS', 'N/A')}"
        print(msg1)
        print(msg2)

        # Try to use direct non-silent detection from pydub
        from pydub.silence import detect_nonsilent
        
        # Get audio's dBFS level for threshold calculation
        dbfs = audio_segment.dBFS if hasattr(audio_segment, 'dBFS') else -20
        
        # Calculate absolute threshold from relative threshold
        abs_threshold = dbfs + (silence_thresh if silence_thresh < 0 else -silence_thresh)
        
        # Use detect_nonsilent directly
        segments = detect_nonsilent(
            audio_segment, 
            min_silence_len=min_silence_len,
            silence_thresh=abs_threshold,
            seek_step=seek_step
        )
        msg3 = f"[DEBUG][Segmentation] Initial detected segments: {len(segments)}"
        print(msg3)
        for idx, (start, end) in enumerate(segments):
            seg_msg = f"[DEBUG][Segmentation] Segment {idx+1}: {start} ms - {end} ms (duration: {end-start} ms)"
            print(seg_msg)
        if not segments:
            print("[DEBUG][Segmentation] No segments detected; returning full audio.")
            return [(0, len(audio_segment))]
        return segments
        
    except Exception as e:
        # If any error occurs, log it and return the entire audio as one segment
        import traceback
        print(f"Error in detect_silence_efficiently: {str(e)}")
        print(traceback.format_exc())
        print("[DEBUG][Segmentation] Exception encountered; returning full audio as one segment.")
        # Return whole audio as one segment
        if isinstance(audio_segment, AudioSegment):
            return [(0, len(audio_segment))]
        else:
            return [(0, 1)]  # Minimal segment for invalid audio