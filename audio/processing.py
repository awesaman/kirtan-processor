# filepath: c:\git\kirtan-processor\audio\processing.py
import numpy as np
import gc
from pydub import AudioSegment

from audio.normalization import normalize_peak, normalize_rms, normalize_lufs
from audio.dynamics import apply_compressor, apply_limiter
from audio.filters import apply_low_pass, apply_high_pass

from audio.processing_logplan import PROCESSING_STEPS

def process_audio_efficiently(audio_track, profile, log_callback=print, detailed=False, log_mode="detailed", skip_normalization=False):
    """Apply all audio processing steps defined in profile, with standard log showing step checklist."""
    import time
    steps_applied = []
    debug_mode = profile.get('debug_mode', False) or profile.get('settings', {}).get('debug_mode', False)
    step_times = {}
    def debug_log(msg):
        if debug_mode:
            log_callback(f"[DEBUG] {msg}", detailed=True)
    # Only build the checklist for standard log
    if log_mode == "detailed" and not detailed:
        step_status = {step[1]: False for step in PROCESSING_STEPS}
        step_lines = [f"[PLAN] Processing steps:" + " " + ", ".join([s[0] for s in PROCESSING_STEPS])]
        log_callback(step_lines[0])
    # --- Normalization ---
    t0 = time.time()
    normalize_settings = profile.get('normalize', {"enabled": True})
    normalization_applied = False
    if not skip_normalization and normalize_settings.get("enabled", True):
        method = normalize_settings.get("method", "peak")
        target_level = normalize_settings.get("target_level", -1.0)
        pre_norm_dbfs = audio_track.dBFS if hasattr(audio_track, 'dBFS') else None
        debug_log("Normalization started")
        if method == "peak":
            audio_track = normalize_peak(audio_track, target_level, log_callback=log_callback)
            normalization_applied = True
            steps_applied.append("normalize")
        elif method == "rms":
            audio_track = normalize_rms(audio_track, target_level, log_callback=log_callback)
            normalization_applied = True
            steps_applied.append("normalize")
        elif method == "lufs":
            audio_track = normalize_lufs(audio_track, target_level, log_callback=log_callback)
            normalization_applied = True
            steps_applied.append("normalize")
        if normalization_applied:
            post_norm_dbfs = audio_track.dBFS if hasattr(audio_track, 'dBFS') else None
            log_callback(f"[DEBUG] Normalization ({method}) target: {target_level}, Pre: {pre_norm_dbfs:.2f} dBFS, Post: {post_norm_dbfs:.2f} dBFS", detailed=True)
        debug_log(f"Normalization finished in {1000*(time.time()-t0):.1f} ms")
    # --- Dynamics ---
    t1 = time.time()
    dynamic_settings = profile.get('dynamic_processing', {"enabled": False})
    if dynamic_settings.get("enabled", False):
        comp_settings = dynamic_settings.get("compressor", {})
        limit_settings = dynamic_settings.get("limiter", {})
        pre_dyn_dbfs = audio_track.dBFS if hasattr(audio_track, 'dBFS') else None
        debug_log("Dynamics started")
        # Compressor
        if comp_settings.get("enabled", False):
            audio_track = apply_compressor(audio_track, 
                                          threshold=comp_settings.get("threshold", -18.0),
                                          ratio=comp_settings.get("ratio", 2.5),
                                          attack=comp_settings.get("attack", 20.0),
                                          release=comp_settings.get("release", 250.0),
                                          log_callback=log_callback)
            steps_applied.append("compressor")
            log_callback(f"[DEBUG] Compressor applied: threshold={comp_settings.get('threshold', -18.0)}, ratio={comp_settings.get('ratio', 2.5)}, attack={comp_settings.get('attack', 20.0)}, release={comp_settings.get('release', 250.0)}", detailed=True)
        # Limiter
        if limit_settings.get("enabled", False):
            audio_track = apply_limiter(audio_track, 
                                       threshold=limit_settings.get("threshold", -1.0),
                                       release=limit_settings.get("release", 50.0),
                                       log_callback=log_callback)
            steps_applied.append("limiter")
            log_callback(f"[DEBUG] Limiter applied: threshold={limit_settings.get('threshold', -1.0)}, release={limit_settings.get('release', 50.0)}", detailed=True)
        post_dyn_dbfs = audio_track.dBFS if hasattr(audio_track, 'dBFS') else None
        log_callback(f"[DEBUG] Dynamics: Pre: {pre_dyn_dbfs:.2f} dBFS, Post: {post_dyn_dbfs:.2f} dBFS", detailed=True)
        debug_log(f"Dynamics finished in {1000*(time.time()-t1):.1f} ms")
    # --- Gain ---
    t2 = time.time()
    gain_db = profile.get('gain', 0)
    if gain_db != 0:
        pre_gain_dbfs = audio_track.dBFS if hasattr(audio_track, 'dBFS') else None
        debug_log("Gain started")
        audio_track = audio_track + gain_db
        steps_applied.append("gain")
        post_gain_dbfs = audio_track.dBFS if hasattr(audio_track, 'dBFS') else None
        log_callback(f"[DEBUG] Gain applied: {gain_db} dB, Pre: {pre_gain_dbfs:.2f} dBFS, Post: {post_gain_dbfs:.2f} dBFS", detailed=True)
        debug_log(f"Gain finished in {1000*(time.time()-t2):.1f} ms")
    # --- EQ/Filters ---
    t3 = time.time()
    use_eq = profile.get('use_eq', False)
    if use_eq:
        eq_settings = profile.get('eq', {})
        hp_enabled = eq_settings.get('high_pass', None) is not None
        lp_enabled = eq_settings.get('low_pass', None) is not None
        hp_freq = eq_settings.get('high_pass', 100)
        lp_freq = eq_settings.get('low_pass', 10000)
        att_db = eq_settings.get('att_db', -24.0)
        debug_log("EQ/Filters started")
        # High-pass
        if hp_enabled:
            audio_track = apply_high_pass(audio_track, hp_freq)
            steps_applied.append("high_pass")
            log_callback(f"[DEBUG] High-pass filter applied: cutoff={hp_freq} Hz", detailed=True)
        # Low-pass
        if lp_enabled:
            audio_track = apply_low_pass(audio_track, lp_freq)
            steps_applied.append("low_pass")
            log_callback(f"[DEBUG] Low-pass filter applied: cutoff={lp_freq} Hz", detailed=True)
        # Attenuate
        if att_db > -24.0:
            audio_track = audio_track + att_db
            steps_applied.append("eq_attenuate")
            log_callback(f"[DEBUG] EQ attenuation applied: {att_db} dB", detailed=True)
        debug_log(f"EQ/Filters finished in {1000*(time.time()-t3):.1f} ms")
    # Final debug
    final_dbfs = audio_track.dBFS if hasattr(audio_track, 'dBFS') else None
    log_callback(f"[DEBUG] Final processed audio dBFS: {final_dbfs:.2f}", detailed=True)
    # Emit checklist to standard log after all steps
    if log_mode == "detailed" and not detailed:
        checked = []
        for step_name, step_key in PROCESSING_STEPS:
            if step_key in steps_applied:
                checked.append(f"[x] {step_name}")
            else:
                checked.append(f"[ ] {step_name}")
        log_callback("[CHECKLIST] " + ", ".join(checked))
    elif log_mode == "simple":
        log_callback("[OK] Processing complete.")
    return audio_track


def process_audio_in_chunks(audio_track, profile, chunk_size_minutes=10, log_callback=print, batch_normalization=True, log_mode="detailed"):
    """Process audio in chunks to reduce memory usage"""
    # Convert minutes to milliseconds
    chunk_size_ms = chunk_size_minutes * 60 * 1000
    
    # If audio is shorter than chunk size, process normally
    if len(audio_track) <= chunk_size_ms:
        return process_audio_efficiently(audio_track, profile, log_callback)
    
    # Otherwise, process in chunks
    result_chunks = []
    total_length = len(audio_track)
    chunk_count = (total_length + chunk_size_ms - 1) // chunk_size_ms
    
    if log_mode == "detailed":
        log_callback(f"Processing large audio in {chunk_count} chunks")
    elif log_mode == "simple":
        log_callback(f"Processing {chunk_count} chunks...")
    
    # Create a copy of the profile to modify for consistent processing
    modified_profile = profile.copy() if isinstance(profile, dict) else {}
    
    # --- Batch normalization: calculate gain for the whole file and apply to each chunk ---
    normalization_gain_db = 0.0
    normalize_settings = profile.get('normalize', {"enabled": True})
    normalization_applied = False
    norm_method = normalize_settings.get("method", "peak").lower()
    norm_enabled = normalize_settings.get("enabled", True)
    target_level = normalize_settings.get("target_level", -1.0)
    
    # Step 1: Analysis Phase - Scan the entire file to get measurements
    if batch_normalization and norm_enabled:
        # Store audio metrics across all chunks for consistent processing
        total_rms = 0
        max_peak = 0
        total_samples = 0
        lufs_measurements = []
        
        # First pass - gather metrics from all chunks (without modifying audio)
        log_callback("[BATCH NORM] Analyzing entire audio file for consistent normalization...")
        
        for i in range(chunk_count):
            start = i * chunk_size_ms
            end = min((i+1) * chunk_size_ms, total_length)
            chunk = audio_track[start:end]
            samples = np.array(chunk.get_array_of_samples()).astype(np.float32)
            
            # Calculate RMS
            if len(samples) > 0:
                chunk_rms = chunk.rms
                total_rms += chunk_rms * len(samples)
                total_samples += len(samples)
            
            # Calculate peak
            if len(samples) > 0:
                max_possible = float(2 ** (chunk.sample_width * 8 - 1))
                chunk_peak = np.max(np.abs(samples)) / max_possible if max_possible > 0 else 0
                max_peak = max(max_peak, chunk_peak)
            
            # For LUFS, store the samples for later processing
            if norm_method == "lufs":
                try:
                    import pyloudnorm as pyln
                    normalized_samples = samples.copy().astype(np.float32)
                    max_possible = float(2 ** (8 * chunk.sample_width - 1))
                    normalized_samples /= max_possible
                    if chunk.channels == 2:
                        normalized_samples = normalized_samples.reshape((-1, 2))
                    lufs_measurements.append((normalized_samples, len(samples)))
                except ImportError:
                    # Will fall back to RMS if pyloudnorm is not available
                    pass
        
        # Step 2: Calculate the appropriate normalization gain
        if norm_method == "peak":
            # Use max peak across all chunks
            if max_peak > 0:
                peak_db = 20 * np.log10(max_peak)
                normalization_gain_db = target_level - peak_db
                log_callback(f"[BATCH NORM] Peak normalization: {peak_db:.2f}dB -> {target_level:.2f}dB (gain: {normalization_gain_db:.2f}dB)")
            else:
                log_callback("[BATCH NORM] Peak normalization: No valid peaks found")
                
        elif norm_method == "rms":
            # Calculate average RMS across all chunks
            if total_samples > 0:
                avg_rms = total_rms / total_samples
                sample_width = audio_track.sample_width
                ref = float(2 ** (8 * sample_width - 1))
                current_db = 20 * np.log10(avg_rms / ref) if avg_rms > 0 else -80
                normalization_gain_db = target_level - current_db
                log_callback(f"[BATCH NORM] RMS normalization: {current_db:.2f}dBFS -> {target_level:.2f}dBFS (gain: {normalization_gain_db:.2f}dB)")
            else:
                log_callback("[BATCH NORM] RMS normalization: No valid samples found")
                
        elif norm_method == "lufs":
            try:
                import pyloudnorm as pyln
                
                # Perform integrated LUFS measurement across all chunks
                if lufs_measurements:
                    meter = pyln.Meter(audio_track.frame_rate)
                    
                    # For more accurate LUFS measurement, we'd need a single continuous array
                    # This is an approximation using weighted average of chunk measurements
                    loudness_values = []
                    weights = []
                    
                    for samples, sample_count in lufs_measurements:
                        try:
                            chunk_loudness = meter.integrated_loudness(samples)
                            if not np.isnan(chunk_loudness) and chunk_loudness > -100:  # Filter invalid measurements
                                loudness_values.append(chunk_loudness)
                                weights.append(sample_count)
                        except Exception as e:
                            log_callback(f"[DEBUG] LUFS measurement error on chunk: {str(e)}")
                    
                    if loudness_values:
                        # Calculate weighted average loudness
                        weights = np.array(weights)
                        loudness_values = np.array(loudness_values)
                        loudness = np.average(loudness_values, weights=weights)
                        
                        normalization_gain_db = target_level - loudness
                        log_callback(f"[BATCH NORM] LUFS normalization: {loudness:.2f} LUFS -> {target_level:.2f} LUFS (gain: {normalization_gain_db:.2f}dB)")
                    else:
                        log_callback("[BATCH NORM] LUFS normalization: Could not calculate valid LUFS values")
                        
            except ImportError:
                log_callback("[BATCH NORM] pyloudnorm not available, falling back to RMS normalization")
                # Fall back to the RMS normalization already calculated
        
        # Step 3: Modify the profile to apply consistent processing
        if abs(normalization_gain_db) > 1e-4:  # Only if meaningful gain adjustment needed
            # Disable per-chunk normalization as we'll apply the global adjustment
            modified_profile['normalize'] = {'enabled': False}
            
            # Add the normalization gain to the profile's gain setting
            current_gain = modified_profile.get('gain', 0.0)
            modified_profile['gain'] = current_gain + normalization_gain_db
            
            log_callback(f"[BATCH NORM] Applying consistent gain of {normalization_gain_db:.2f}dB to all chunks")
            normalization_applied = True
    
    # Process each chunk with the modified profile settings
    for i in range(chunk_count):
        start = i * chunk_size_ms
        end = min((i+1) * chunk_size_ms, total_length)
        
        # Log processing information
        if log_mode == "detailed":
            log_callback(f"Processing chunk {i+1}/{chunk_count} ({start/1000:.1f}s to {end/1000:.1f}s)")
            settings_lines = [
                f"[DEBUG] === Processing Settings for Chunk {i+1}/{chunk_count} ===",
                f"  Profile: {modified_profile.get('name', '<unnamed>')}",
                f"  Normalization: {modified_profile.get('normalize', {})}",
                f"  Gain: {modified_profile.get('gain', 0)} dB",
                f"  Compressor: {modified_profile.get('dynamic_processing', {}).get('compressor', {})}",
                f"  Limiter: {modified_profile.get('dynamic_processing', {}).get('limiter', {})}",
                f"  EQ: {modified_profile.get('eq', {})}",
                f"  Use EQ: {modified_profile.get('use_eq', False)}",
                f"  High-pass: {modified_profile.get('eq', {}).get('high_pass', None)} Hz",
                f"  Low-pass: {modified_profile.get('eq', {}).get('low_pass', None)} Hz",
                f"  Attenuation: {modified_profile.get('eq', {}).get('att_db', None)} dB",
                f"  Debug Mode: {modified_profile.get('debug_mode', False)}",
                f"  Other keys: {[k for k in modified_profile.keys() if k not in ['normalize','gain','dynamic_processing','eq','use_eq','debug_mode','name']]}",
                f"[DEBUG] ========================================="
            ]
            log_callback("\n" + "\n".join(settings_lines))
        elif log_mode == "simple":
            log_callback(f"Chunk {i+1}/{chunk_count} ({start/1000:.1f}s to {end/1000:.1f}s)")
        
        # Extract and process this chunk
        chunk = audio_track[start:end]
        processed_chunk = process_audio_efficiently(chunk, modified_profile, log_callback, log_mode=log_mode)
        result_chunks.append(processed_chunk)
        
        # Clean up
        del chunk
        gc.collect(generation=0)
    
    # Concatenate chunks
    if not result_chunks:
        return audio_track
        
    if log_mode == "detailed":
        log_callback(f"Concatenating {len(result_chunks)} processed chunks")
    elif log_mode == "simple":
        log_callback(f"All chunks processed and combined.")
    result = result_chunks[0]
    for i in range(1, len(result_chunks)):
        result += result_chunks[i]
        result_chunks[i] = None
    
    result_chunks.clear()
    gc.collect()
    
    return result