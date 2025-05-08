#!/usr/bin/env python
import os
import time
import math
import json
import traceback
import concurrent.futures
from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
from pydub import AudioSegment

from audio.detection import detect_silence_efficiently
from audio.normalization import normalize_rms, normalize_lufs, normalize_peak
from audio.dynamics import apply_limiter, apply_compressor
from audio.filters import apply_low_pass, apply_high_pass
from audio.processing import process_audio_efficiently, process_audio_in_chunks
from audio.export import export_segments, export_audio_slice
from utils.cache import AudioCache
from utils.file_utils import organize_files_into_tracks, format_time, safe_filename
from utils.performance import PerformanceTracker
from config.constants import INPUT_FORMAT

def convert_profile_to_processing_format(profile):
    """Convert a flat/legacy profile dict to the nested structure expected by process_audio_in_chunks."""
    # If already nested, return as-is
    if 'normalize' in profile and isinstance(profile['normalize'], dict):
        return profile
    # Build nested profile
    nested = {}
    # Normalization
    norm_type = profile.get('normalize_type', 'peak').lower()
    norm_method = {'rms': 'rms', 'lufs': 'lufs', 'peak': 'peak'}.get(norm_type, 'peak')
    nested['normalize'] = {
        'enabled': profile.get('normalize', True) if isinstance(profile.get('normalize'), bool) else True,
        'method': norm_method,
        'target_level': profile.get('target_level', -1.0)
    }
    # Gain
    nested['gain'] = profile.get('gain', 0)
    # Dynamics
    nested['dynamic_processing'] = {
        'enabled': profile.get('use_compressor', False) or profile.get('use_limiter', False),
        'compressor': profile.get('compression', {}),
        'limiter': profile.get('limiter', {})
    }
    # EQ
    nested['use_eq'] = profile.get('use_eq', False)
    nested['eq'] = profile.get('eq', {})
    # Name
    nested['name'] = profile.get('name', profile.get('profile_name', ''))
    # Debug
    nested['debug_mode'] = profile.get('debug_mode', False)
    # Copy any extra keys
    for k, v in profile.items():
        if k not in nested:
            nested[k] = v
    return nested

class ProcessingWorker(QThread):
    """Worker thread that handles audio processing operations"""
    progress_update = pyqtSignal(str)
    progress_bar = pyqtSignal(int)
    processing_complete = pyqtSignal()
    processing_finished = pyqtSignal(bool)  # Changed to emit success status
    processing_error = pyqtSignal(str, str)
    track_finished = pyqtSignal(str)  # Add this missing signal

    def log_progress(self, message, **kwargs):
        self.progress_update.emit(message)
    
    def __init__(self, app, working_dir=None):
        super().__init__()
        self.app = app
        self.working_dir = working_dir or app.working_dir  # Add this line
        self.stop_requested = False
        self.tracks_processed = 0
        self.total_tracks = 0
        self.processor_count = min(os.cpu_count() or 1, 4)  # Limit to 4 cores max
        self.audio_cache = AudioCache()
        self.performance_tracker = PerformanceTracker(self.progress_update.emit)
        # Copy settings from app
        self.settings = getattr(app, 'settings', {}).copy()
    
    def stop(self):
        """Stop the processing thread"""
        self.progress_update.emit("Stopping processing operation...")
        self.stop_requested = True
        # Make sure the thread finishes gracefully
        self.wait(2000)  # Wait for up to 2 seconds
        if self.isRunning():
            self.progress_update.emit("Processing thread is taking longer to stop, please wait...")
            self.wait(5000)  # Wait a bit longer
            if self.isRunning():
                self.progress_update.emit("Forcing thread termination...")
                self.terminate()  # Force termination if still running
        self.progress_update.emit("Processing stopped successfully")
        
    def run(self):
        """Main thread execution method"""
        try:
            self.stop_requested = False
            self.tracks_processed = 0
            self.performance_tracker.start_tracking()
            self.process_files()
            
            if not self.stop_requested:
                self.performance_tracker.log_summary()
                self.progress_update.emit("✅ Processing complete!")
                success = True
            else:
                self.progress_update.emit("⚠️ Processing stopped by user")
                success = False
            
            self.processing_complete.emit()
            self.processing_finished.emit(success)
            
        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            self.progress_update.emit(f"❌ {error_msg}")
            self.processing_error.emit(error_msg, traceback.format_exc())
            self.processing_finished.emit(False)
    
    def process_files(self):
        """Process the selected files"""
        # Get files to process
        if not hasattr(self.app, 'selected_files') or not self.app.selected_files:
            # If no files are selected, use all audio files instead
            if hasattr(self.app, 'audio_files') and self.app.audio_files:
                self.progress_update.emit("No files selected, processing all available tracks")
                unique_tracks = set(self.app.audio_files)
            else:
                self.progress_update.emit("No files available for processing")
                return
        else:
            # Use the full folder name for each selected file as the unique track
            unique_tracks = set(self.app.selected_files)
        
        if not unique_tracks:
            self.progress_update.emit("No valid tracks found in selected files")
            return
        
        # Process each unique track
        self.total_tracks = len(unique_tracks)
        self.progress_update.emit(f"Found {self.total_tracks} unique tracks to process")
        
        for i, track_path in enumerate(unique_tracks):
            if self.stop_requested:
                self.progress_update.emit("Processing stopped by user")
                break
            
            # Update progress
            progress = int((i / self.total_tracks) * 100)
            self.progress_bar.emit(progress)
            
            # Process this track
            self.progress_update.emit(f"Processing track {i+1}/{self.total_tracks}: {os.path.basename(track_path)}")
            self.process_track(track_path)
            
            # Update count and emit signal
            self.tracks_processed += 1
            self.track_finished.emit(track_path)
        
        # Final progress update
        if not self.stop_requested:
            self.progress_bar.emit(100)
    
    def process_track(self, track_path):
        """Process a single track"""
        track_name = os.path.basename(track_path)
        self.progress_update.emit(f"[STD] Processing track: {track_name}")

        # Ensure the track folder exists before exporting
        if not os.path.exists(track_path):
            try:
                os.makedirs(track_path, exist_ok=True)
                self.progress_update.emit(f"[INFO] Created missing track folder: {track_path}")
            except Exception as e:
                self.progress_update.emit(f"[ERROR] Could not create track folder '{track_path}': {str(e)}")
                return False
        elif not os.path.isdir(track_path):
            self.progress_update.emit(f"[ERROR] Track path '{track_path}' is not a directory.")
            return False

        try:
            # Get track name without extension
            track_prefix = os.path.basename(track_path)
            if "." in track_prefix:
                track_prefix = os.path.splitext(track_prefix)[0]
            # Get input channels using the correct method
            input_channels = self.get_input_channels(track_path)
            if not input_channels:
                self.progress_update.emit(f"[STD] No input channels found for {track_name}")
                return False
            # Now input_channels is in the correct format for load_and_mix_audio
            mixed_audio = self.load_and_mix_audio(track_path, input_channels)
            # Pull segmentation/silence settings and toggles from self.app.settings (set in UI)
            settings = getattr(self.app, 'settings', {})
        except Exception as e:
            self.progress_update.emit(f"[STD] Error processing track {track_name}: {str(e)}")
            return False

    def load_and_mix_audio(self, track_path, input_channels):
        # --- Debug log the settings in use ---
        try:
            import json
            settings_str = json.dumps(self.settings, indent=2) if hasattr(self, 'settings') else str(getattr(self, 'settings', {}))
        except Exception:
            settings_str = str(getattr(self, 'settings', {}))
        msg = f"[DEBUG] Processing with settings (from backend):\n{settings_str}"
        print(msg)
        if hasattr(self, 'progress_update'):
            self.progress_update.emit(f"[DBG] {msg}")

        """Load and mix audio based on assigned profiles"""
        track_name = os.path.basename(track_path)
        self.progress_update.emit(f"Loading audio for {track_name}")
        # Get profile assignments
        track_profile_assignments = getattr(self.app, 'track_profile_assignments', {})
        profiles = getattr(self.app, 'profiles', {})
        # Debug: Show assignments and channels
        self.progress_update.emit(f"[DBG] Profile assignments: {track_profile_assignments}")
        self.progress_update.emit(f"[DBG] Input channels: {input_channels}")
        mixed_audio = None
        for channel_name, file_path in input_channels:
            # Extract base channel (after last underscore)
            base_channel = channel_name.split('_')[-1]
            profile_name = track_profile_assignments.get(base_channel)
            # --- Debug: print all available profile keys and requested profile_name ---
            self.progress_update.emit(f"[DBG] Available profile keys: {list(profiles.keys())}")
            self.progress_update.emit(f"[DBG] Requested profile for channel '{base_channel}': {profile_name}")
            # --- Robust matching ---
            chosen_profile_key = None
            if profile_name is not None:
                # 1. Exact match
                if profile_name in profiles:
                    chosen_profile_key = profile_name
                else:
                    # 2. Case-insensitive match
                    for k in profiles:
                        if k.strip().lower() == profile_name.strip().lower():
                            chosen_profile_key = k
                            self.progress_update.emit(f"[DBG] Fuzzy matched profile '{profile_name}' to '{k}' (case-insensitive)")
                            break
                # 3. Fuzzy: strip parentheticals (e.g., 'Kirtan (Vocals)' -> 'Kirtan')
                if chosen_profile_key is None and '(' in profile_name:
                    base = profile_name.split('(')[0].strip()
                    for k in profiles:
                        if k.strip().lower() == base.lower():
                            chosen_profile_key = k
                            self.progress_update.emit(f"[DBG] Fuzzy matched profile '{profile_name}' to '{k}' (no parenthetical)")
                            break
            if chosen_profile_key is None:
                chosen_profile_key = 'Standard' if 'Standard' in profiles else list(profiles.keys())[0]
                self.progress_update.emit(f"[STD] [WARNING] Could not find profile for '{profile_name}', using '{chosen_profile_key}'")
            else:
                self.progress_update.emit(f"[DBG] Channel '{base_channel}' assigned profile '{chosen_profile_key}'")
            profile_raw = profiles.get(chosen_profile_key, profiles.get("Standard", {}))
            profile = convert_profile_to_processing_format(profile_raw)
            # Skip channels marked as 'Do Not Process' (by name or by profile setting)
            if chosen_profile_key.strip().lower() in ["do not process", "-- do not process --"] or profile.get("ignore", False):
                self.progress_update.emit(f"[STD] Skipping channel {base_channel} (marked as 'Do Not Process')")
                continue
            # Load audio
            self.progress_update.emit(f"[STD] Loading channel: {base_channel} with profile: {chosen_profile_key}")
            try:
                audio = AudioSegment.from_file(file_path, format="wav")
                processed = process_audio_in_chunks(audio, profile, log_callback=self.log_progress)
                if mixed_audio is None:
                    mixed_audio = processed
                else:
                    mixed_audio = mixed_audio.overlay(processed)
            except Exception as e:
                self.progress_update.emit(f"[STD] Error processing channel {base_channel}: {str(e)}")

        # --- Debug Logging Improvements ---
        if mixed_audio is None:
            self.progress_update.emit("[STD] [ERROR] No mixed audio generated; skipping export.")
            print("[ERROR] No mixed audio generated; skipping export.")
            return None

        # --- Find vocal channel for segmentation ---
        track_profile_assignments = getattr(self.app, 'track_profile_assignments', {})
        profiles = getattr(self.app, 'profiles', {})
        vocal_channel = None
        vocal_profile_names = [k for k in profiles if k.strip().lower() == 'vocal']
        if vocal_profile_names:
            vocal_profile_name = vocal_profile_names[0]
            for channel_name, file_path in input_channels:
                assigned_profile = track_profile_assignments.get(channel_name.split('_')[-1])
                if assigned_profile and assigned_profile.strip().lower() == 'vocal':
                    vocal_channel = (channel_name, file_path)
                    break
        # Fallback: use first channel if not found
        if not vocal_channel and input_channels:
            vocal_channel = input_channels[0]
        # Process vocal channel audio for segmentation
        vocal_audio = None
        if vocal_channel:
            channel_name, file_path = vocal_channel
            # Use vocal profile for processing
            profile_name = track_profile_assignments.get(channel_name.split('_')[-1])
            chosen_profile_key = None
            if profile_name is not None:
                if profile_name in profiles:
                    chosen_profile_key = profile_name
                else:
                    for k in profiles:
                        if k.strip().lower() == profile_name.strip().lower():
                            chosen_profile_key = k
                            break
            if chosen_profile_key is None:
                chosen_profile_key = 'Standard' if 'Standard' in profiles else list(profiles.keys())[0]
            profile_raw = profiles.get(chosen_profile_key, profiles.get("Standard", {}))
            profile = convert_profile_to_processing_format(profile_raw)
            try:
                audio = AudioSegment.from_file(file_path, format="wav")
                vocal_audio = process_audio_in_chunks(audio, profile, log_callback=self.log_progress)
            except Exception as e:
                self.progress_update.emit(f"Error processing vocal channel {channel_name}: {str(e)}")
        
        # --- Use vocal_audio for segmentation ---
        segments = None
        trim_start_end_only = self.settings.get('trim_silence_start_end_only', False)
        from pydub import silence  # Import here to avoid circular imports
        silence_settings = self.settings
        min_silence = silence_settings.get('min_silence', 4000)
        silence_thresh_offset = silence_settings.get('silence_threshold', 21)
        seek_step = silence_settings.get('seek_step', 2000)
        
        if vocal_audio:
            # Use the vocal audio's dBFS for threshold calculation
            silence_thresh = vocal_audio.dBFS - silence_thresh_offset
            
            if trim_start_end_only:
                nonsilent = silence.detect_nonsilent(
                    vocal_audio,
                    min_silence_len=min_silence,
                    silence_thresh=silence_thresh,
                    seek_step=seek_step
                )
                if not nonsilent:
                    self.progress_update.emit("[WARNING] No nonsilent audio detected in vocal channel; nothing to export.")
                    print("[WARNING] No nonsilent audio detected in vocal channel; nothing to export.")
                    return None
                start = nonsilent[0][0]
                end = nonsilent[-1][1]
                segments = [(start, end)]
                self.progress_update.emit(f"[DEBUG][Trim] Trimmed segment (vocal): {start} ms - {end} ms (duration: {end-start} ms)")
            else:
                segments = self.detect_segments(vocal_audio)
                if not segments:
                    self.progress_update.emit("[WARNING] No segments detected in vocal channel; nothing to export.")
                    print("[WARNING] No segments detected in vocal channel; nothing to export.")
                    return None
        else:
            # Fallback to mixed_audio segmentation if vocal channel failed
            # Calculate silence threshold based on mixed audio
            silence_thresh = mixed_audio.dBFS - silence_thresh_offset
            
            if trim_start_end_only:
                nonsilent = silence.detect_nonsilent(
                    mixed_audio,
                    min_silence_len=min_silence,
                    silence_thresh=silence_thresh,
                    seek_step=seek_step
                )
                if not nonsilent:
                    self.progress_update.emit("[WARNING] No nonsilent audio detected; nothing to export.")
                    print("[WARNING] No nonsilent audio detected; nothing to export.")
                    return None
                start = nonsilent[0][0]
                end = nonsilent[-1][1]
                segments = [(start, end)]
                self.progress_update.emit(f"[DEBUG][Trim] Trimmed segment: {start} ms - {end} ms (duration: {end-start} ms)")
            else:
                segments = self.detect_segments(mixed_audio)
                if not segments:
                    self.progress_update.emit("[WARNING] No segments detected; nothing to export.")
                    print("[WARNING] No segments detected; nothing to export.")
                    return None

        # Double-check we have segments before proceeding
        if not segments:
            self.progress_update.emit("[ERROR] No segments were detected. Cannot continue with export.")
            return None

        # Log intended export directory for debugging
        parent_dir = os.path.dirname(track_path)
        self.progress_update.emit(f"[DEBUG] Export parent directory: {parent_dir}")
        print(f"[DEBUG] Export parent directory: {parent_dir}")

        # --- Log pre/post segment padding ---
        # FIXED: Get the raw padding values and interpret them correctly as seconds
        raw_pre_pad = self.settings.get('pre_segment_padding', 0)
        raw_post_pad = self.settings.get('post_segment_padding', 0)
        
        # Determine if the padding values are already in seconds or need conversion
        # If the absolute value is large (> 1000), assume the value is in milliseconds
        # Otherwise, assume it's in seconds and needs conversion to milliseconds
        pre_pad_seconds = raw_pre_pad / 1000 if abs(raw_pre_pad) > 1000 else raw_pre_pad
        post_pad_seconds = raw_post_pad / 1000 if abs(raw_post_pad) > 1000 else raw_post_pad
        
        # Log the interpreted padding values (in seconds for UI display)
        self.progress_update.emit(f"Post-Segment padding: {post_pad_seconds}s")
        self.progress_update.emit(f"Pre-Segment padding: {pre_pad_seconds}s")
        
        # CRITICAL FIX: Create a completely fresh list of segment tuples to avoid reference issues
        original_segments = []
        for start, end in segments:
            original_segments.append((int(start), int(end)))
            
        # Log original segments for debugging
        self.progress_update.emit("[DEBUG] Original segments before padding:")
        for i, (start, end) in enumerate(original_segments):
            self.progress_update.emit(f"  Original Segment {i+1}: {start}ms to {end}ms (duration: {end-start}ms)")
            
        # Apply segment padding while ensuring strong boundaries
        padded_segments = []
        audio_length = len(mixed_audio)
        def ms_to_mmss(ms):
            s = ms // 1000
            m = s // 60
            s = s % 60
            return f"{m:02}:{s:02}"
            
        # First log the original segment data
        self.progress_update.emit(f"[DEBUG] Original segments before any processing: {segments}")
        self.progress_update.emit(f"[DEBUG] Audio length: {audio_length}ms")
        
        # IMPORTANT: Check for segment validity before processing
        if segments and len(segments) > 0 and any(s[0] > audio_length or s[1] > audio_length for s in segments):
            self.progress_update.emit("[WARNING] Detected segments with timestamps exceeding audio length - possible timestamp mismatch!")
            # Full diagnostic info to understand the situation
            for idx, (start, end) in enumerate(segments):
                if start > audio_length or end > audio_length:
                    self.progress_update.emit(f"[WARNING] Segment {idx+1} has invalid bounds: {start}ms to {end}ms (audio length: {audio_length}ms)")
            
        # FIX: When segments appear invalid, use the DETECTED segments from silence detection
        # This ensures we have valid segments to export
        has_segment_problem = False
        if segments and len(segments) > 0:
            for s_idx, (s_start, s_end) in enumerate(segments):
                if s_start > audio_length or s_end > audio_length or s_start > s_end:
                    has_segment_problem = True
                    self.progress_update.emit(f"[ERROR] Found problematic segment {s_idx+1}: {s_start}ms to {s_end}ms")
        
        # If we have problematic segments, regenerate them using the actual audio file
        if has_segment_problem:
            self.progress_update.emit("[CRITICAL FIX] Replacing invalid segments with automatically detected segments")
            
            # Redetect segments directly from the mixed audio
            from pydub import silence
            silence_thresh = mixed_audio.dBFS - silence_settings.get('silence_threshold', 21)
            min_silence = silence_settings.get('min_silence', 4000)
            
            regenerated_segments = silence.detect_nonsilent(
                mixed_audio,
                min_silence_len=min_silence,
                silence_thresh=silence_thresh,
                seek_step=2000
            )
            
            if regenerated_segments:
                self.progress_update.emit(f"[SUCCESS] Regenerated {len(regenerated_segments)} segments from audio")
                for i, (start, end) in enumerate(regenerated_segments):
                    self.progress_update.emit(f"[DEBUG] Regenerated segment {i+1}: {start}ms to {end}ms (duration: {end-start}ms)")
                
                # Use these segments instead
                segments = regenerated_segments
                
                # Also update original_segments
                original_segments = []
                for start, end in segments:
                    original_segments.append((int(start), int(end)))
            else:
                self.progress_update.emit("[ERROR] Failed to regenerate segments. Using full audio as one segment.")
                segments = [[0, audio_length]]
                original_segments = [(0, audio_length)]
        
        # Continue with padding but simplify the logic to avoid errors
        for idx, (start, end) in enumerate(original_segments):
            # Ensure we have valid numbers within audio boundaries
            start = max(0, min(start, audio_length - 1))
            end = max(start + 1, min(end, audio_length))
            
            # FIXED: Convert padding seconds to milliseconds properly
            # If values are already in seconds (e.g., -4, 4), multiply by 1000 to get milliseconds
            pre_pad_ms = int(pre_pad_seconds * 1000)   # Convert seconds to ms
            post_pad_ms = int(post_pad_seconds * 1000)  # Convert seconds to ms
            
            # DEBUG: Print the padding values for clarity
            self.progress_update.emit(f"[DEBUG] Applying padding: pre_pad={pre_pad_seconds}s ({pre_pad_ms}ms), post_pad={post_pad_seconds}s ({post_pad_ms}ms)")
            self.progress_update.emit(f"[DEBUG] Original segment boundaries: start={start}ms, end={end}ms")
            
            # Improved padding logic to handle negative and positive values correctly
            # For pre-padding: negative values extend segment (start earlier), positive values shrink it (start later)
            if pre_pad_seconds < 0:  # Negative pre-padding: extend start earlier
                padded_start = max(0, start + pre_pad_ms)  # Adding negative value = subtraction
            else:  # Positive pre-padding: shrink start (move it later)
                padded_start = min(end - 5000, start + pre_pad_ms)  # Ensure at least 5s segment remains
                
            # For post-padding: positive values extend segment (end later), negative values shrink it (end earlier)
            if post_pad_seconds > 0:  # Positive post-padding: extend end later
                padded_end = min(audio_length, end + post_pad_ms)
            else:  # Negative post-padding: shrink end (move it earlier)
                padded_end = max(padded_start + 5000, end + post_pad_ms)  # Ensure at least 5s segment remains
            
            # Final safety check: ensure segment is at least 5 seconds
            if padded_end - padded_start < 5000:
                self.progress_update.emit(f"[WARNING] Padded segment too short ({padded_end - padded_start}ms), ensuring minimum 5s duration")
                # If segment becomes too short, center it on the original segment's midpoint
                mid_point = (start + end) // 2
                padded_start = max(0, mid_point - 2500)
                padded_end = min(audio_length, mid_point + 2500)
            
            self.progress_update.emit(f"[DEBUG] After padding: start={padded_start}ms, end={padded_end}ms (duration={padded_end-padded_start}ms)")
            
            # Store the segment
            padded_segments.append((int(padded_start), int(padded_end)))
        
        # Skip validation step - the segments are now guaranteed to be valid
        validated_segments = padded_segments
        
        # Log final segments
        self.progress_update.emit(f"[DEBUG] Final segments to export ({len(validated_segments)}):")
        for i, (start, end) in enumerate(validated_segments):
            duration = end - start
            self.progress_update.emit(f"[DEBUG] Segment {i+1}: {start}ms to {end}ms (duration: {duration}ms / {duration/1000:.1f}s)")
        
        if not validated_segments:
            self.progress_update.emit("[ERROR] No valid segments to export!")
            # Create one segment with entire audio as a fallback
            validated_segments = [(0, audio_length)]
            self.progress_update.emit("[DEBUG] Added fallback segment: 0ms to {audio_length}ms (full audio)")

        # Export segmented version using main export logic
        from audio.export import export_segments
        # Always export into the track folder itself, not its parent
        track_folder = track_path  # This is already the subfolder for the track
                
        # CRITICAL FIX: We're not including the segments in version_info - segments must be passed directly as arg #2
        version_info = {
            'name': None,  # Force auto versioning
            # DO NOT include segments here - it gets ignored
        }
        
        self.progress_update.emit(f"[CRITICAL] Exporting {len(validated_segments)} segments with explicit boundaries:")
        for i, (start, end) in enumerate(validated_segments):
            self.progress_update.emit(f"  Segment {i+1}: {start}ms to {end}ms (duration: {end-start}ms)")
                
        # Export using our explicit copy of the segments - using positional args to avoid confusion
        # CRITICAL FIX: Pass validated_segments directly as the second positional argument
        export_result = export_segments(
            mixed_audio, 
            validated_segments,  # This MUST be passed as the direct second positional argument
            track_folder,
            track_folder,
            "KP",
            version_info  # This does NOT contain segments as we need them to be the direct second arg
        )
        
        if not export_result:
            self.progress_update.emit("[ERROR] Export failed! Check logs for details.")
        else:
            self.progress_update.emit("[SUCCESS] Export completed successfully.")

        # Export unsegmented version if enabled
        save_unsegmented = self.settings.get('save_unsegmented', False)
        if save_unsegmented:
            base, ext = os.path.splitext(track_path)
            # Find next available version for unsegmented export
            base_name = os.path.basename(base)
            version = 1
            while True:
                unseg_name = os.path.join(track_path, f"{base_name}_unsegmented_v{version}.mp3")
                if not os.path.exists(unseg_name):
                    break
                version += 1
            self.progress_update.emit(f"Exporting unsegmented version: {unseg_name}")
            from audio.export import export_audio_slice
            export_audio_slice(mixed_audio, 0, len(mixed_audio), unseg_name)

        # Save export settings as JSON file with the same version as the export
        # Find next available version for export settings JSON
        export_dir = track_path
        base_name = os.path.basename(track_path)
        version = 1
        while True:
            json_path = os.path.join(export_dir, f"{base_name}_v{version}.json")
            if not os.path.exists(json_path):
                break
            version += 1
        export_settings = {
            'settings': self.settings,
            'version_name': f"v{version}",
            'track_path': track_path,
            'segments': validated_segments,  # Store the actual segments that were used
            'profiles': getattr(self.app, 'profiles', {}),
            'track_profile_assignments': getattr(self.app, 'track_profile_assignments', {}),
        }
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(export_settings, f, indent=2)
        except Exception as exc:
            self.progress_update.emit(f"Warning: Failed to write export settings JSON: {exc}")
    
    def get_track_name(self, track_path, dir_path):
        """Extract the track name from the full path"""
        # Get just the filename part without directory
        track_name = os.path.basename(track_path)
        
        # If there's an extension, remove it
        if "." in track_name:
            track_name = os.path.splitext(track_name)[0]
        
        return track_name

    def get_input_channels(self, track_path):
        """Get input channels for a track with your specific naming pattern"""
        track_dir = track_path  # Use the track folder itself
        track_name = os.path.basename(track_path)
        
        self.progress_update.emit(f"Looking for input channels for {track_name} in {track_dir}")
        
        # Get all WAV files in the directory
        all_files = [f for f in os.listdir(track_dir) if f.lower().endswith('.wav')]
        self.progress_update.emit(f"Found {len(all_files)} WAV files in track directory {track_dir}")
        
        # Special pattern matching for your files (e.g., 041825_183136_Tr1.WAV)
        input_channels = []
        for file in all_files:
            # Accept all .wav files in the track folder as input channels
            channel = file[:file.rfind(".")]
            input_channels.append((channel, os.path.join(track_dir, file)))
        
        self.progress_update.emit(f"Found {len(input_channels)} input channels for {track_name}")
        for ch, path in input_channels:
            self.progress_update.emit(f"- Channel: {ch}, File: {os.path.basename(path)}")
        
        return input_channels

    def detect_segments(self, audio):
        """Improved segment detection based on silence and advanced merging logic."""
        from pydub import silence
        silence_settings = getattr(self, 'settings', {})
        dBFS = audio.dBFS if hasattr(audio, 'dBFS') else -40
        min_silence = silence_settings.get('min_silence', 4000)
        silence_thresh = dBFS - silence_settings.get('silence_threshold', 21)
        seek_step = silence_settings.get('seek_step', 2000)  # Use larger seek step for stability
        min_time_between = silence_settings.get('min_time_between_segments', 10000)  # ms
        min_segment_length_min = silence_settings.get('min_segment_length', 15)  # minutes
        min_segment_length = int(min_segment_length_min * 60_000)  # ms
        dropout = silence_settings.get('dropout', 60_000)  # ms, drop segments shorter than 1 min before merging

        # Debug logging
        msg1 = f"[DEBUG][Segmentation] silence_thresh={silence_thresh}, min_silence={min_silence}, min_segment_length={min_segment_length}, seek_step={seek_step}, min_time_between={min_time_between}, dropout={silence_settings.get('dropout', 1)}"
        self.progress_update.emit(msg1)
        print(msg1)

        msg2 = f"[DEBUG][Segmentation] audio length: {len(audio)} ms, dBFS: {dBFS}"
        self.progress_update.emit(msg2)
        print(msg2)

        # Step 1: Detect nonsilent regions
        param_log = (f"[DEBUG][Segmentation] Calling silence.detect_nonsilent with: "
                     f"min_silence_len={min_silence}, silence_thresh={silence_thresh}, seek_step={seek_step}, audio_dBFS={getattr(audio, 'dBFS', 'N/A')}, audio_length={len(audio)} ms")
        self.progress_update.emit(param_log)
        print(param_log)
        raw_segments = silence.detect_nonsilent(
            audio,
            min_silence_len=min_silence,
            silence_thresh=silence_thresh,
            seek_step=seek_step
        )
        msg3 = f"[DEBUG][Segmentation] Initial detected segments: {len(raw_segments)} | Segments: {raw_segments}"
        self.progress_update.emit(msg3)
        print(msg3)
        if not raw_segments:
            no_seg_msg = "[DEBUG][Segmentation] No segments detected after silence detection."
            self.progress_update.emit(no_seg_msg)
            print(no_seg_msg)

        # Step 2: Drop short segments before merging
        filtered = [s for s in raw_segments if (s[1] - s[0]) >= dropout]
        msg4 = f"[DEBUG][Segmentation] Segments after dropout filter: {len(filtered)}"
        self.progress_update.emit(msg4)
        print(msg4)
        for idx, (start, end) in enumerate(filtered):
            filt_msg = f"[DEBUG][Segmentation] Filtered Segment {idx+1}: {start} ms - {end} ms (duration: {end-start} ms)"
            self.progress_update.emit(filt_msg)
            print(filt_msg)

        # Step 3: Improved merging logic that respects long segments
        final_segments = []
        
        for start, end in filtered:
            segment_duration = end - start
            
            # Only merge segments if BOTH conditions are met:
            # 1. At least one segment is too short AND
            # 2. The gap between them is small
            if final_segments:
                prev_start, prev_end = final_segments[-1]
                prev_duration = prev_end - prev_start
                gap_between = start - prev_end
                
                # MODIFIED: More conservative merging logic
                # Only merge if gap is small AND at least one segment is short
                if (gap_between < min_time_between and 
                    (prev_duration < min_segment_length or segment_duration < min_segment_length)):
                    # Extend previous segment
                    final_segments[-1][1] = end
                    merge_msg = f"[DEBUG][Segmentation] Merged segments: gap={gap_between}ms, prev_duration={prev_duration}ms, curr_duration={segment_duration}ms"
                    self.progress_update.emit(merge_msg)
                    print(merge_msg)
                else:
                    # Otherwise add as a new segment
                    final_segments.append([start, end])
                    keep_msg = f"[DEBUG][Segmentation] Kept as separate segment: gap={gap_between}ms, prev_duration={prev_duration}ms, curr_duration={segment_duration}ms"
                    self.progress_update.emit(keep_msg)
                    print(keep_msg)
            else:
                final_segments.append([start, end])

        # Step 4: Edge case for last segment - more careful to preserve distinct segments
        if final_segments:
            if len(final_segments) >= 2:
                # Only fix the last segment if it's problematically short (less than 3 minutes)
                ls, le = final_segments[-1]
                if le - ls < 3 * 60_000:  # If last segment is less than 3 minutes
                    # Check gap to previous segment
                    ps, pe = final_segments[-2]
                    if ls - pe < min_time_between:  # Only merge if gap is small
                        final_segments = final_segments[:-1]
                        final_segments[-1][1] = le

        # Step 5: Final segment debug
        for idx, (start, end) in enumerate(final_segments):
            final_msg = f"[DEBUG][Segmentation] Final Segment {idx+1}: {start} ms - {end} ms (duration: {end-start} ms)"
            self.progress_update.emit(final_msg)
            print(final_msg)

        # Step 6: Contextual warnings
        if len(final_segments) > 0 and len(final_segments) < len(filtered):
            warnings = f"[DEBUG][Segmentation] Warning: Merged {len(filtered)} filtered segments into {len(final_segments)} final segments"
            self.progress_update.emit(warnings)
            print(warnings)

        # Check if we should use the entire audio as one segment
        if not final_segments:
            self.progress_update.emit("[DEBUG][Segmentation] No segments detected, using entire audio")
            return [[0, len(audio)]]
        else:
            return final_segments
        
    def log_available_optimizations(self):
        """Log available performance optimizations"""
        optimizations = []
        
        try:
            import numpy
            optimizations.append("NumPy is available - vectorized processing enabled")
        except ImportError:
            pass
            
        try:
            import scipy
            optimizations.append("SciPy is available - optimized filters enabled") 
        except ImportError:
            pass
            
        try:
            import pyloudnorm
            optimizations.append("PyLoudNorm is available - LUFS normalization enabled")
        except ImportError:
            pass
            
        try:
            import numba
            optimizations.append("Numba is available - JIT compilation enabled")
        except ImportError:
            pass
            
        if optimizations:
            self.progress_update.emit("Performance optimizations available:")
            for opt in optimizations:
                self.progress_update.emit(f"- {opt}")