# filepath: c:\git\kirtan-processor\core\processor.py

import os
import math
from audio.detection import detect_silence_efficiently
from audio.processing import process_audio_in_chunks
from audio.export import export_segments
from utils.cache import AudioCache
from utils.performance import PerformanceTracker
from utils.file_utils import format_time

class KirtanProcessor:
    """Core processing engine for Kirtan audio files"""
    
    def __init__(self, log_callback=print, progress_callback=None):
        self.log = log_callback
        self.update_progress = progress_callback or (lambda x: None)
        self.performance_tracker = PerformanceTracker(log_callback)
        self.audio_cache = AudioCache()
        self.stop_requested = False
        self.batch_normalize = False
    
    def process_track(self, track, dir_path, chosen_dir_path, profiles, track_profile_assignments, 
                      kirtan_channel=None, version_info=None, silence_settings=None):
        """
        Process a single track using Kirtan vocals as reference for segmentation
        
        Args:
            track: Track path
            dir_path: Source directory path
            chosen_dir_path: Output directory path
            profiles: Dictionary of processing profiles
            track_profile_assignments: Dictionary mapping channels to profile names
            kirtan_channel: Kirtan vocals channel name (autodetect if None)
            version_info: Version information for output files
            silence_settings: Settings for silence detection
            
        Returns:
            Success status
        """
        # Create a local audio_tracks dictionary for this track only
        local_audio_tracks = {}
        
        # Extract filenames from the directory
        filenames = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        
        # Get the track prefix (remove directory part)
        track_prefix = os.path.basename(track)
        
        # Find all files for this track
        track_files = {}
        for filename in filenames:
            if filename.startswith(track_prefix) and filename.endswith(".WAV"):
                # Extract input channel name from filename
                input_channel = filename[filename.rfind('_')+1:filename.rfind('.')]
                track_files[input_channel] = os.path.join(dir_path, filename)
        
        # Find or identify kirtan vocal channel if not provided
        if not kirtan_channel:
            for channel, profile_name in track_profile_assignments.items():
                if profile_name == "Kirtan (Vocals)" and channel in track_files:
                    kirtan_channel = channel
                    self.log(f"Found Kirtan vocal channel: {kirtan_channel}")
                    break
        
        if not kirtan_channel or kirtan_channel not in track_files:
            self.log("Error: No Kirtan vocal channel found for segmentation reference")
            return False
        
        # Load kirtan vocals for segmentation
        self.log(f"Loading Kirtan vocal track as segmentation reference")
        kirtan_file = track_files[kirtan_channel]
        kirtan_audio = self._load_audio(kirtan_file)
        
        if not kirtan_audio:
            self.log(f"Error loading Kirtan vocals from {kirtan_file}")
            return False
        
        # Get silence detection settings
        if not silence_settings:
            # Default settings
            silence_settings = {
                'threshold': 21,  # dB below average level
                'min_silence': 4000,  # ms
                'seek_step': 100,  # ms
                'min_segment': 60000,  # ms (1 minute)
                'min_time_between': 10000,  # ms
                'pre_padding': 0,  # ms
                'post_padding': 0  # ms
            }
        
        # Detect segments using silence detection
        self.log("Detecting segments using silence detection")
        segments = detect_silence_efficiently(
            kirtan_audio, 
            silence_settings['min_silence'],
            silence_settings['threshold'],
            silence_settings['seek_step']
        )
        
        # Process segments
        segments = self._refine_segments(
            segments, 
            min_segment_ms=silence_settings['min_segment'],
            min_time_between_ms=silence_settings['min_time_between']
        )
        
        if not segments:
            self.log("No segments found after processing")
            return False
        
        self.log(f"Found {len(segments)} segments after processing")
        
        # Process each track channel
        for channel, file_path in track_files.items():
            if self.stop_requested:
                self.log("Processing stopped by user")
                return False
            
            # Get profile name for this channel
            profile_name = track_profile_assignments.get(channel)
            if profile_name == "-- DO NOT PROCESS --":
                self.log(f"Skipping channel {channel} (marked as DO NOT PROCESS)")
                continue
                
            # Get profile settings
            profile = profiles.get(profile_name)
            if not profile:
                self.log(f"No profile found for {channel}, using default")
                profile = {}  # Use default settings
            
            # Import and process audio
            self.log(f"Processing {os.path.basename(file_path)}")
            audio_track = self._load_audio(file_path)
            if not audio_track:
                self.log(f"Error loading audio from {file_path}")
                continue
                
            # Process with profile
            processed_audio = process_audio_in_chunks(audio_track, profile, log_callback=self.log)
            
            # Store in local_audio_tracks dictionary
            local_audio_tracks[channel] = processed_audio
            
        # Mix all tracks AFTER all channels have been processed
        self.log(f"Mixing audio tracks")
        if local_audio_tracks and kirtan_channel in local_audio_tracks:
            # Apply pre-mix scaling to prevent level summing issues
            track_count = len(local_audio_tracks)
            if track_count > 1:
                # Calculate logarithmic scaling factor based on number of tracks
                # When tracks are combined, volume increases by ~10*log10(N) dB
                scale_db = -10 * math.log10(track_count)
                self.log(f"Applying pre-mix scaling of {scale_db:.1f} dB to prevent overload")
                
                # Scale all tracks by this factor before mixing
                for channel in local_audio_tracks:
                    local_audio_tracks[channel] = local_audio_tracks[channel] + scale_db
            
            # Now mix the scaled tracks
            audio = local_audio_tracks[kirtan_channel]  # Start with Kirtan track
            for channel, track_audio in local_audio_tracks.items():
                if channel != kirtan_channel:  # Mix in other tracks
                    audio = audio.overlay(track_audio)
                    
            # Export segments
            self.log(f"Exporting segments based on Kirtan vocals")
            result = export_segments(
                audio, 
                segments, 
                track, 
                chosen_dir_path, 
                track_prefix, 
                version_info,
                log_callback=self.log
            )
            
            return result is not None
        
        self.log("Error: No audio tracks to mix")
        return False
            
    def _load_audio(self, file_path):
        """Load audio file with caching"""
        # Try to get from cache first
        cached_audio = self.audio_cache.get(file_path)
        if cached_audio:
            self.log(f"Using cached version of {os.path.basename(file_path)}")
            return cached_audio
        
        # Load the file
        try:
            # Check file size
            file_size = os.path.getsize(file_path)
            
            # For very large files, use a different approach
            if file_size > 500_000_000:  # 500 MB
                self.log(f"Large file detected ({file_size/1_000_000:.1f} MB), using optimized import")
                audio = AudioSegment.from_file(file_path)
            else:
                # Standard import for smaller files
                audio = AudioSegment.from_file(file_path)
                
            # Add to cache
            self.audio_cache.add(file_path, audio)
            return audio
            
        except Exception as e:
            self.log(f"Error loading {os.path.basename(file_path)}: {str(e)}")
            return None
            
    def _refine_segments(self, segments, min_segment_ms=60000, min_time_between_ms=10000):
        """Refine detected segments based on user settings"""
        if not segments:
            return []
            
        final_segments = []
        prev_end = 0
        
        # Log segments with durations
        for i, (start, end) in enumerate(segments):
            # Ensure start and end are in the right order
            if start > end:
                self.log(f"[DEBUG][Segmentation] Warning: Found swapped start/end times in segment {i+1}: {format_time(start)} > {format_time(end)}, fixing...")
                start, end = end, start
                segments[i] = (start, end)
                
            duration = end - start
            self.log(f"[DEBUG][Segmentation] Filtered Segment {i+1}: {format_time(start)} - {format_time(end)} (duration: {format_time(duration)})")
            
        # First pass: filter out segments that are too short
        filtered_by_length = []
        
        for start, end in segments:
            # Double-check segment order again
            if start > end:
                start, end = end, start
                
            if end - start >= min_segment_ms:
                filtered_by_length.append((start, end))
            else:
                self.log(f"Dropping segment {format_time(start)}-{format_time(end)} (too short: {format_time(end-start)} < {format_time(min_segment_ms)})")

        # If no segments remain after filtering, return the longest original segment
        if not filtered_by_length and segments:
            longest_segment = max(segments, key=lambda seg: seg[1] - seg[0])
            # Ensure longest segment has valid start/end
            start, end = longest_segment
            if start > end:
                start, end = end, start
                longest_segment = (start, end)
                
            self.log(f"No segments meet minimum length. Using longest segment: {format_time(longest_segment[0])}-{format_time(longest_segment[1])}")
            return [longest_segment]
        elif not filtered_by_length:
            return []
            
        # Second pass: check for merging based on min_time_between_ms
        for start, end in filtered_by_length:
            # Validate segment again
            if start > end:
                start, end = end, start
                
            if final_segments:
                last_start, last_end = final_segments[-1]
                
                # Only merge if segments are close together
                if start - last_end < min_time_between_ms:
                    self.log(f"Merging with previous segment (gap: {format_time(start-last_end)} < {format_time(min_time_between_ms)})")
                    # Ensure the merged segment has valid boundaries
                    new_start = min(last_start, start)
                    new_end = max(last_end, end)
                    final_segments[-1] = (new_start, new_end)
                else:
                    self.log(f"New segment {format_time(start)}-{format_time(end)} (gap: {format_time(start-last_end)})")
                    final_segments.append((start, end))
            else:
                self.log(f"First segment {format_time(start)}-{format_time(end)}")
                final_segments.append((start, end))
                
            prev_end = end
        
        # Final validation pass
        validated_segments = []
        original_count = len(final_segments)
        
        for i, (start, end) in enumerate(final_segments):
            if start > end:
                self.log(f"[DEBUG][Segmentation] Fixing swapped start/end times in final segment {i+1}: {format_time(start)} > {format_time(end)}")
                start, end = end, start
                
            validated_segments.append((start, end))
            duration = end - start
            self.log(f"[DEBUG][Segmentation] Final Segment {i+1}: {format_time(start)} - {format_time(end)} (duration: {format_time(duration)})")
        
        if len(validated_segments) < len(final_segments):
            self.log(f"[DEBUG][Segmentation] Warning: Dropped {len(final_segments) - len(validated_segments)} invalid segments")
        elif original_count > 0 and len(validated_segments) > 0:
            self.log(f"[DEBUG][Segmentation] Warning: Merged {original_count} filtered segments into {len(validated_segments)} final segments")
            
        return validated_segments

    def process_directory(self, chosen_dir_path):
        """Process all subdirectories while preserving structure"""
        # Create edited directory if it doesn't exist
        edited_dir = os.path.join(chosen_dir_path, "edited")
        if not os.path.exists(edited_dir):
            os.makedirs(edited_dir)
        
        # Find all directories to process
        all_subdirs = []
        for root, dirs, files in os.walk(chosen_dir_path):
            if "edited" not in os.path.basename(root):
                # Only include directories with audio files
                audio_files = [f for f in files if f.lower().endswith('.wav')]
                if audio_files:
                    all_subdirs.append((root, dirs, audio_files))
        
        if not all_subdirs:
            self.log("No valid tracks found")
            return
        
        # Process each directory
        total_dirs = len(all_subdirs)
        for i, (dir_path, _, filenames) in enumerate(all_subdirs):
            if self.stop_requested:
                self.log("Processing stopped by user")
                break
            
            # Process tracks in this directory
            self.log(f"Processing directory {i+1}/{total_dirs}: {dir_path}")
            
            # Process each track in this directory
            self.process_directory_tracks(dir_path, filenames, chosen_dir_path)
            
            # Update progress
            self.update_progress(int((i+1) / total_dirs * 100))