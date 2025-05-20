import os
import re
import time
import json
from datetime import datetime
from pydub import AudioSegment
import numpy as np

def organize_files_into_tracks(files, input_format='.wav'):
    """Organize audio files into tracks based on filename patterns"""
    tracks = {}
    
    for file in files:
        # Handle case-insensitive file extension matching
        if not file.lower().endswith(input_format.lower()):
            continue
            
        # Find last underscore which separates track name from channel
        last_underscore = file.rfind('_')
        if last_underscore <= 0:
            continue
            
        # Extract track name and channel
        file_name = os.path.basename(file)
        track_name = file_name[:last_underscore]
        # Get channel (everything between the last underscore and file extension)
        channel = file_name[last_underscore+1:file_name.rfind('.')]
        
        # Add to tracks dictionary
        if track_name not in tracks:
            tracks[track_name] = {}
            
        tracks[track_name][channel] = file
    
    return tracks

def detect_track_type(filename):
    """
    Detect the type of track from filename
    
    Args:
        filename: Track filename
        
    Returns:
        Detected track type (vocal, tabla, etc.)
    """
    filename = filename.lower()
    
    if any(x in filename for x in ["voc", "vocal", "kirtan", "tr1", "track1", "track 1", "mic1"]):
        return "Kirtan (Vocals)"
    elif any(x in filename for x in ["tabla", "tr2", "track2", "track 2", "mic2"]):
        return "Tabla"
    elif any(x in filename for x in ["harmonium", "harm", "tr3", "track3", "track 3", "mic3"]):
        return "Sangat (Harmonium)"
    elif any(x in filename for x in ["tamboura", "tanp", "tr4", "track4", "track 4", "mic4"]):
        return "Sangat (Tamboura)"
    elif any(x in filename for x in ["room", "amb", "lr", "audience"]):
        return "Sangat (Ambient)"
    else:
        return "Sangat (Other)"

def detect_version_info(output_dir, track_name):
    """
    Detect existing versions of a track in output directory
    
    Args:
        output_dir: Directory to check for existing versions
        track_name: Base track name to search for
        
    Returns:
        Dictionary with version information
    """
    if not os.path.exists(output_dir):
        return None
        
    # Find all segment files for this track
    version_pattern = re.compile(f"{track_name}_v([^\\s-]+)")
    versions = {}
    
    for file in os.listdir(output_dir):
        if file.startswith(track_name):
            match = version_pattern.match(file)
            if match:
                version = match.group(1)
                if version not in versions:
                    versions[version] = []
                versions[version].append(file)
    
    # Prepare version info
    if versions:
        # Get last version
        last_version = sorted(versions.keys())[-1]
        return {
            'name': f"v{last_version}",
            'files': versions[last_version]
        }
    
    return None

def format_time(ms):
    """Format milliseconds as MM:SS"""
    seconds = ms / 1000
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def safe_filename(name):
    """Convert a string to a safe filename"""
    return "".join([c for c in name if c.isalpha() or c.isdigit() or c in ' ._-']).rstrip()

def trim_mp3_segment(mp3_path, out_path, pre_seconds=0, post_seconds=0):
    """
    Trim pre_seconds from the start and post_seconds from the end of an MP3 file.
    This function is used in 'Trim Only' mode and MUST NOT apply any normalization, compression, limiting, or any processing that could alter the audio volume or quality.
    Args:
        mp3_path: Path to the input MP3 file.
        out_path: Path to save the trimmed MP3 file. The function will add a suffix with the trim parameters.
        pre_seconds: Seconds to trim from the start.
        post_seconds: Seconds to trim from the end.
    Returns:
        Path to the trimmed MP3 file.
    """
    import logging
    
    # Create new filename with suffix indicating trim parameters
    file_dir = os.path.dirname(out_path)
    file_base = os.path.splitext(os.path.basename(out_path))[0]
    file_ext = os.path.splitext(out_path)[1]
    
    # Create suffix based on parameters
    suffix = f"_trim_pre{pre_seconds}_post{post_seconds}"
    
    # Create new output path with suffix
    new_out_path = os.path.join(file_dir, f"{file_base}{suffix}{file_ext}")
    
    logging.debug(f"[Trim Only] Trimming MP3: {mp3_path} (pre={pre_seconds}s, post={post_seconds}s) -> {new_out_path}")
    
    # Absolutely no normalization, compression, or limiting should be done here!
    from pydub import AudioSegment
    audio = AudioSegment.from_mp3(mp3_path)
    duration = len(audio) / 1000.0
    start_ms = max(0, int(pre_seconds * 1000))
    end_ms = len(audio) - int(post_seconds * 1000) if post_seconds > 0 else len(audio)
    if end_ms < start_ms:
        raise ValueError("Trim values exceed audio duration.")
    trimmed = audio[start_ms:end_ms]
    trimmed.export(new_out_path, format="mp3")
    return new_out_path

def fade_in_out_mp3(mp3_path, out_path, fade_in_sec=0, fade_out_sec=0):
    """
    Apply fade in and/or fade out to an MP3 file and save the result.
    Args:
        mp3_path: Path to the input MP3 file.
        out_path: Path to save the faded MP3 file. The function will add a suffix with the fade parameters.
        fade_in_sec: Seconds for fade in at the start.
        fade_out_sec: Seconds for fade out at the end.
    Returns:
        Path to the faded MP3 file.
    """
    # Create new filename with suffix indicating fade parameters
    file_dir = os.path.dirname(out_path)
    file_base = os.path.splitext(os.path.basename(out_path))[0]
    file_ext = os.path.splitext(out_path)[1]
    
    # Create suffix based on parameters
    suffix = f"_fade_in{fade_in_sec}_out{fade_out_sec}"
    
    # Create new output path with suffix
    new_out_path = os.path.join(file_dir, f"{file_base}{suffix}{file_ext}")
    
    from pydub import AudioSegment
    audio = AudioSegment.from_mp3(mp3_path)
    if fade_in_sec > 0:
        audio = audio.fade_in(int(fade_in_sec * 1000))
    if fade_out_sec > 0:
        audio = audio.fade_out(int(fade_out_sec * 1000))
    audio.export(new_out_path, format="mp3")
    return new_out_path

def export_segments(audio, segments, track, chosen_dir_path, prefix, version_info=None, log_callback=print):
    """Export segments based on the detected vocal segments"""
    try:
        # Import this to check master export flag
        from audio.export import IS_MASTER_EXPORT
        
        # Set this module as non-master to avoid duplicate JSON files
        # We'll let audio.export module handle the JSON creation
        from audio.export import IS_MASTER_EXPORT as _
        import sys
        if 'audio.export' in sys.modules:
            sys.modules['audio.export'].IS_MASTER_EXPORT = True
            this_is_master = False  # We're being called as a secondary export
        else:
            this_is_master = True   # We're the only export happening
        
        # CRITICAL FIX: Validate segments first
        if not segments or len(segments) == 0:
            log_callback("[ERROR] No segments provided to export function!")
            return None
        
        # CRITICAL FIX: Make a deep copy of the segments to prevent corruption
        validated_segments = []
        for i, segment in enumerate(segments):
            if isinstance(segment, (list, tuple)) and len(segment) == 2:
                # Force integer conversion and ensure correct order (start, end)
                try:
                    start = int(segment[0])
                    end = int(segment[1])
                    # Fix: Only add if start < end and within audio bounds
                    if start < end and end <= len(audio):
                        validated_segments.append((start, end))
                        log_callback(f"[DEBUG] Validated segment {i+1}: {start}ms to {end}ms (duration: {end-start}ms)")
                    else:
                        log_callback(f"[WARNING] Invalid segment boundaries: {start}-{end}, skipping")
                except (ValueError, TypeError):
                    log_callback(f"[WARNING] Invalid segment data type: {segment}, skipping")
            else:
                log_callback(f"[WARNING] Invalid segment format: {segment}, skipping")
                
        # If validation removed all segments, return without exporting
        if not validated_segments:
            log_callback("[ERROR] No valid segments after validation, nothing to export!")
            return None
        
        # Use validated segments from now on
        segments = validated_segments
        
        # Log the validated segments
        log_callback(f"[DEBUG] Using {len(segments)} validated segments:")
        for i, (s, e) in enumerate(segments):
            log_callback(f"[DEBUG] Segment {i+1}: {s}ms to {e}ms (duration: {e-s}ms)")
        
        # Validate version_info is a dictionary if provided
        if version_info is not None and not isinstance(version_info, dict):
            log_callback(f"Warning: Invalid version info format for {track}, using default")
            version_info = None
        
        # Get version name from info or generate a new one if none provided
        if isinstance(version_info, dict) and version_info.get('name'):
            version_name = version_info.get('name')
            log_callback(f"Using provided version: {version_name}")
        else:
            # Generate timestamp-based unique version name
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            version_name = f"v{timestamp}"
            log_callback(f"Created unique timestamp version: {version_name}")
        
        # Create the edited directory if it doesn't exist
        rel_path = os.path.dirname(os.path.relpath(track, chosen_dir_path))
        export_dir = os.path.join(chosen_dir_path, "edited", rel_path)
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        
        # Get base track name
        track_name = os.path.basename(track)
        
        # Create base export path 
        export_path_base = os.path.join(export_dir, f"{track_name}_{version_name}")
        
        # Log the actual output path for debugging
        log_callback(f"Export path will be: {export_path_base} - Segment X.mp3")
        
        # Create base metadata
        base_metadata = {
            'track': track,
            'processed_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'segments': len(segments),
            'version': version_name,
            'profile': version_info.get('profile') if isinstance(version_info, dict) else None,
            'description': version_info.get('description', '') if isinstance(version_info, dict) else None
        }
        
        # Export each segment
        for i, (start, end) in enumerate(segments):
            segment_number = i + 1
            segment_filename = f"{export_path_base} - Segment {segment_number}.mp3"
            log_callback(f"[LOG] Exporting to: {segment_filename} (start={start}, end={end})")
            
            # CRITICAL FIX: Verify segment boundaries before slicing
            if start < 0 or end > len(audio) or start >= end:
                log_callback(f"[ERROR] Invalid segment boundaries: {start}-{end}, skipping")
                continue
                
            try:
                # Export the audio segment
                segment = audio[start:end]
                
                # Export with appropriate MP3 parameters
                mp3_params = {
                    "format": "mp3",
                    "bitrate": "192k",
                    "tags": {
                        "album": "Kirtan Recording",
                        "title": f"{os.path.basename(track)} - Segment {segment_number}"
                    }
                }
                
                segment.export(segment_filename, **mp3_params)
                log_callback(f"[LOG] Successfully exported segment {segment_number} to {segment_filename}")
                
                # Save matching JSON metadata file ONLY if we're the master export
                # This prevents duplicate JSON files
                if this_is_master:
                    segment_metadata = base_metadata.copy()
                    segment_metadata['segment_number'] = segment_number
                    segment_metadata['segment_range'] = [start, end]
                    segment_metadata['duration_ms'] = end - start
                    
                    # Use the same filename as the MP3 but with .json extension
                    json_filename = segment_filename.replace('.mp3', '.json')
                    with open(json_filename, 'w') as f:
                        json.dump(segment_metadata, f, indent=2)
                    log_callback(f"[LOG] Saved metadata to {os.path.basename(json_filename)}")
                
                duration = (end - start) / 1000  # Convert to seconds
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                
                log_callback(f"Exported segment {segment_number}: {minutes}:{seconds:02d}")
            except Exception as seg_err:
                log_callback(f"[ERROR] Failed to export segment {segment_number}: {str(seg_err)}")
                import traceback
                log_callback(traceback.format_exc())
        
        log_callback(f"Created {len(segments)} segments with version {version_name}")
        
        # Restore the master flag if we changed it
        if not this_is_master and 'audio.export' in sys.modules:
            sys.modules['audio.export'].IS_MASTER_EXPORT = True
            
        return version_name
            
    except Exception as e:
        log_callback(f"Error exporting segments: {str(e)}")
        import traceback
        log_callback(traceback.format_exc())
        return None