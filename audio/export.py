import os
import json
from datetime import datetime
import gc
import time
from pydub import AudioSegment

# Global flag to track if this is the master export process
# This prevents duplicate JSON files from being created
IS_MASTER_EXPORT = True

def export_segments(audio, segments, track, chosen_dir_path, prefix, version_info=None, log_callback=None):
    """Export segments based on the detected vocal segments"""
    import time
    from utils.file_utils import format_time
    
    if log_callback is None:
        log_callback = print
    debug_mode = True  # Always enable debug mode for segment export
    if version_info and isinstance(version_info, dict):
        debug_mode = debug_mode or version_info.get('debug_mode', False) or version_info.get('settings', {}).get('debug_mode', False)
    def debug_log(msg):
        if debug_mode:
            log_callback(f"[DEBUG] {msg}")
    
    # Use global flag
    global IS_MASTER_EXPORT
    
    try:
        # Fix: Make sure segments contains actual segment data and is not empty
        if not segments or len(segments) == 0:
            log_callback("[ERROR] No segments provided to export function!")
            return False
            
        # Fix: Ensure segments have valid boundaries
        validated_segments = []
        log_callback(f"[DEBUG] Starting validation of {len(segments)} segments")
        log_callback(f"[DEBUG] Audio length: {format_time(len(audio))}")
        
        for i, segment in enumerate(segments):
            if isinstance(segment, (list, tuple)) and len(segment) == 2:
                # Fix: Make sure start/end are valid integers
                try:
                    start = int(segment[0])
                    end = int(segment[1])
                    
                    log_callback(f"[DEBUG] Validating segment {i+1}: pre-swap start={format_time(start)}, end={format_time(end)}")
                    
                    # Fix: Auto-correct swapped start/end times
                    if start > end:
                        log_callback(f"[WARNING] Fixing swapped start/end times for segment {i+1}: {format_time(start)} > {format_time(end)}")
                        start, end = end, start
                        log_callback(f"[DEBUG] After swap: start={format_time(start)}, end={format_time(end)}")
                    
                    # Additional validation details
                    valid_segment = True
                    validation_messages = []
                    
                    # Check boundaries
                    if start < 0:
                        valid_segment = False
                        validation_messages.append(f"start ({format_time(start)}) < 0")
                        
                    if end > len(audio):
                        valid_segment = False
                        validation_messages.append(f"end ({format_time(end)}) > audio_length ({format_time(len(audio))})")
                    
                    if start >= end:
                        valid_segment = False
                        validation_messages.append(f"start ({format_time(start)}) >= end ({format_time(end)})")
                        
                    # Final segment check
                    if valid_segment:
                        validated_segments.append((start, end))
                        log_callback(f"[DEBUG] Segment {i+1} validation PASSED: {format_time(start)} to {format_time(end)} (duration: {format_time(end-start)})")
                    else:
                        log_callback(f"[ERROR] Invalid segment {i+1}: {format_time(start)} to {format_time(end)} - failing conditions: {', '.join(validation_messages)}")
                except (ValueError, TypeError):
                    log_callback(f"[WARNING] Invalid segment data type: {segment}, skipping")
            else:
                log_callback(f"[WARNING] Invalid segment format: {segment}, skipping")
        
        # Fix: If validation removed all segments, return without exporting
        if not validated_segments:
            log_callback("[ERROR] No valid segments after validation, nothing to export!")
            return False

        # Fix: Store the validated segment list as our working set
        segments = validated_segments

        t0 = time.time()
        debug_log("Export: Calculating export directory path")
        export_dir = track  # Always save in the track folder itself
        t1 = time.time()
        debug_log(f"Export: Directory path calculated in {1000*(t1-t0):.1f} ms")
        debug_log(f"Export: Ensuring export directory exists: {export_dir}")
        os.makedirs(export_dir, exist_ok=True)
        t2 = time.time()
        
        debug_log(f"Export: Directory creation checked in {1000*(t2-t1):.1f} ms")
        
        track_name = os.path.basename(track)
        version_name = version_info.get('name') if isinstance(version_info, dict) else None
        if not version_name:
            ver = 1
            while os.path.exists(os.path.join(export_dir, f"{track_name}_v{ver} - Segment 1.mp3")):
                ver += 1
            version_name = f"v{ver}"
            log_callback(f"Auto-assigned version: {version_name}")
        export_path_base = os.path.join(export_dir, f"{track_name}_{version_name}")
        log_callback(f"Export path will be: {export_path_base} - Segment X.mp3")
        
        # Debug: Log received segments
        log_callback(f"[DEBUG] Received {len(segments)} segments for export:")
        for i, (s, e) in enumerate(segments):
            log_callback(f"[DEBUG] Segment {i+1}: {format_time(s)} to {format_time(e)} (duration: {format_time(e-s)})")
            
        export_times = []
        
        for i, (start, end) in enumerate(segments):
            segment_number = i + 1
            segment_filename = f"{export_path_base} - Segment {segment_number}.mp3"
            log_callback(f"[LOG] Preparing to export segment {segment_number}: {segment_filename} (start={format_time(start)}, end={format_time(end)})")
            debug_log(f"Export: Starting segment {segment_number} ({format_time(start)}-{format_time(end)})")
            t_seg_start = time.time()
            try:
                # Use our improved export_audio_slice function for better compatibility
                result = export_audio_slice(audio, start, end, segment_filename)
                
                if result:
                    log_callback(f"[LOG] Successfully exported segment {segment_number} to {segment_filename}")
                    
                    # Save metadata alongside the segment - only if this is master export
                    if IS_MASTER_EXPORT:
                        try:
                            # Use exactly the same filename but with .json extension
                            metadata_filename = segment_filename.replace('.mp3', '.json')
                            metadata = {
                                'segment_number': segment_number,
                                'segment_range': [start, end],
                                'duration_ms': end - start,
                                'duration_formatted': format_time(end - start),
                                'export_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'version': version_name
                            }
                            
                            # Add version_info if available
                            if version_info and isinstance(version_info, dict):
                                metadata['profile'] = version_info.get('profile')
                                metadata['settings'] = version_info.get('settings')
                                metadata['description'] = version_info.get('description', '')
                            
                            with open(metadata_filename, 'w') as f:
                                json.dump(metadata, f, indent=2)
                                
                            log_callback(f"[LOG] Saved metadata to {os.path.basename(metadata_filename)}")
                        except Exception as metadata_err:
                            log_callback(f"[WARNING] Could not save metadata for segment {segment_number}: {metadata_err}")
                else:
                    log_callback(f"[ERROR] Failed to export segment {segment_number} - unknown error")
            except Exception as seg_export_err:
                log_callback(f"[ERROR] Failed to export segment {segment_number}: {seg_export_err}")
                import traceback
                log_callback(traceback.format_exc())
                continue
                
            t_seg_end = time.time()
            debug_log(f"Export: Segment {segment_number} exported in {1000*(t_seg_end-t_seg_start):.1f} ms")
            export_times.append(t_seg_end-t_seg_start)
            
        t3 = time.time()
        debug_log(f"Export: All segments exported in {1000*(t3-t2):.1f} ms; avg per segment: {1000*(sum(export_times)/len(export_times)) if export_times else 0:.1f} ms")
        log_callback(f"Created {len(segments)} segments with version {version_name}")
        return True
    except Exception as e:
        log_callback(f"Error exporting segments: {str(e)}")
        import traceback
        log_callback(traceback.format_exc())
        return False

def export_audio_slice(audio, start, end, export_path, padding_settings=None):
    """Export a slice of audio with configurable padding"""
    try:
        # Validate inputs as integers
        start = int(start)
        end = int(end)
        
        # Debug info
        print(f"[DEBUG] Slicing audio from {start}ms to {end}ms (duration: {end-start}ms)")
        print(f"[DEBUG] Full audio duration: {len(audio)}ms")
        
        # Handle potential boundary errors gracefully
        if start < 0:
            print(f"[WARNING] Start time {start}ms is negative, adjusting to 0ms")
            start = 0
            
        if end > len(audio):
            print(f"[WARNING] End time {end}ms exceeds audio length {len(audio)}ms, adjusting")
            end = len(audio)
            
        if start >= end:
            print(f"[ERROR] Invalid slice: start time {start}ms >= end time {end}ms")
            return None
        
        print(f"[DEBUG] Adjusted slice: {start}ms to {end}ms (duration: {end-start}ms)")
        
        # Extract the segment
        segment = audio[start:end]
        print(f"[DEBUG] Created segment with duration: {len(segment)}ms")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
        
        # Export with optimized MP3 settings
        segment.export(
            export_path, 
            format="mp3", 
            bitrate="192k",
            parameters=["-q:a", "2"]  # High quality settings
        )
        
        file_size = os.path.getsize(export_path)
        print(f"Successfully exported file: {export_path} ({file_size} bytes, duration: {len(segment)}ms)")
        
    except Exception as e:
        print(f"Error exporting audio slice: {str(e)}")
        return None
        
    # Memory cleanup
    del segment
    gc.collect(generation=0)
    
    return export_path