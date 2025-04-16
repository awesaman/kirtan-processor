#!/usr/bin/env python
import sys
import os
import time
import json
import threading
import psutil
import concurrent.futures as conc
from itertools import chain
import gc
from gc import collect
import subprocess
import multiprocessing
import numpy as np
import concurrent.futures
from pydub import silence
from datetime import datetime


from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                           QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, 
                           QComboBox, QSpinBox, QTextEdit, QTabWidget, 
                           QScrollArea, QGroupBox, QSlider, QCheckBox, 
                           QProgressBar, QSplitter, QGridLayout, QLineEdit,
                           QMessageBox, QFrame, QTableWidget, QTableWidgetItem,
                           QHeaderView, QStyledItemDelegate, QSizePolicy, QDialog, QListWidget, QDoubleSpinBox)
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, QSettings, QTimer, QEventLoop
from PyQt6.QtGui import QFont, QPixmap, QIcon, QColor, QBrush

from pydub import AudioSegment, silence, effects

# Near the top of the file, add these imports
try:
    import numpy as np
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# If Numba is available, add this JIT-compiled function
if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def _compute_rms_windows(samples, window_size, hop_size):
        """JIT-compiled function to compute RMS values for sliding windows"""
        result = []
        positions = []
        
        for i in range(0, len(samples) - window_size, hop_size):
            window = samples[i:i+window_size]
            # Compute RMS for this window
            rms = np.sqrt(np.mean(np.square(window.astype(np.float64))))
            result.append(rms)
            positions.append(i)
            
        return positions, result

from contextlib import contextmanager
import time

ONE_MIN = 60000
INPUT_FORMAT = ".WAV"

# Add this after the existing imports
DEFAULT_PROFILES = ["Kirtan (Vocals)", "Tabla", "Sangat (Ambient)"]

class ProcessingWorker(QThread):
    progress_update = pyqtSignal(str)
    progress_bar = pyqtSignal(int)
    processing_finished = pyqtSignal()
    track_finished = pyqtSignal(str)
    version_finished = pyqtSignal(str, str)  # Add signal for version completion
    
    def __init__(self, app, dir_path):
        super().__init__()
        self.app = app
        self.dir_path = dir_path
        self.stop_requested = False
        self.processor_count = self.determine_processor_count()
        self.audio_cache = {}
        self.audio_cache_size = 0  # Initialize audio cache size
        self.max_cache_size = 1024 * 1024 * 1024  # 1GB max cache size
        self.audio_tracks = {}  # Initialize audio_tracks dictionary
        self.track_versions = getattr(app, 'track_versions', {})  # Get versions from app
        
        # Add performance tracking
        self.performance_log = {}  # Dictionary to track operation times
    
    @contextmanager
    def measure_time(self, operation_name):
        """Context manager to measure operation time"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed_time = time.time() - start_time
            if operation_name not in self.performance_log:
                self.performance_log[operation_name] = []
            self.performance_log[operation_name].append(elapsed_time)
            
            # Log if operation takes longer than expected
            if elapsed_time > 5.0:  # Only log slow operations
                self.progress_update.emit(f"Performance: {operation_name} took {elapsed_time:.2f}s")
    
    def get_cached_audio(self, file_path):
        """Get audio from cache or load if not cached"""
        if file_path in self.audio_cache:
            self.progress_update.emit(f"Using cached version of {os.path.basename(file_path)}")
            return self.audio_cache[file_path]
        
        # Load the audio
        audio = self.import_audio(os.path.dirname(file_path), file_path)
        
        # Only cache if we have enough memory
        memory = psutil.virtual_memory()
        if memory.percent < 70 and self.audio_cache_size < self.max_cache_size:
            # Estimate size: samples × bit depth × channels
            estimated_size = len(audio) * audio.frame_width
            
            # Check if adding this would exceed cache limit
            if self.audio_cache_size + estimated_size < self.max_cache_size:
                self.audio_cache[file_path] = audio
                self.audio_cache_size += estimated_size
                self.progress_update.emit(f"Added {os.path.basename(file_path)} to cache ({estimated_size/1_000_000:.1f} MB)")
        
        return audio

    
    # In the determine_processor_count method - use ALL cores with no restrictions
    def determine_processor_count(self):
        """More adaptive processor count determination"""
        cpu_count = multiprocessing.cpu_count()
        available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
        
        if self.app.processing_speed == "Full Speed":
            # Use all cores but cap based on available memory
            memory_based_limit = max(1, int(available_memory / 1.5))
            return max(1, min(cpu_count, memory_based_limit))
        elif self.app.processing_speed == "Auto":
            # More nuanced auto determination
            if cpu_count <= 4:
                return max(1, cpu_count - 1)  # Leave 1 core free on small systems
            else:
                return max(1, int(cpu_count * 0.7))  # Use 70% of cores on larger systems
        else:  # "Slow"
            return 1
    
    def detect_silence_efficiently(self, audio, min_silence_len, silence_thresh, seek_step):
        """More efficient silence detection algorithm with optional JIT acceleration"""
        try:
            import numpy as np
            from pydub import silence
            
            # Get audio parameters
            sample_rate = audio.frame_rate
            samples = np.array(audio.get_array_of_samples())
            
            # Convert to mono if stereo for faster processing
            if audio.channels == 2:
                # Simple average of channels
                samples = samples.reshape(-1, 2).mean(axis=1).astype(samples.dtype)
            
            # Parameters in samples rather than ms
            min_silence_samples = (min_silence_len * sample_rate) // 1000
            seek_samples = (seek_step * sample_rate) // 1000
            
            # Convert silence threshold to RMS value
            dBFS = audio.dBFS
            silence_thresh_db = dBFS - silence_thresh
            ref_max = float(2 ** (audio.sample_width * 8 - 1))
            silence_rms_thresh = ref_max * (10 ** (silence_thresh_db / 20))
            
            # Process in windows for memory efficiency
            window_size = min(seek_samples, 16384)  # Use reasonable window size
            hop_size = window_size // 2  # 50% overlap
            
            # Use accelerated version if available
            if 'NUMBA_AVAILABLE' in globals() and NUMBA_AVAILABLE:
                self.progress_update.emit("Using JIT-accelerated silence detection")
                positions, rms_values = _compute_rms_windows(samples, window_size, hop_size)
            else:
                # Vanilla version
                positions = []
                rms_values = []
                for i in range(0, len(samples) - window_size, hop_size):
                    window = samples[i:i+window_size]
                    rms = np.sqrt(np.mean(np.square(window.astype(np.float64))))
                    positions.append(i)
                    rms_values.append(rms)
            
            # Find silent regions
            silent_regions = []
            is_silence = False
            current_silence_start = 0
            
            for i, (pos, rms) in enumerate(zip(positions, rms_values)):
                # Convert position from samples to milliseconds
                ms_pos = (pos * 1000) // sample_rate
                
                if rms < silence_rms_thresh:  # This is silence
                    if not is_silence:
                        # Start of a new silent region
                        is_silence = True
                        current_silence_start = ms_pos
                else:  # Not silence
                    if is_silence:
                        # End of a silent region - check if it's long enough
                        silence_duration = ms_pos - current_silence_start
                        if silence_duration >= min_silence_len:
                            silent_regions.append((current_silence_start, ms_pos))
                        is_silence = False
            
            # Add final silent region if file ends with silence
            if is_silence:
                ms_pos = (len(samples) * 1000) // sample_rate
                silence_duration = ms_pos - current_silence_start
                if silence_duration >= min_silence_len:
                    silent_regions.append((current_silence_start, ms_pos))
            
            
            # Convert to non-silent regions
            non_silent_regions = []
            
            # If no silent regions found, the whole file is non-silent
            if not silent_regions:
                non_silent_regions = [(0, len(audio))]
            else:
                # Add non-silent region at beginning if needed
                if silent_regions[0][0] > 0:
                    non_silent_regions.append((0, silent_regions[0][0]))
                    
                # Add non-silent regions between silent regions
                for i in range(len(silent_regions) - 1):
                    non_silent_regions.append((silent_regions[i][1], silent_regions[i+1][0]))
                    
                # Add non-silent region at end if needed
                if silent_regions[-1][1] < len(audio):
                    non_silent_regions.append((silent_regions[-1][1], len(audio)))
            
            # Handle trim_only mode
            if hasattr(self.app, 'trim_only') and self.app.trim_only and len(non_silent_regions) > 0:
                self.progress_update.emit("Trim-only mode: will only trim silence at start/end")
                # Just return a single segment from the start of the first non-silent region
                # to the end of the last non-silent region
                first_start = non_silent_regions[0][0]
                last_end = non_silent_regions[-1][1]
                non_silent_regions = [(first_start, last_end)]
                self.progress_update.emit(f"Trimmed segment: {self.format_time(first_start)} to {self.format_time(last_end)}")
            
            self.progress_update.emit(f"Found {len(non_silent_regions)} non-silent regions using efficient algorithm")
            return non_silent_regions
    
            
        except Exception as e:
            self.progress_update.emit(f"Error in efficient silence detection: {str(e)}")
            self.progress_update.emit("Falling back to standard silence detection")
            # Fall back to standard silence detection
            from pydub import silence
            non_silent = silence.detect_nonsilent(
                audio,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                seek_step=seek_step
            )
            
            # Handle trim_only mode for the fallback method too
            if hasattr(self.app, 'trim_only') and self.app.trim_only and len(non_silent) > 0:
                self.progress_update.emit("Trim-only mode: will only trim silence at start/end")
                first_start = non_silent[0][0]
                last_end = non_silent[-1][1]
                non_silent = [(first_start, last_end)]
                self.progress_update.emit(f"Trimmed segment: {self.format_time(first_start)} to {self.format_time(last_end)}")
                
            return non_silent
    def summarize_performance(self):
        """Generate a summary of operation times"""
        if not self.performance_log:
            return "No performance data available."
            
        summary = ["Performance Summary:"]
        
        for operation, times in self.performance_log.items():
            avg_time = sum(times) / len(times)
            max_time = max(times)
            total_time = sum(times)
            summary.append(f"  {operation}: {len(times)} calls, avg={avg_time:.2f}s, max={max_time:.2f}s, total={total_time:.2f}s")
        
        return "\n".join(summary)
        
    def finish_processing(self):
        """Finish processing without showing performance summary"""
        self.progress_update.emit("Processing complete!")
        self.processing_finished.emit()  # Now this properly refers to the signal
    
    def process_audio_in_chunks(self, audio_track, profile, chunk_size_minutes=10):
        """Process audio in chunks to reduce memory usage"""
        # Convert minutes to milliseconds
        chunk_size_ms = chunk_size_minutes * 60 * 1000
        
        # If audio is shorter than chunk size, process normally
        if len(audio_track) <= chunk_size_ms:
            return self.process_audio_efficiently(audio_track, profile)
        
        # Otherwise, process in chunks and concatenate
        result_chunks = []
        total_length = len(audio_track)
        chunk_count = (total_length + chunk_size_ms - 1) // chunk_size_ms  # Ceiling division
        
        self.progress_update.emit(f"Processing large audio in {chunk_count} chunks")
        
        for i in range(chunk_count):
            if self.stop_requested:
                return None
                
            start = i * chunk_size_ms
            end = min((i+1) * chunk_size_ms, total_length)
            
            self.progress_update.emit(f"Processing chunk {i+1}/{chunk_count} ({start/1000:.1f}s to {end/1000:.1f}s)")
            
            # Extract and process this chunk
            chunk = audio_track[start:end]
            processed_chunk = self.process_audio_efficiently(chunk, profile)
            result_chunks.append(processed_chunk)
            
            # Explicit cleanup
            del chunk
            gc.collect(generation=0)
            
            # Periodically check memory usage and perform deeper GC if needed
            if i % 3 == 0:  # Every third chunk
                self.optimize_memory_usage()
        
        # Concatenate processed chunks
        if not result_chunks:
            return audio_track
            
        self.progress_update.emit(f"Concatenating {len(result_chunks)} processed chunks")
        
        # Concatenate in a memory-efficient way
        result = result_chunks[0]
        for i in range(1, len(result_chunks)):
            if self.stop_requested:
                return None
            result += result_chunks[i]
            # Release memory for the chunk we just added
            result_chunks[i] = None
            
        # Clean up
        result_chunks.clear()
        gc.collect()
        return result
    
    # Reduce garbage collection frequency - only do it when necessary
    def optimize_memory_usage(self):
        """Smart memory management with multiple thresholds"""
        memory = psutil.virtual_memory()
        
        if memory.percent > 90:  # Critical - force immediate collection
            self.progress_update.emit("Memory usage critical, performing full collection")
            # Clear audio cache first
            self.clear_audio_cache()
            gc.collect(generation=2)  # Full collection
            
            # Force release of cached memory by Python and the OS
            if sys.platform == 'win32':
                # On Windows, try to release memory back to the OS
                try:
                    import ctypes
                    ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
                except Exception as e:
                    self.progress_update.emit(f"Warning: Could not release memory: {str(e)}")
            else:
                # On Linux/Unix, sync and try to release cache
                try:
                    subprocess.run(["sync"], timeout=1)  # Sync filesystem to free page cache
                except Exception as e:
                    self.progress_update.emit(f"Warning: Could not sync: {str(e)}")
                    
        elif memory.percent > 80:  # High - normal collection
            gc.collect(generation=1)  # Collect generations 0 and 1
        elif memory.percent > 70:  # Moderate - collect only youngest generation
            gc.collect(generation=0)
    
    def run(self):
        # Set process priority to high
        try:
            if sys.platform == 'win32':
                import psutil
                process = psutil.Process()
                process.nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                import os
                os.nice(-10)  # Lower value = higher priority on Unix
        except:
            self.progress_update.emit("Could not set process priority")  # This line was indented incorrectly

        try:
            self.progress_update.emit("Starting processing files...")
            self.start_time = time.time()
            self.total_tracks = self.count_tracks_to_process()
            self.tracks_processed = 0
            self.process_directory()
            self.finish_processing()  # This already emits processing_finished signal
            
        except Exception as e:
            self.progress_update.emit(f"Error: {str(e)}")
            self.processing_finished.emit()

    # Fix progress bar accuracy in count_tracks_to_process method
    def count_tracks_to_process(self):
        """Count total number of tracks to process for progress estimation"""
        total = 0
        for root, dirs, files in os.walk(self.dir_path):
            if "/edited" in root.replace("\\", "/"):
                continue
                
            # Get all audio files in directory
            audio_files = [f for f in files if f[-4:].lower() == INPUT_FORMAT.lower()]
            
            # Count unique track prefixes
            prefixes = set()
            for f in audio_files:
                if "_" in f:
                    prefix_end = f.rfind("_")
                    if prefix_end > 4:
                        prefix = f[:prefix_end]
                        prefixes.add(prefix)
            
            # Add number of unique prefixes to total
            total += len(prefixes)
            
        return max(1, total)  # Ensure at least 1 to avoid division by zero
        
    def analyze_tracks_for_batch_normalization(self):
        """Analyze tracks to find reference levels for batch normalization"""
        try:
            import math
            reference_levels = {
                'peak': -float('inf'),
                'rms': -float('inf'),
                'lufs': -float('inf')
            }
            
            # Scan a sample of tracks to determine levels
            sample_track_count = 0
            max_sample_tracks = 3  # Limit how many tracks we analyze for performance
            
            for root, dirs, files in os.walk(self.dir_path):
                if "/edited" in root.replace("\\", "/") or sample_track_count >= max_sample_tracks:
                    continue
                    
                # Get audio files
                audio_files = [f for f in files if f[-4:].lower() == INPUT_FORMAT.lower()]
                
                # Sample some files
                sample_files = audio_files[:min(2, len(audio_files))]
                for file in sample_files:
                    try:
                        audio = AudioSegment.from_file(os.path.join(root, file))
                        
                        # Analyze peak level
                        peak_amp = max(abs(s) for s in audio.get_array_of_samples())
                        max_possible = 2**(audio.sample_width * 8 - 1)
                        peak_db = 20 * math.log10(peak_amp / max_possible) if peak_amp > 0 else -80
                        reference_levels['peak'] = max(reference_levels['peak'], peak_db)
                        
                        # Analyze RMS level
                        if audio.rms > 0:
                            rms_db = 20 * math.log10(audio.rms / max_possible)
                            reference_levels['rms'] = max(reference_levels['rms'], rms_db)
                        
                        # Skip LUFS analysis for performance (requires pyloudnorm)
                        
                        sample_track_count += 1
                        if sample_track_count >= max_sample_tracks:
                            break
                            
                    except Exception as e:
                        self.progress_update.emit(f"Warning: Error analyzing {file}: {str(e)}")
                        
            self.progress_update.emit(f"Analyzed {sample_track_count} files for batch normalization")
            
            # Add offsets to bring everything to target levels
            reference_levels['peak_offset'] = -3.0 - reference_levels['peak']  # Target -3 dB
            reference_levels['rms_offset'] = -18.0 - reference_levels['rms']  # Target -18 dB
            reference_levels['lufs_offset'] = -14.0 - reference_levels['lufs']  # Target -14 LUFS
            
            # Add safety checks for infinite values
            if not np.isfinite(reference_levels['peak_offset']):
                reference_levels['peak_offset'] = 0.0
            if not np.isfinite(reference_levels['rms_offset']):
                reference_levels['rms_offset'] = 0.0
            if not np.isfinite(reference_levels['lufs_offset']):
                reference_levels['lufs_offset'] = 0.0  # Skip LUFS normalization if we can't get a valid measurement
                
            # Cap gain values to prevent extreme changes
            reference_levels['peak_offset'] = max(min(reference_levels['peak_offset'], 20.0), -20.0)
            reference_levels['rms_offset'] = max(min(reference_levels['rms_offset'], 20.0), -20.0)
            reference_levels['lufs_offset'] = max(min(reference_levels['lufs_offset'], 20.0), -20.0)
            
            return reference_levels
            
        except Exception as e:
            self.progress_update.emit(f"Error in batch normalization analysis: {str(e)}")
            return None
        
        
    def apply_dynamic_processing(self, audio_track, settings):
        """Apply a blended compressor-limiter for smooth dynamic control with vectorized operations"""
        with self.measure_time("dynamic_processing"):
            try:
                if not settings.get("enabled", False):
                    return audio_track
                    
                import numpy as np
                
                # Get settings
                comp_settings = settings.get("compressor", {})
                limit_settings = settings.get("limiter", {})
                
                # Convert to numpy array for processing
                samples = np.array(audio_track.get_array_of_samples())
                sample_width = audio_track.sample_width
                max_value = float(2 ** (8 * sample_width - 1))
                samples_float = samples.astype(np.float32) / max_value
                
                # Get compressor settings
                comp_threshold = 10 ** (comp_settings.get("threshold", -18.0) / 20.0)
                comp_ratio = comp_settings.get("ratio", 2.5)
                comp_attack_ms = comp_settings.get("attack", 20.0)
                comp_release_ms = comp_settings.get("release", 250.0)
                
                # Convert time constants to samples
                sample_rate = audio_track.frame_rate
                attack_samples = int(comp_attack_ms * sample_rate / 1000.0)
                release_samples = int(comp_release_ms * sample_rate / 1000.0)
                
                # Attack and release coefficients
                attack_coef = np.exp(-1.0 / max(1, attack_samples))
                release_coef = np.exp(-1.0 / max(1, release_samples))
                
                # Process in chunks to improve cache efficiency
                CHUNK_SIZE = 100000  # Adjust based on CPU cache
                processed = np.zeros_like(samples_float)
                
                # Pre-allocate envelope array for entire signal (more efficient)
                envelope = np.zeros_like(samples_float)
                
                # First pass: calculate envelope
                env_val = 0.0
                for i in range(len(samples_float)):
                    # Get current sample value (use absolute for envelope)
                    current = abs(samples_float[i])
                    
                    # Update envelope (peak detector with attack/release)
                    if current > env_val:
                        # Attack phase
                        env_val = attack_coef * env_val + (1.0 - attack_coef) * current
                    else:
                        # Release phase
                        env_val = release_coef * env_val + (1.0 - release_coef) * current
                        
                    envelope[i] = env_val
                
                # Second pass: apply compression based on envelope
                gain_reduction = np.ones_like(samples_float)
                mask = envelope > comp_threshold
                
                # Vectorized gain calculation
                if np.any(mask):
                    # Calculate gain reduction in dB
                    gain_db = np.zeros_like(envelope)
                    gain_db[mask] = (comp_settings.get("threshold", -18.0) -
                                   (comp_settings.get("threshold", -18.0) + 
                                    20.0 * np.log10(envelope[mask] / comp_threshold)) / comp_ratio)
                    
                    # Convert from dB to linear
                    gain_reduction[mask] = np.power(10, gain_db[mask] / 20.0)
                    
                    # Apply compression
                    processed = samples_float * gain_reduction
                else:
                    processed = samples_float
                
                # Limiter stage
                limit_threshold = 10 ** (limit_settings.get("threshold", -1.0) / 20.0)
                limit_release_samples = int(limit_settings.get("release", 50.0) * sample_rate / 1000.0)
                
                # Find peaks that exceed limiter threshold
                scale_factor = np.ones_like(processed)
                mask = np.abs(processed) > limit_threshold
                
                if np.any(mask):
                    # Calculate limiting
                    scale_factor[mask] = limit_threshold / np.abs(processed[mask])
                    
                    # Apply smooth release (this part is still a loop due to sample dependencies)
                    for i in range(1, len(scale_factor)):
                        if scale_factor[i] > scale_factor[i-1]:
                            # Apply release smoothing
                            release_factor = np.exp(-1.0 / max(1, limit_release_samples))
                            scale_factor[i] = scale_factor[i-1] + (scale_factor[i] - scale_factor[i-1]) * (1.0 - release_factor)
                    
                    # Apply limiting
                    processed *= scale_factor
                
                # Convert back to integer samples
                processed_samples = (processed * max_value).astype(samples.dtype)
                
                # Create new audio segment
                processed_audio = audio_track._spawn(processed_samples.tobytes())
                
                # Add memory check before returning
                self.optimize_memory_usage()
                return processed_audio
            
            except Exception as e:
                self.progress_update.emit(f"Warning: Dynamic processing error: {str(e)}")
                return audio_track  # Return original if there's an error
    
    def process_directory(self):
        chosen_dir_path = self.dir_path
        if "edited" not in os.listdir(chosen_dir_path):
            os.mkdir(f"{chosen_dir_path}/edited")
        
        # If batch normalization is enabled, analyze all tracks first
        self.batch_reference_levels = None
        if self.app.batch_normalize:
            self.progress_update.emit("Analyzing tracks for batch normalization...")
            self.batch_reference_levels = self.analyze_tracks_for_batch_normalization()
            self.progress_update.emit(f"Batch reference levels determined: {self.batch_reference_levels}")
        
        # Process all subdirectories in parallel when possible
        all_subdirs = [folder for folder in os.walk(chosen_dir_path) 
                      if "/edited" not in folder[0].replace("\\", "/")]
        
        # Skip processing if stopped
        if self.stop_requested:
            return
            
        # Set initial progress
        self.progress_bar.emit(0)
        
        # Process folders in parallel if possible
        if len(all_subdirs) > 1 and self.processor_count > 1:
            self.progress_update.emit(f"Processing {len(all_subdirs)} folders using {self.processor_count} processors")
            
            # Use a smaller number of workers for folder-level processing
            folder_workers = max(1, min(self.processor_count // 2, len(all_subdirs)))
            
            with conc.ThreadPoolExecutor(max_workers=folder_workers) as executor:
                # Submit all folders for processing
                futures = {executor.submit(self.edit_tracks, folder, chosen_dir_path): folder[0] 
                          for folder in all_subdirs}
                
                # Monitor completion
                for future in concurrent.futures.as_completed(futures):
                    if self.stop_requested:
                        executor.shutdown(wait=False)
                        return
                        
                    try:
                        future.result()  # Get result to capture any exceptions
                        folder_path = futures[future]
                        self.progress_update.emit(f"Completed processing folder: {os.path.basename(folder_path)}")
                    except Exception as e:
                        self.progress_update.emit(f"Error processing folder: {str(e)}")
        else:
            # Process sequentially if only one folder or one processor
            for folder in all_subdirs:
                if self.stop_requested:
                    return
                self.edit_tracks(folder, chosen_dir_path)
            
    def apply_limiter(self, audio_track, threshold=-1.0, release_time=50):
        """Apply a simple limiter to prevent peaks above threshold"""
        try:
            # Convert to numpy array for processing
            samples = np.array(audio_track.get_array_of_samples())
            sample_width = audio_track.sample_width
            
            # Convert to float in range [-1, 1]
            max_value = float(2 ** (8 * sample_width - 1))
            samples_float = samples.astype(float) / max_value
            
            # Find peaks that exceed threshold
            threshold_linear = 10 ** (threshold / 20.0)
            scale_factor = np.ones_like(samples_float)
            mask = np.abs(samples_float) > threshold_linear
            
            if np.any(mask):
                # Apply limiting only where needed
                scale_factor[mask] = threshold_linear / np.abs(samples_float[mask])
                
                # Apply release to smooth transitions
                for i in range(1, len(scale_factor)):
                    if scale_factor[i] > scale_factor[i-1]:
                        scale_factor[i] = scale_factor[i-1] + min(1.0, (1.0 - scale_factor[i-1]) / release_time)
                
                # Apply scaling
                samples_float *= scale_factor
            
            # Convert back to integer samples
            limited_samples = (samples_float * max_value).astype(samples.dtype)
            
            # Create new audio segment
            limited_audio = audio_track._spawn(limited_samples.tobytes())
            return limited_audio
            
        except Exception as e:
            self.progress_update.emit(f"Warning: Limiter error: {str(e)}")
            return audio_track  # Return original if there's an error

    def edit_tracks(self, tup, chosen_dir_path):
        dir_path, _, filenames = tup
        dir_path = dir_path.replace("\\", "/")
        
        # Skip invalid paths
        if not filenames or dir_path[-7:] == "edited/": return
        filenames = [f for f in filenames if "0" in f and f[-4:].lower() == INPUT_FORMAT.lower()]
        if not filenames: return
        
        # Ensure dir_path format
        if dir_path[-1] != "/": dir_path += "/"
        
        # Track identification
        prefix_end_index = filenames[0].rfind("_") - 4
        if prefix_end_index < 0: return    
        prefix = filenames[0][:prefix_end_index]
        
        # Get tracks to process
        tracks = {dir_path + tr[: prefix_end_index + 4] for tr in filenames if tr.startswith(prefix)}
        if not tracks: return
        
        # REMOVED the first sequential processing loop that was causing double processing
        
        # Process tracks either in parallel or sequentially based on configuration
        # Process tracks either in parallel or sequentially based on configuration
        if len(tracks) > 1 and self.processor_count > 1:
            # Parallel processing with minimal overhead
            # Determine versions for each track
            track_versions = {}
            for t in tracks:
                ver = 1
                while os.path.exists(f"{self.get_export_name(t, chosen_dir_path, prefix, ver)} - Segment 1.mp3"):
                    ver += 1
                track_versions[t] = ver
            
            # Use thread pool with optimal chunk size to reduce overhead
            chunk_size = max(1, len(tracks) // self.processor_count)
            
            with conc.ThreadPoolExecutor(max_workers=self.processor_count) as executor:
                futures = {}
                for t in tracks:
                    ver = track_versions[t]
                    # Add more context to each task
                    future = executor.submit(
                        self.process_track, 
                        t, dir_path, chosen_dir_path, prefix, filenames, ver
                    )
                    futures[future] = t
                    
                # Monitor futures with timeout to prevent deadlocks
                completed = 0
                pending = set(futures.keys())
                
                while pending and not self.stop_requested:
                    # Process completed futures
                    just_completed, pending = concurrent.futures.wait(
                        pending, 
                        timeout=5.0,  # Check every 5 seconds
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    
                    for future in just_completed:
                        if self.stop_requested:
                            break
                            
                        try:
                            future.result()  # Get result or raise exception
                            track = futures.get(future)
                            self.progress_update.emit(f"Completed track: {os.path.basename(track)}")
                            
                            completed += 1
                            self.tracks_processed += 1
                            progress = int((self.tracks_processed / self.total_tracks) * 100)
                            self.progress_bar.emit(progress)
                            
                            # Explicit memory cleanup after each track
                            gc.collect(generation=0)
                        except Exception as e:
                            self.progress_update.emit(f"Error: {e}")
                    
                    # Check if we should cancel remaining futures
                    if self.stop_requested:
                        self.progress_update.emit("Canceling remaining tasks...")
                        for future in pending:
                            future.cancel()
                        break
                        
                # If stopped, make sure we exit the executor
                if self.stop_requested:
                    executor.shutdown(wait=False)
                    return
        else:
            # Sequential processing
            for track in tracks:
                if self.stop_requested: return
                
                # Determine version
                ver = 1
                # Check for existing versions directly in the chosen directory
                while os.path.exists(os.path.join(chosen_dir_path, f"{track}_v{ver} - Segment 1.mp3")):
                    ver += 1
                    
                # Create version info dictionary
                version_info = {
                    'name': f'v{ver}',
                    'profile': self.app.track_profile_assignments.get(track),
                    'description': f'Version {ver}'
                }
                    
                self.process_track(track, dir_path, chosen_dir_path, prefix, filenames, version_info)
                self.tracks_processed += 1
                self.progress_bar.emit(int((self.tracks_processed / self.total_tracks) * 100))
        
        # Count total tracks for this folder
        folder_track_count = len(tracks)
        # Update progress once per folder
        progress_pct = int((self.tracks_processed / self.total_tracks) * 100)
        self.progress_bar.emit(progress_pct)

        # Optimize memory management
    def process_audio_efficiently(self, audio_track, profile):
        """Apply audio processing with enhanced normalization"""
        
        # Ensure profile has the new normalize format
        profile = self.ensure_normalize_format(profile)
        normalize_settings = profile.get('normalize', {"enabled": True})
        
        # Apply normalization if enabled
        if normalize_settings.get("enabled", True):
            method = normalize_settings.get("method", "peak")
            target_level = normalize_settings.get("target_level", -1.0)
            headroom = normalize_settings.get("headroom", 2.0)
            
            # Use batch reference levels if available
            if self.app.batch_normalize and hasattr(self, 'batch_reference_levels') and self.batch_reference_levels:
                if method == "peak":
                    gain_db = self.batch_reference_levels.get('peak_offset', 0)
                    # Safety check to prevent extreme gain values
                    if np.isfinite(gain_db) and abs(gain_db) < 40:
                        self.progress_update.emit(f"Applying batch normalization gain: {gain_db:.1f} dB")
                        audio_track = audio_track + gain_db
                    else:
                        self.progress_update.emit(f"Skipping extreme gain adjustment: {gain_db} dB")
                elif method == "rms":
                    gain_db = self.batch_reference_levels.get('rms_offset', 0)
                    # Safety check to prevent extreme gain values
                    if np.isfinite(gain_db) and abs(gain_db) < 40:
                        self.progress_update.emit(f"Applying batch normalization gain: {gain_db:.1f} dB")
                        audio_track = audio_track + gain_db
                    else:
                        self.progress_update.emit(f"Skipping extreme gain adjustment: {gain_db} dB")
                elif method == "lufs":
                    gain_db = self.batch_reference_levels.get('lufs_offset', 0)
                    # Safety check to prevent extreme gain values
                    if np.isfinite(gain_db) and abs(gain_db) < 40:
                        self.progress_update.emit(f"Applying batch normalization gain: {gain_db:.1f} dB")
                        audio_track = audio_track + gain_db
                    else:
                        self.progress_update.emit(f"Skipping extreme gain adjustment: {gain_db} dB")
            else:
                # Regular per-track normalization
                if method == "peak":
                    # Use standard PyDub normalization with custom headroom
                    audio_track = effects.normalize(audio_track, headroom=headroom)
                elif method == "rms":
                    # Apply RMS normalization
                    audio_track = self.normalize_rms(audio_track, target_level)
                elif method == "lufs":
                    # Apply LUFS normalization (requires pyloudnorm)
                    audio_track = self.normalize_lufs(audio_track, target_level)
            
            # Apply limiter after normalization if enabled
            if normalize_settings.get("limiter", False):
                limiter_threshold = normalize_settings.get("limiter_threshold", -0.5)
                audio_track = self.apply_limiter(audio_track, threshold=limiter_threshold)
        
        # Apply dynamic processing (compressor-limiter)
        dynamic_settings = profile.get('dynamic_processing', {"enabled": False})
        audio_track = self.apply_dynamic_processing(audio_track, dynamic_settings)

        # Apply gain
        gain_db = profile.get('gain', 0)
        if gain_db != 0:
            audio_track = audio_track + gain_db
        
        # Apply filters
        if profile.get('low_pass', False) or profile.get('high_pass', False):
            # Batch filter operations to reduce memory allocation
            if profile.get('low_pass', False):
                audio_track = effects.low_pass_filter(audio_track, profile.get('low_pass_freq', 8000))
            if profile.get('high_pass', False):
                audio_track = effects.high_pass_filter(audio_track, profile.get('high_pass_freq', 200))
        
        # Apply panning if needed
        pan = profile.get('pan', 0)
        if pan != 0 and audio_track.channels == 2:
            left, right = audio_track.split_to_mono()
            if pan < 0:  # Left
                right = right + (pan * 10)
            else:  # Right
                left = left + (-pan * 10)
            audio_track = AudioSegment.from_mono_audiosegments(left, right)
            
        return audio_track
        
    def normalize_rms(self, audio_track, target_level=-18.0):
        """Vectorized RMS normalization using NumPy"""
        try:
            import numpy as np
            
            # Get samples as numpy array
            samples = np.array(audio_track.get_array_of_samples())
            
            # Calculate RMS more efficiently with numpy
            rms = np.sqrt(np.mean(np.square(samples.astype(np.float64))))
            
            if rms <= 0:  # Avoid division by zero
                self.progress_update.emit("Warning: Zero RMS detected, skipping RMS normalization")
                return audio_track
                
            # Get reference level based on bit depth
            if audio_track.sample_width == 4 and hasattr(audio_track, 'array_type') and audio_track.array_type == 'float32':
                # For float32 files, reference level is 1.0 rather than max int value
                max_possible = 1.0
            else:
                # For integer PCM formats
                max_possible = float(2 ** (audio_track.sample_width * 8 - 1))
            
            # Calculate current level in dB
            current_db = 20 * np.log10(rms / max_possible) if max_possible > 0 else -80
            
            # Calculate gain needed
            gain_db = target_level - current_db
            
            # Apply gain
            self.progress_update.emit(f"Applying RMS normalization: {gain_db:.1f} dB gain")
            return audio_track + gain_db
            
        except Exception as e:
            self.progress_update.emit(f"Error in RMS normalization: {str(e)}")
            return audio_track

    def normalize_lufs(self, audio_track, target_lufs=-14.0):
        """Normalize to LUFS (Loudness Units Full Scale)"""
        try:
            # Check if pyloudnorm is installed
            try:
                import pyloudnorm as pyln
                import numpy as np
            except ImportError:
                self.progress_update.emit("Warning: pyloudnorm not installed, falling back to peak normalization")
                return effects.normalize(audio_track)
            
            # Convert audio to numpy array for analysis
            samples = np.array(audio_track.get_array_of_samples())
            
            # Handle float32 format correctly
            if audio_track.sample_width == 4:  # 32-bit samples (likely float32)
                # If samples are already float32, we may just need to reshape
                if samples.dtype.kind == 'f':
                    samples_float = samples  # Already float
                else:
                    # Convert if needed (unlikely for float32 recordings)
                    max_value = float(2 ** 31)  # For 32-bit integer PCM
                    samples_float = samples.astype(np.float32) / max_value
            else:
                # Standard conversion for integer PCM formats
                max_value = float(2 ** (8 * audio_track.sample_width - 1))
                samples_float = samples.astype(np.float32) / max_value
            
            # If stereo, reshape to 2D array
            channels = 2 if audio_track.channels == 2 else 1
            if channels == 2:
                samples_float = samples_float.reshape(-1, 2)
            else:
                samples_float = samples_float.reshape(-1, 1)
            
            # Create meter
            meter = pyln.Meter(audio_track.frame_rate)
            
            # Measure current loudness
            current_lufs = meter.integrated_loudness(samples_float)
            
            # Calculate gain adjustment
            gain_db = target_lufs - current_lufs
            
            self.progress_update.emit(f"Applying LUFS normalization: {gain_db:.1f} dB gain")
            return audio_track + gain_db
            
        except Exception as e:
            self.progress_update.emit(f"Error in LUFS normalization: {str(e)}")
            # Fall back to standard normalization
            return effects.normalize(audio_track)
    
    def ensure_normalize_format(self, profile):
        """Ensure the normalize attribute has the new dictionary format"""
        if isinstance(profile.get('normalize'), bool):
            # Convert old boolean format to new dictionary format
            normalize_enabled = profile['normalize']
            profile['normalize'] = {
                "enabled": normalize_enabled,
                "target_level": -1.0,
                "headroom": 2.0,
                "method": "peak"
            }
        elif profile.get('normalize') is None:
            # Handle case where normalize key doesn't exist
            profile['normalize'] = {
                "enabled": True,
                "target_level": -1.0,
                "headroom": 2.0,
                "method": "peak"
            }
        return profile
        
    def apply_limiter(self, audio_track, threshold=-1.0, release_time=50):
        """Apply a simple limiter to prevent peaks above threshold"""
        try:
            # Convert to numpy array for processing
            samples = np.array(audio_track.get_array_of_samples())
            sample_width = audio_track.sample_width
            
            # Convert to float in range [-1, 1]
            max_value = float(2 ** (8 * sample_width - 1))
            samples_float = samples.astype(float) / max_value
            
            # Find peaks that exceed threshold
            threshold_linear = 10 ** (threshold / 20.0)
            scale_factor = np.ones_like(samples_float)
            mask = np.abs(samples_float) > threshold_linear
            
            if np.any(mask):
                # Apply limiting only where needed
                scale_factor[mask] = threshold_linear / np.abs(samples_float[mask])
                
                # Apply release to smooth transitions
                for i in range(1, len(scale_factor)):
                    if scale_factor[i] > scale_factor[i-1]:
                        scale_factor[i] = scale_factor[i-1] + min(1.0, (1.0 - scale_factor[i-1]) / release_time)
                
                # Apply scaling
                samples_float *= scale_factor
            
            # Convert back to integer samples
            limited_samples = (samples_float * max_value).astype(samples.dtype)
            
            # Create new audio segment
            limited_audio = audio_track._spawn(limited_samples.tobytes())
            return limited_audio
            
        except Exception as e:
            self.progress_update.emit(f"Warning: Limiter error: {str(e)}")
            return audio_track  # Return original if there's an error
            
    def process_track(self, track, dir_path, chosen_dir_path, prefix, filenames, version_info=None):
        """Process a single track using Kirtan vocals as the reference for segmentation"""
        # Clear the audio_tracks dictionary for this track
        self.audio_tracks.clear()
        
        try:
            # Validate version_info is a dictionary if provided
            if version_info is not None and not isinstance(version_info, dict):
                self.progress_update.emit(f"Warning: Invalid version info format for {track}, using default")
                version_info = None
            
            # Get version name safely
            version_name = version_info.get('name') if isinstance(version_info, dict) else None
            profile_name = version_info['profile'] if version_info else None
            
            # Update progress
            if version_name:
                self.progress_update.emit(f"Processing {track} - Version: {version_name}")
            else:
                self.progress_update.emit(f"Processing {track}")
            
            # First, identify which input channel is assigned to Kirtan profile
            kirtan_channel = None
            for channel, profile in self.app.track_profile_assignments.items():
                if profile == "Kirtan (Vocals)":
                    kirtan_channel = channel
                    self.progress_update.emit(f"Found Kirtan vocal channel: {kirtan_channel}")
                    break
            
            if not kirtan_channel:
                self.progress_update.emit("Error: No input channel assigned to Kirtan (Vocals) profile!")
                return
            
            # Find the Kirtan vocal file
            track_prefix = self.get_track_name(track, dir_path)
            kirtan_file = None
            for filename in filenames:
                if filename.startswith(track_prefix) and filename.endswith(f"_{kirtan_channel}{INPUT_FORMAT}"):
                    kirtan_file = dir_path + filename
                    break
            
            if not kirtan_file:
                self.progress_update.emit(f"Error: Could not find Kirtan vocal file for {track_prefix}")
                return
            
            # Load and analyze Kirtan vocals first
            self.progress_update.emit(f"Loading Kirtan vocal track as segmentation reference")
            kirtan_audio = self.get_cached_audio(kirtan_file)
            
            # Detect segments using original approach
            self.progress_update.emit("Detecting segments using silence detection")
            
            # Get silence detection parameters 
            min_silence_len = self.app.min_silence
            seek_step = self.app.seek_step
            
            # Use detect_silence_efficiently function
            auto_detected_segments = self.detect_silence_efficiently(
                kirtan_audio, 
                min_silence_len=min_silence_len, 
                silence_thresh=self.app.silence_threshold, 
                seek_step=seek_step
            )
            
            if not auto_detected_segments:
                self.progress_update.emit("No segments detected in Kirtan vocals, aborting processing")
                return
            
            # Process segments
            min_time_between_vaaris = self.app.min_time_between_segments
            min_vaari_length = self.app.min_segment_length * 60 * 1000
            dropout = self.app.dropout * 60 * 1000
            
            # Apply segment refinement logic
            final_segments = []
            prev_end = 0
            
            for start, end in auto_detected_segments:
                if end - start < dropout:
                    self.progress_update.emit(f"Dropping segment {self.format_time(start)}-{self.format_time(end)} (too short)")
                    continue
                    
                if final_segments:
                    prev_length = final_segments[-1][1] - final_segments[-1][0]
                    
                    if (prev_length < min_vaari_length or 
                        start - prev_end < min_time_between_vaaris):
                        self.progress_update.emit(f"Merging with previous segment (end now {self.format_time(end)})")
                        final_segments[-1][1] = end
                    else:
                        self.progress_update.emit(f"New segment {self.format_time(start)}-{self.format_time(end)}")
                        final_segments.append([start, end])
                else:
                    self.progress_update.emit(f"First segment {self.format_time(start)}-{self.format_time(end)}")
                    final_segments.append([start, end])
                    
                prev_end = end
            
            if len(final_segments) == 0:
                self.progress_update.emit("No segments found after filtering, aborting")
                return
                
            # Handle the last-segment-too-short case
            if len(final_segments) >= 2:
                ls, le = final_segments[-1]
                if le - ls < 10 * 60 * 1000:  # 10 minutes
                    self.progress_update.emit(f"Last segment too short, merging with previous")
                    final_segments = final_segments[:-1]
                    final_segments[-1][1] = le
            
            self.progress_update.emit(f"Found {len(final_segments)} segments after processing")
            
            # Convert segments to tuples
            segments = [(start, end) for start, end in final_segments]
            
            # Process all input channels using self.audio_tracks (initialized in __init__)
            
            # Get track prefix
            track_prefix = self.get_track_name(track, dir_path)
            self.progress_update.emit(f"Processing {track_prefix}")
            
            # Find and group all files for this track by input channel
            track_files = {}
            for filename in filenames:
                if self.stop_requested:
                    return
                    
                if filename.startswith(track_prefix):
                    # Extract input channel name
                    input_channel = filename[filename.rfind('_')+1:filename.rfind('.')]
                    
                    # Handle split files
                    if '-' in input_channel and input_channel[-5:-1].isdigit():
                        base_input = input_channel.split('-')[0]
                        if base_input not in track_files:
                            track_files[base_input] = []
                        track_files[base_input].append(filename)
                    else:
                        if input_channel not in track_files:
                            track_files[input_channel] = []
                        track_files[input_channel].append(filename)
            
            # Process each channel
            for channel, files in track_files.items():
                if self.stop_requested:
                    return
                
                # Use version-specific profile if provided, otherwise use default assignment
                if version_info and profile_name:
                    profile = self.app.profiles.get(profile_name)
                else:
                    profile_name = self.app.track_profile_assignments.get(channel)
                    profile = self.app.profiles.get(profile_name)
                
                if not profile:
                    continue
                
                # Import and process audio
                try:
                    if len(files) == 1:
                        audio_file = dir_path + files[0]
                        audio_track = self.get_cached_audio(audio_file)
                    else:
                        files.sort()
                        self.progress_update.emit(f"Importing {len(files)} segments for {channel}")
                        
                        audio_parts = []
                        for f in files:
                            if self.stop_requested:
                                return
                            file_path = dir_path + f
                            audio_part = AudioSegment.from_file(file_path)
                            audio_parts.append(audio_part)
                                
                        if audio_parts:
                            audio_track = sum(audio_parts, AudioSegment.empty())
                        else:
                            continue
                    
                    # Process with profile
                    audio_track = self.process_audio_in_chunks(audio_track, profile)
                    self.audio_tracks[channel] = audio_track
                    
                except Exception as e:
                    self.progress_update.emit(f"Error importing {channel}: {str(e)}")
                    continue
            
            # Mix all tracks AFTER all channels have been processed
            self.progress_update.emit(f"Mixing audio tracks")
            if self.audio_tracks and kirtan_channel in self.audio_tracks:
                audio = self.audio_tracks[kirtan_channel]  # Start with Kirtan track
                for channel, track_audio in self.audio_tracks.items():
                    if channel != kirtan_channel:  # Mix in other tracks
                        audio = audio.overlay(track_audio)
                
                # Export segments
                self.progress_update.emit(f"Exporting segments based on Kirtan vocals")
                self.export_segments(audio, segments, track, chosen_dir_path, prefix, version_info)
            else:
                self.progress_update.emit(f"Error: Kirtan channel {kirtan_channel} not found in processed tracks")
            
            # Clean up
            self.audio_tracks.clear()
            self.optimize_memory_usage()
                
        except Exception as e:
            self.progress_update.emit(f"Error processing track: {str(e)}")
            import traceback
            self.progress_update.emit(traceback.format_exc())

    def detect_vocal_segments(self, kirtan_audio):
        """Detect segments based on Kirtan vocals using pydub's silence detection"""
        try:
            # Get the audio's dBFS (decibels relative to full scale)
            dbfs = kirtan_audio.dBFS
            
            # Use the silence threshold as an offset from dBFS
            # If silence_threshold is positive, we subtract it from dBFS (making threshold lower)
            # If silence_threshold is negative, we add its absolute value to dBFS (making threshold higher)
            silence_thresh = dbfs - self.app.silence_threshold
            
            self.progress_update.emit(f"Using silence threshold: {silence_thresh:.1f} dB (dBFS: {dbfs:.1f} dB, offset: {self.app.silence_threshold} dB)")
            
            # Use pydub's silence detection with user-configurable parameters
            segments = silence.detect_nonsilent(
                kirtan_audio,
                min_silence_len=self.app.min_silence,
                silence_thresh=silence_thresh,
                seek_step=self.app.seek_step
            )
            
            # Convert segments to the expected format (list of tuples)
            segments = [(start, end) for start, end in segments]
            
            # Filter out segments that are too short
            min_segment_duration = self.app.min_segment_length * 60 * 1000  # minutes to milliseconds
            filtered_segments = []
            
            for start, end in segments:
                duration = end - start
                if duration < min_segment_duration:
                    self.progress_update.emit(f"Dropping segment {self.format_time(start)}-{self.format_time(end)} (too short)")
                    continue
                filtered_segments.append((start, end))
            
            # Merge segments that are too close together
            if filtered_segments:
                min_gap = self.app.min_time_between_segments  # already in milliseconds
                merged_segments = [filtered_segments[0]]
                
                for seg in filtered_segments[1:]:
                    prev_end = merged_segments[-1][1]
                    curr_start = seg[0]
                    
                    if curr_start - prev_end < min_gap:
                        merged_segments[-1] = (merged_segments[-1][0], seg[1])
                    else:
                        merged_segments.append(seg)
                
                segments = merged_segments
            
            # If no segments found, use the entire file
            if not segments:
                self.progress_update.emit("No segments detected, using entire file as one segment")
                segments = [(0, len(kirtan_audio))]
            
            self.progress_update.emit(f"Found {len(segments)} non-silent regions using pydub silence detection")
            
            # Log the first segment for reference
            if segments:
                first_start, first_end = segments[0]
                self.progress_update.emit(f"First segment {self.format_time(first_start)}-{self.format_time(first_end)}")
            
            return segments
            
        except Exception as e:
            self.progress_update.emit(f"Error in vocal segment detection: {str(e)}")
            import traceback
            self.progress_update.emit(traceback.format_exc())
            # Return one segment for the entire file
            return [(0, len(kirtan_audio))]

    def export_segments(self, audio, segments, track, chosen_dir_path, prefix, version_info=None):
        """Export segments based on the detected vocal segments"""
        try:
            # Validate version_info is a dictionary if provided
            if version_info is not None and not isinstance(version_info, dict):
                self.progress_update.emit(f"Warning: Invalid version info format for {track}, using default")
                version_info = None
            
            # Get version info safely
            version_name = version_info.get('name') if isinstance(version_info, dict) else None
            
            # Determine if a version should be added if not already present
            if not version_name:
                # Find the next available version number by checking existing files
                ver = 1
                while os.path.exists(os.path.join(chosen_dir_path, f"{track}_v{ver} - Segment 1.mp3")):
                    ver += 1
                version_name = f"v{ver}"
                self.progress_update.emit(f"Auto-assigned version: {version_name}")
            
            # Create base export path directly in the input folder
            export_path_base = os.path.join(chosen_dir_path, f"{track}_{version_name}")
            
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
                if self.stop_requested:
                    return
                
                # Calculate segment information
                segment_number = i + 1
                segment_filename = os.path.join(os.path.dirname(export_path_base), f"{os.path.basename(export_path_base)} - Segment {segment_number}.mp3")
                self.progress_update.emit(f"Exporting to: {segment_filename}")
                
                # Export the audio segment
                self.export_audio_slice(audio, start, end, segment_filename)
                
                # Save matching JSON metadata file
                segment_metadata = base_metadata.copy()
                segment_metadata['segment_number'] = segment_number
                segment_metadata['segment_range'] = [start, end]
                segment_metadata['duration_ms'] = end - start
                
                # Use the same filename as the MP3 but with .json extension
                json_filename = segment_filename.replace('.mp3', '.json')
                with open(json_filename, 'w') as f:
                    json.dump(segment_metadata, f, indent=2)
                
                duration = (end - start) / 1000  # Convert to seconds
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                
                # Only show minutes if greater than zero
                if minutes > 0:
                    time_display = f"{minutes}:{seconds:02d}"
                else:
                    time_display = f":{seconds:02d}"
                    
                self.progress_update.emit(f"<b>Exported segment {segment_number}: {time_display}</b>")
            
            self.progress_update.emit(f"Created {len(segments)} segments with version {version_name}")
            
        except Exception as e:
            self.progress_update.emit(f"Error exporting segments: {str(e)}")
            import traceback
            self.progress_update.emit(traceback.format_exc())

    def get_track_name(self, track_file_path, dir_path):
        """Returns track name without directory path"""
        return track_file_path[len(dir_path):]
    
    def clear_audio_cache(self):
        """Clear the audio cache to free memory"""
        cache_size_mb = self.audio_cache_size / 1_000_000
        self.progress_update.emit(f"Clearing audio cache ({cache_size_mb:.1f} MB)")
        self.audio_cache.clear()
        self.audio_cache_size = 0
        gc.collect(generation=2)

    def get_export_name(self, track_file_path, chosen_dir_path, prefix, version=None):
        """Returns export path and name for a track, optionally with version"""
        try:
            # Check if track_file_path is an AudioSegment object (not a string path)
            if not isinstance(track_file_path, str):
                # Use a timestamp-based fallback path
                timestamp = int(time.time())
                return f"{chosen_dir_path}/track_{timestamp}_v{version if version else 1}"
                
            path_after_chosen_dir = track_file_path[len(chosen_dir_path)+1:]
            path_after_chosen_dir = path_after_chosen_dir.replace("/", " - ")
            
            # Remove duplicate prefix in name
            if path_after_chosen_dir.count(prefix) > 1:
                end = path_after_chosen_dir.rfind(prefix)
                path_after_chosen_dir = path_after_chosen_dir[:end-3]
            
            # Add version if specified and greater than 1
            if version and version > 1:
                return f"{chosen_dir_path}/{path_after_chosen_dir}_v{version}"
            else:
                return f"{chosen_dir_path}/{path_after_chosen_dir}"
        except Exception as e:
            # If any error occurs, log it and return a safe default path
            self.progress_update.emit(f"Error in get_export_name: {str(e)}")
            # Use a simple name with timestamp as fallback
            timestamp = int(time.time())
            return f"{chosen_dir_path}/track_{timestamp}_v{version if version else 1}"

    def import_audio(self, dir_path, track_name):
        """Import audio file with memory optimization for large files"""
        file_path = track_name
        try:
            self.progress_update.emit(f"Importing {self.get_track_name(track_name, dir_path)}")
            
            # Check file size
            file_size = os.path.getsize(file_path)
            
            # For very large files, use a different approach
            if file_size > 500_000_000:  # 500 MB
                self.progress_update.emit(f"Large file detected ({file_size/1_000_000:.1f} MB), using optimized import")
                
                # Force garbage collection before loading large file
                gc.collect(generation=2)
                
                # Import the file using default pydub method
                audio = AudioSegment.from_file(file_path)
                
                # Immediately after loading, run garbage collection
                gc.collect(generation=0)
                
                return audio
            else:
                # Standard import for smaller files
                return AudioSegment.from_file(file_path)
        except Exception as e:
            self.progress_update.emit(f"Error importing {track_name}: {str(e)}")
            return None

    def export_audio_slice(self, audio, start, end, export_path):
        """Export audio segment with user-configurable padding
        
        Pre-segment padding:
         - NEGATIVE values move the start point EARLIER (pulls back start time before vocals)
         - POSITIVE values move the start point LATER (adds delay before vocals)
        
        Post-segment padding:
         - NEGATIVE values move the end point EARLIER (reduces silence after vocals)
         - POSITIVE values move the end point LATER (adds time after vocals)
        """
        # Create export directory if it doesn't exist
        export_dir = os.path.dirname(export_path)
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
            
        # Log original segment boundaries for debugging
        self.progress_update.emit(f"Original segment: {self.format_time(start)} to {self.format_time(end)} (duration: {self.format_time(end-start)})")
        
        # Debug the pre-padding value
        self.progress_update.emit(f"DEBUG: Pre-padding value: {self.app.pre_segment_padding}")
        
        # Apply user-configured padding
        # For pre-padding, NEGATIVE means move start EARLIER, POSITIVE means move start LATER
        if self.app.pre_segment_padding < 0:
            # Negative pre-padding: move start EARLIER (subtract the absolute value)
            adjustment = abs(self.app.pre_segment_padding) * 1000
            adjusted_start = max(0, start - adjustment)
            self.progress_update.emit(f"DEBUG: Negative pre-padding, moving start EARLIER by {abs(self.app.pre_segment_padding)}s")
        else:
            # Positive pre-padding: move start LATER (add the value)
            adjustment = self.app.pre_segment_padding * 1000
            adjusted_start = max(0, start + adjustment)
            self.progress_update.emit(f"DEBUG: Positive pre-padding, moving start LATER by {self.app.pre_segment_padding}s")
        
        # Note: Positive post_segment_padding means add time after (move end later)
        adjusted_end = min(len(audio), end + self.app.post_segment_padding * 1000)
        
        # Log adjusted segment boundaries for debugging
        self.progress_update.emit(f"Adjusted with padding: {self.format_time(adjusted_start)} to {self.format_time(adjusted_end)} (duration: {self.format_time(adjusted_end-adjusted_start)})")
        self.progress_update.emit(f"Applied padding: Pre={self.app.pre_segment_padding}s, Post={self.app.post_segment_padding}s")
        
        # Ensure the segment isn't too short after adjustments
        if adjusted_end - adjusted_start < 1000:  # Ensure at least 1 second
            self.progress_update.emit("Warning: Segment too short after adjustments, using original boundaries")
            adjusted_start = start
            adjusted_end = end
        
        # Extract the segment with adjusted boundaries
        segment = audio[adjusted_start:adjusted_end]
        
        # Apply fade in/out if enabled
        if self.app.fade_in > 0:
            segment = segment.fade_in(self.app.fade_in * 1000)
        if self.app.fade_out > 0:
            segment = segment.fade_out(self.app.fade_out * 1000)
                
        # Determine optimal export parameters
        mp3_params = {
            "format": "mp3",
            "bitrate": f"{self.app.export_bitrate}k",
            "tags": {
                "album": self.app.export_album if self.app.export_album else "Kirtan Recording",
                "title": os.path.basename(export_path).replace(".mp3", "")
            }
        }
                
        # Try to use multi-threaded encoding if available
        try:
            # Add parameters for ffmpeg/libmp3lame if we're using that backend
            mp3_params["parameters"] = ["-threads", "2"]
        except:
            pass
        
        # Export with optimized parameters
        segment.export(export_path, **mp3_params)
        
        # Log export completion
        self.progress_update.emit(f"Exported: {os.path.basename(export_path)}")
        
        # Memory cleanup
        del segment
        self.optimize_memory_usage()

    def format_time(self, t):
        """Format milliseconds to readable time"""
        mins = t // ONE_MIN
        secs = (t // 1000) % 60
        
        if mins > 59:
            return f"{mins//60}:{mins%60:02}:{secs:02}"
        elif mins > 0:
            return f"{mins}:{secs:02}"
        else:
            return f":{secs:02}"
        
    def stop_processing(self):
        """Signal to stop processing"""
        self.stop_requested = True
        self.progress_update.emit("Stop requested. Aborting processing...")


# Simplified button class
class StyledButton(QPushButton):
    def __init__(self, text, parent=None, primary=False):
        super().__init__(text, parent)
        self.setMinimumHeight(35)
        self.setMinimumWidth(100)
        if primary:
            self.setProperty("primary", "true")

class KirtanProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set default font for the entire application
        app_font = QFont("Arial", 10)
        QApplication.instance().setFont(app_font)
        
        # Load style sheet
        style_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'style.qss')
        if os.path.exists(style_file):
            with open(style_file, 'r') as f:
                style = f.read()
                self.setStyleSheet(style)
                QApplication.instance().setStyleSheet(style)
        
        # App configuration
        self.setWindowTitle("Kirtan Processor")
        self.setMinimumSize(1500, 1000)
        
        self.settings_loaded_from_track = False
        
        # Add batch normalization setting
        self.batch_normalize = False  # Default: no batch normalization
        
        # Default profiles
        self.profiles = {
            "Kirtan (Vocals)": {
                "gain": 0,
                "normalize": {
                    "enabled": True,         # Whether normalization is active
                    "target_level": -3.0,    # Target dB level for peaks (negative number)
                    "headroom": 2.0,         # dB of headroom to preserve
                    "method": "peak"         # 'peak', 'rms', or 'lufs'
                },
                "dynamic_processing": {
                    "enabled": True,  # Enable for vocals
                    "compressor": {
                        "threshold": -18.0,  # dB
                        "ratio": 2.5,        # compression ratio (e.g., 2.5:1)
                        "attack": 20.0,      # ms
                        "release": 250.0,    # ms
                    },
                    "limiter": {
                        "threshold": -1.0,   # dB
                        "release": 50.0,     # ms
                    }
                },
                "low_pass": False,
                "low_pass_freq": 8000,
                "low_pass_db": -12,
                "high_pass": True,
                "high_pass_freq": 120,
                "high_pass_db": -12,
                "pan": 0
            },
            "Tabla": {
                "gain": 0,
                "normalize": {
                    "enabled": True,         # Whether normalization is active
                    "target_level": -4.0,    # Target dB level for peaks (negative number)
                    "headroom": 2.0,         # dB of headroom to preserve
                    "method": "peak"         # 'peak', 'rms', or 'lufs'
                },
                "dynamic_processing": {
                    "enabled": True,  # Enable for tabla
                    "compressor": {
                        "threshold": -20.0,  # dB
                        "ratio": 3.5,        # compression ratio (e.g., 3.5:1)
                        "attack": 10.0,      # ms
                        "release": 150.0,    # ms
                    },
                    "limiter": {
                        "threshold": -0.5,   # dB
                        "release": 40.0,     # ms
                    }
                },
                "low_pass": False,
                "low_pass_freq": 10000,
                "low_pass_db": -12,
                "high_pass": True,
                "high_pass_freq": 80,
                "high_pass_db": -12,
                "pan": 0
            },
            "Sangat (Ambient)": {
                "gain": 0,
                "normalize": {
                    "enabled": True,         # Whether normalization is active
                    "target_level": -2.0,    # Target dB level for peaks (negative number)
                    "headroom": 2.0,         # dB of headroom to preserve
                    "method": "peak"         # 'peak', 'rms', or 'lufs'
                },
                "dynamic_processing": {
                    "enabled": True,  # Enable for ambient
                    "compressor": {
                        "threshold": -24.0,  # dB
                        "ratio": 2.0,        # compression ratio (e.g., 2:1)
                        "attack": 40.0,      # ms
                        "release": 400.0,    # ms
                    },
                    "limiter": {
                        "threshold": -1.5,   # dB
                        "release": 100.0,    # ms
                    }
                },
                "low_pass": True,
                "low_pass_freq": 8000,
                "low_pass_db": -12,
                "high_pass": False,
                "high_pass_freq": 60,
                "high_pass_db": -12,
                "pan": 0
            }
        }
        
        # Track profile assignments
        self.track_profile_assignments = {}
        
        # Segmentation settings
        self.silence_threshold = 21  # dB below average
        self.min_silence = 4000      # ms
        self.seek_step = 2000        # ms
        self.min_time_between_segments = 10000  # ms
        self.min_segment_length = 15  # minutes
        self.dropout = 1             # minutes (minimum segment length)
        
        # Default padding settings - match original kirtan-processor.py values
        self.pre_segment_padding = 8  # 8 seconds before segment (original value)
        self.post_segment_padding = 3  # 3 seconds after segment (original value)
        
        # Export settings
        self.export_bitrate = 192
        self.export_album = ""
        self.fade_in = 0
        self.fade_out = 0
        self.save_unsegmented = False
        self.trim_only = False
        self.batch_normalize = False
        
        # Visualization settings
        self.show_waveform = True
        
        # Processing speed
        self.processing_speed = "Auto"  # "Slow", "Auto", "Full Speed"
        
        # Working directory
        self.working_dir = ""
        
        # Detected input channels
        self.detected_inputs = []
        
        # Processing state
        self.is_processing = False
        
        # Add QSettings for remembering last directory
        self.settings = QSettings("KirtanProcessor", "App")
        
        # Load last directory if available
        last_dir = self.settings.value("last_directory", "", type=str)
        if last_dir and os.path.exists(last_dir):
            self.working_dir = last_dir
        
        # Init UI
        self.init_ui()
        
        # Update UI with saved directory if available
        if self.working_dir:
            self.dir_path.setText(self.working_dir)
            self.scan_directory(self.working_dir)
            self.process_button.setEnabled(True)
        
        # CHECK FOR PERFORMANCE DEPENDENCIES - ADD THIS LINE
        self.check_performance_dependencies()
        
        # Load settings if available
        self.load_default_settings()
        
    
    def update_ui_during_processing(self):
        """Update UI elements during processing without blocking worker thread"""
        # Use processEvents() without specific flags
        QApplication.instance().processEvents()
        
        # Update resource monitors
        self.update_resource_usage()
        
        # Check if worker has completed
        if hasattr(self, 'worker') and not self.worker.isRunning():
            self.ui_update_timer.stop()
    
    def create_dynamic_processing_widget(self, row_idx, profile_name, dynamic_settings):
        """Create UI controls for the blended compressor-limiter"""
        # Create container for all dynamic processing controls
        container = QWidget()
        layout = QGridLayout(container)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setHorizontalSpacing(5)
        layout.setVerticalSpacing(2)
        
        # If settings don't exist or aren't in the right format, create defaults
        if not isinstance(dynamic_settings, dict):
            dynamic_settings = {
                "enabled": False,
                "compressor": {
                    "threshold": -18.0,
                    "ratio": 2.5,
                    "attack": 20.0,
                    "release": 250.0,
                },
                "limiter": {
                    "threshold": -1.0,
                    "release": 50.0,
                }
            }
        
        # Enable checkbox
        enable_check = QCheckBox("Enable Dynamic Processing")
        enable_check.setChecked(dynamic_settings.get("enabled", False))
        enable_check.stateChanged.connect(
            lambda state, name=profile_name: self.update_profile_dynamic_enabled(name, state)
        )
        enable_check.setToolTip("Enable/disable dynamic processing (compression and limiting)")
        layout.addWidget(enable_check, 0, 0, 1, 3)
        
        # Compressor settings
        comp_settings = dynamic_settings.get("compressor", {})
        
        # Compressor Threshold
        layout.addWidget(QLabel("Comp Threshold:"), 1, 0)
        comp_thresh_spin = QDoubleSpinBox()
        comp_thresh_spin.setRange(-60.0, 0.0)
        comp_thresh_spin.setSingleStep(0.5)
        comp_thresh_spin.setSuffix(" dB")
        comp_thresh_spin.setValue(comp_settings.get("threshold", -18.0))
        comp_thresh_spin.valueChanged.connect(
            lambda value, name=profile_name: self.update_profile_comp_threshold(name, value)
        )
        comp_thresh_spin.setToolTip("Level at which compression begins (in dB)\nSignals above this level will be compressed")
        layout.addWidget(comp_thresh_spin, 1, 1, 1, 2)
        
        # Compressor Ratio
        layout.addWidget(QLabel("Comp Ratio:"), 2, 0)
        comp_ratio_spin = QDoubleSpinBox()
        comp_ratio_spin.setRange(1.0, 20.0)
        comp_ratio_spin.setSingleStep(0.1)
        comp_ratio_spin.setSuffix(":1")
        comp_ratio_spin.setValue(comp_settings.get("ratio", 2.5))
        comp_ratio_spin.valueChanged.connect(
            lambda value, name=profile_name: self.update_profile_comp_ratio(name, value)
        )
        comp_ratio_spin.setToolTip("Compression ratio (e.g., 2:1 means for every 2dB increase in input,\nonly 1dB increase in output)")
        layout.addWidget(comp_ratio_spin, 2, 1, 1, 2)
        
        # Compressor Attack
        layout.addWidget(QLabel("Comp Attack:"), 3, 0)
        comp_attack_spin = QDoubleSpinBox()
        comp_attack_spin.setRange(0.1, 200.0)
        comp_attack_spin.setSingleStep(1.0)
        comp_attack_spin.setSuffix(" ms")
        comp_attack_spin.setValue(comp_settings.get("attack", 20.0))
        comp_attack_spin.valueChanged.connect(
            lambda value, name=profile_name: self.update_profile_comp_attack(name, value)
        )
        comp_attack_spin.setToolTip("Time it takes for compression to begin after signal exceeds threshold\nFaster attack = more aggressive compression")
        layout.addWidget(comp_attack_spin, 3, 1, 1, 2)
        
        # Compressor Release
        layout.addWidget(QLabel("Comp Release:"), 4, 0)
        comp_release_spin = QDoubleSpinBox()
        comp_release_spin.setRange(10.0, 1000.0)
        comp_release_spin.setSingleStep(10.0)
        comp_release_spin.setSuffix(" ms")
        comp_release_spin.setValue(comp_settings.get("release", 250.0))
        comp_release_spin.valueChanged.connect(
            lambda value, name=profile_name: self.update_profile_comp_release(name, value)
        )
        comp_release_spin.setToolTip("Time it takes for compression to stop after signal falls below threshold\nFaster release = more natural sound")
        layout.addWidget(comp_release_spin, 4, 1, 1, 2)
        
        # Limiter settings
        limit_settings = dynamic_settings.get("limiter", {})
        
        # Limiter Threshold
        layout.addWidget(QLabel("Limit Threshold:"), 5, 0)
        limit_thresh_spin = QDoubleSpinBox()
        limit_thresh_spin.setRange(-10.0, 0.0)
        limit_thresh_spin.setSingleStep(0.1)
        limit_thresh_spin.setSuffix(" dB")
        limit_thresh_spin.setValue(limit_settings.get("threshold", -1.0))
        limit_thresh_spin.valueChanged.connect(
            lambda value, name=profile_name: self.update_profile_limit_threshold(name, value)
        )
        limit_thresh_spin.setToolTip("Maximum level allowed (in dB)\nSignals above this level will be limited")
        layout.addWidget(limit_thresh_spin, 5, 1, 1, 2)
        
        # Limiter Release
        layout.addWidget(QLabel("Limit Release:"), 6, 0)
        limit_release_spin = QDoubleSpinBox()
        limit_release_spin.setRange(10.0, 500.0)
        limit_release_spin.setSingleStep(10.0)
        limit_release_spin.setSuffix(" ms")
        limit_release_spin.setValue(limit_settings.get("release", 50.0))
        limit_release_spin.valueChanged.connect(
            lambda value, name=profile_name: self.update_profile_limit_release(name, value)
        )
        limit_release_spin.setToolTip("Time it takes for limiting to stop after signal falls below threshold\nFaster release = more transparent limiting")
        layout.addWidget(limit_release_spin, 6, 1, 1, 2)
        
        return container
    
    def update_profile_dynamic_enabled(self, profile_name, state):
        """Update whether dynamic processing is enabled"""
        if profile_name in self.profiles:
            if 'dynamic_processing' not in self.profiles[profile_name]:
                self.profiles[profile_name]['dynamic_processing'] = {
                    "enabled": bool(state),
                    "compressor": {
                        "threshold": -18.0,
                        "ratio": 2.5,
                        "attack": 20.0,
                        "release": 250.0,
                    },
                    "limiter": {
                        "threshold": -1.0,
                        "release": 50.0,
                    }
                }
            else:
                self.profiles[profile_name]['dynamic_processing']["enabled"] = bool(state)

    def update_profile_comp_threshold(self, profile_name, value):
        """Update compressor threshold"""
        if profile_name in self.profiles:
            if 'dynamic_processing' not in self.profiles[profile_name]:
                self.profiles[profile_name]['dynamic_processing'] = {
                    "enabled": False,
                    "compressor": {
                        "threshold": value,
                        "ratio": 2.5,
                        "attack": 20.0,
                        "release": 250.0,
                    },
                    "limiter": {
                        "threshold": -1.0,
                        "release": 50.0,
                    }
                }
            else:
                if 'compressor' not in self.profiles[profile_name]['dynamic_processing']:
                    self.profiles[profile_name]['dynamic_processing']['compressor'] = {
                        "threshold": value,
                        "ratio": 2.5,
                        "attack": 20.0,
                        "release": 250.0,
                    }
                else:
                    self.profiles[profile_name]['dynamic_processing']['compressor']["threshold"] = value

    def update_profile_comp_ratio(self, profile_name, value):
        """Update compressor ratio"""
        if profile_name in self.profiles:
            if 'dynamic_processing' not in self.profiles[profile_name]:
                self.profiles[profile_name]['dynamic_processing'] = {
                    "enabled": False,
                    "compressor": {
                        "threshold": -18.0,
                        "ratio": value,
                        "attack": 20.0,
                        "release": 250.0,
                    },
                    "limiter": {
                        "threshold": -1.0,
                        "release": 50.0,
                    }
                }
            else:
                if 'compressor' not in self.profiles[profile_name]['dynamic_processing']:
                    self.profiles[profile_name]['dynamic_processing']['compressor'] = {
                        "threshold": -18.0,
                        "ratio": value,
                        "attack": 20.0,
                        "release": 250.0,
                    }
                else:
                    self.profiles[profile_name]['dynamic_processing']['compressor']["ratio"] = value

    def update_profile_comp_attack(self, profile_name, value):
        """Update compressor attack time"""
        if profile_name in self.profiles:
            if 'dynamic_processing' not in self.profiles[profile_name]:
                self.profiles[profile_name]['dynamic_processing'] = {
                    "enabled": False,
                    "compressor": {
                        "threshold": -18.0,
                        "ratio": 2.5,
                        "attack": value,
                        "release": 250.0,
                    },
                    "limiter": {
                        "threshold": -1.0,
                        "release": 50.0,
                    }
                }
            else:
                if 'compressor' not in self.profiles[profile_name]['dynamic_processing']:
                    self.profiles[profile_name]['dynamic_processing']['compressor'] = {
                        "threshold": -18.0,
                        "ratio": 2.5,
                        "attack": value,
                        "release": 250.0,
                    }
                else:
                    self.profiles[profile_name]['dynamic_processing']['compressor']["attack"] = value

    def update_profile_comp_release(self, profile_name, value):
        """Update compressor release time"""
        if profile_name in self.profiles:
            if 'dynamic_processing' not in self.profiles[profile_name]:
                self.profiles[profile_name]['dynamic_processing'] = {
                    "enabled": False,
                    "compressor": {
                        "threshold": -18.0,
                        "ratio": 2.5,
                        "attack": 20.0,
                        "release": value,
                    },
                    "limiter": {
                        "threshold": -1.0,
                        "release": 50.0,
                    }
                }
            else:
                if 'compressor' not in self.profiles[profile_name]['dynamic_processing']:
                    self.profiles[profile_name]['dynamic_processing']['compressor'] = {
                        "threshold": -18.0,
                        "ratio": 2.5,
                        "attack": 20.0,
                        "release": value,
                    }
                else:
                    self.profiles[profile_name]['dynamic_processing']['compressor']["release"] = value

    def update_profile_limit_threshold(self, profile_name, value):
        """Update limiter threshold"""
        if profile_name in self.profiles:
            if 'dynamic_processing' not in self.profiles[profile_name]:
                self.profiles[profile_name]['dynamic_processing'] = {
                    "enabled": False,
                    "compressor": {
                        "threshold": -18.0,
                        "ratio": 2.5,
                        "attack": 20.0,
                        "release": 250.0,
                    },
                    "limiter": {
                        "threshold": value,
                        "release": 50.0,
                    }
                }
            else:
                if 'limiter' not in self.profiles[profile_name]['dynamic_processing']:
                    self.profiles[profile_name]['dynamic_processing']['limiter'] = {
                        "threshold": value,
                        "release": 50.0,
                    }
                else:
                    self.profiles[profile_name]['dynamic_processing']['limiter']["threshold"] = value

    def update_profile_limit_release(self, profile_name, value):
        """Update limiter release time"""
        if profile_name in self.profiles:
            if 'dynamic_processing' not in self.profiles[profile_name]:
                self.profiles[profile_name]['dynamic_processing'] = {
                    "enabled": False,
                    "compressor": {
                        "threshold": -18.0,
                        "ratio": 2.5,
                        "attack": 20.0,
                        "release": 250.0,
                    },
                    "limiter": {
                        "threshold": -1.0,
                        "release": value,
                    }
                }
            else:
                if 'limiter' not in self.profiles[profile_name]['dynamic_processing']:
                    self.profiles[profile_name]['dynamic_processing']['limiter'] = {
                        "threshold": -1.0,
                        "release": value,
                    }
                else:
                    self.profiles[profile_name]['dynamic_processing']['limiter']["release"] = value
    
    def import_track_settings(self, metadata_path):
        """Import settings from a track's metadata file"""
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Apply profiles
            if 'profiles' in metadata:
                self.profiles = metadata['profiles']
            
            # Apply track profile assignments
            if 'track_profile_assignments' in metadata:
                self.track_profile_assignments = metadata['track_profile_assignments']
            
            # Apply segmentation settings
            if 'segmentation_settings' in metadata:
                seg = metadata['segmentation_settings']
                self.silence_threshold = seg.get('silence_threshold', self.silence_threshold)
                self.min_silence = seg.get('min_silence', self.min_silence)
                self.seek_step = seg.get('seek_step', self.seek_step)
                self.min_time_between_segments = seg.get('min_time_between_segments', self.min_time_between_segments)
                self.min_segment_length = seg.get('min_segment_length', self.min_segment_length)
                self.dropout = seg.get('dropout', self.dropout)
                self.pre_segment_padding = seg.get('pre_segment_padding', self.pre_segment_padding)
                self.post_segment_padding = seg.get('post_segment_padding', self.post_segment_padding)
            
            # Update UI to reflect imported settings
            self.update_profiles_table()
            self.update_track_assignments_table()
            self.update_segmentation_ui()
            self.update_export_ui()
            
            # Set the flag to indicate settings were loaded from a track
            self.settings_loaded_from_track = True
            
            track_name = os.path.basename(metadata_path).replace("_metadata.json", "").replace("_metadata_v1.json", "")
            self.log(f"Imported settings from {track_name}")
            return True
        except Exception as e:
            self.log(f"Error importing settings: {str(e)}")
            return False
    
    def create_normalize_widget(self, row_idx, profile_name, normalize_settings):
        # Create container for all normalization controls
        normalize_container = QWidget()
        normalize_layout = QGridLayout(normalize_container)
        normalize_layout.setContentsMargins(5, 2, 5, 2)
        normalize_layout.setHorizontalSpacing(5)
        normalize_layout.setVerticalSpacing(2)
        
        # Convert boolean to dict if needed
        if isinstance(normalize_settings, bool):
            normalize_settings = {
                "enabled": normalize_settings,
                "target_level": -1.0,
                "headroom": 2.0,
                "method": "peak"
            }
        elif normalize_settings is None:
            normalize_settings = {
                "enabled": True,
                "target_level": -1.0,
                "headroom": 2.0,
                "method": "peak"
            }
        
        # Enable checkbox
        enable_check = QCheckBox("Enable")
        enable_check.setChecked(normalize_settings.get("enabled", True))
        enable_check.stateChanged.connect(
            lambda state, name=profile_name: self.update_profile_normalize_enabled(name, state)
        )
        enable_check.setToolTip("Enable/disable normalization for this profile")
        normalize_layout.addWidget(enable_check, 0, 0)
        
        # Method selector
        method_combo = QComboBox()
        method_combo.addItems(["peak", "rms", "lufs"])
        method_combo.setCurrentText(normalize_settings.get("method", "peak"))
        method_combo.currentTextChanged.connect(
            lambda text, name=profile_name: self.update_profile_normalize_method(name, text)
        )
        method_combo.setToolTip("Normalization method:\npeak: Match peak levels\nrms: Match average levels\nlufs: Match perceived loudness")
        normalize_layout.addWidget(QLabel("Method:"), 0, 1)
        normalize_layout.addWidget(method_combo, 0, 2)
        
        # Target level
        target_spin = QDoubleSpinBox()
        target_spin.setRange(-30.0, 0.0)
        target_spin.setSingleStep(0.5)
        target_spin.setSuffix(" dB")
        target_spin.setValue(normalize_settings.get("target_level", -1.0))
        target_spin.valueChanged.connect(
            lambda value, name=profile_name: self.update_profile_normalize_target(name, value)
        )
        target_spin.setToolTip("Target level for normalization (in dB)\nNegative values are quieter")
        normalize_layout.addWidget(QLabel("Target:"), 1, 0)
        normalize_layout.addWidget(target_spin, 1, 1, 1, 2)
        
        # Headroom
        headroom_spin = QDoubleSpinBox()
        headroom_spin.setRange(0.0, 6.0)
        headroom_spin.setSingleStep(0.1)
        headroom_spin.setSuffix(" dB")
        headroom_spin.setValue(normalize_settings.get("headroom", 2.0))
        headroom_spin.valueChanged.connect(
            lambda value, name=profile_name: self.update_profile_normalize_headroom(name, value)
        )
        headroom_spin.setToolTip("Amount of headroom to preserve (in dB)\nHigher values allow for more dynamic range")
        normalize_layout.addWidget(QLabel("Headroom:"), 2, 0)
        normalize_layout.addWidget(headroom_spin, 2, 1, 1, 2)
        
        return normalize_container
    
    def update_profile_normalize_limiter(self, profile_name, state):
        """Update whether limiter is enabled for this profile"""
        if profile_name in self.profiles:
            if not isinstance(self.profiles[profile_name].get('normalize'), dict):
                self.profiles[profile_name]['normalize'] = {
                    "enabled": True,
                    "target_level": -1.0,
                    "headroom": 2.0,
                    "method": "peak",
                    "limiter": bool(state),
                    "limiter_threshold": -0.5
                }
            else:
                self.profiles[profile_name]['normalize']["limiter"] = bool(state)

    def update_profile_normalize_limiter_threshold(self, profile_name, value):
        """Update the limiter threshold for normalization"""
        if profile_name in self.profiles:
            if not isinstance(self.profiles[profile_name].get('normalize'), dict):
                self.profiles[profile_name]['normalize'] = {
                    "enabled": True,
                    "target_level": -1.0,
                    "headroom": 2.0,
                    "method": "peak",
                    "limiter": False,
                    "limiter_threshold": value
                }
            else:
                self.profiles[profile_name]['normalize']["limiter_threshold"] = value
    
    
                
    def import_track_settings_dialog(self):
        """Show dialog to select a track to import settings from"""
        if not self.working_dir:
            self.log("Please select a directory first")
            return
            
        # Find all metadata files in the edited directory
        edited_dir = f"{self.working_dir}/edited"
        if not os.path.exists(edited_dir):
            self.log("No processed tracks found")
            return
            
        # Look for both regular metadata files and versioned ones
        metadata_files = [f for f in os.listdir(edited_dir) 
                          if f.endswith("_metadata.json") or 
                             "_metadata_v" in f and f.endswith(".json")]
        
        if not metadata_files:
            self.log("No track settings found")
            return
        
        # Create a dialog to select which track settings to import
        dialog = QDialog(self)
        dialog.setWindowTitle("Import Track Settings")
        
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select a track to import settings from:"))
        
        # Create a list widget to show available tracks
        list_widget = QListWidget()
        
        # Add items to the list with better names
        for metadata_file in metadata_files:
            # Extract track name and version
            if "_metadata_v" in metadata_file:
                # Handle versioned metadata
                parts = metadata_file.split("_metadata_v")
                track_name = parts[0]
                version = parts[1].split(".")[0]  # Remove .json
                display_name = f"{track_name} (Version {version})"
            else:
                # Handle regular metadata
                track_name = metadata_file.replace("_metadata.json", "")
                display_name = track_name
                
            list_widget.addItem(display_name)
            # Store the metadata filename as item data
            list_widget.item(list_widget.count() - 1).setData(Qt.ItemDataRole.UserRole, metadata_file)
        
        layout.addWidget(list_widget)
        
        # Add buttons
        buttons_layout = QHBoxLayout()
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        buttons_layout.addWidget(cancel_button)
        
        import_button = QPushButton("Import")
        import_button.clicked.connect(dialog.accept)
        import_button.setDefault(True)
        buttons_layout.addWidget(import_button)
        
        layout.addLayout(buttons_layout)
        
        # Set dialog size
        dialog.setMinimumSize(400, 300)
        
        # Show dialog
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_items = list_widget.selectedItems()
            if selected_items:
                metadata_file = selected_items[0].data(Qt.ItemDataRole.UserRole)
                self.import_track_settings(f"{edited_dir}/{metadata_file}")
    
    def select_settings_version_dialog(self, track_name, versions):
        """Show dialog to select which version of settings to import"""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Select Version for {track_name}")
        
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"Multiple versions found for {track_name}. Please select:"))
        
        # Create a list widget for versions
        list_widget = QListWidget()
        
        # Add items with processed timestamp if available
        for track, version, metadata_file in versions:
            try:
                with open(f"{self.working_dir}/edited/{metadata_file}", 'r') as f:
                    metadata = json.load(f)
                    # You might need to add 'processed_time' to your metadata
                    process_time = metadata.get('processed_time', 'Unknown time')
                    list_widget.addItem(f"Version {version} (Processed: {process_time})")
            except:
                list_widget.addItem(f"Version {version}")
            
            # Store the metadata filename as item data
            list_widget.item(len(list_widget) - 1).setData(Qt.ItemDataRole.UserRole, metadata_file)
        
        # Select the latest version by default
        list_widget.setCurrentRow(len(versions) - 1)
        
        layout.addWidget(list_widget)
        
        # Add buttons
        buttons_layout = QHBoxLayout()
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        buttons_layout.addWidget(cancel_button)
        
        select_button = QPushButton("Select")
        select_button.clicked.connect(dialog.accept)
        select_button.setDefault(True)
        buttons_layout.addWidget(select_button)
        
        layout.addLayout(buttons_layout)
        
        # Show dialog
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_item = list_widget.currentItem()
            if selected_item:
                return selected_item.data(Qt.ItemDataRole.UserRole)
        
        return None
    
    def init_ui(self):
        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create tab widget
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)
        
        # Create tabs - use only Main, Profiles, and the new consolidated Settings
        self.create_main_tab(tab_widget)
        self.create_profiles_tab(tab_widget)
        self.create_settings_tab(tab_widget)  # New consolidated tab
        
        # Bottom controls
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(10)
        
        # Progress info with clear button
        progress_frame = QWidget()
        progress_layout = QVBoxLayout(progress_frame)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(5)

        # Add a header with clear button
        progress_header = QHBoxLayout()
        progress_header.setSpacing(5)

        progress_label = QLabel("Processing Log:")
        progress_header.addWidget(progress_label)

        progress_header.addStretch()

        self.clear_log_btn = StyledButton("Clear Log")
        self.clear_log_btn.setFixedWidth(100)
        self.clear_log_btn.setFixedHeight(25)
        self.clear_log_btn.clicked.connect(self.clear_log)
        progress_header.addWidget(self.clear_log_btn)

        progress_layout.addLayout(progress_header)

        # Progress text
        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.setMinimumHeight(100)
        progress_layout.addWidget(self.progress_text)

        bottom_layout.addWidget(progress_frame, 7)
        
        # Control buttons
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(10)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(25)
        buttons_layout.addWidget(self.progress_bar)
        
        # Processing speed dropdown
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Processing Speed:"))
        
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["Slow", "Auto", "Full Speed"])
        self.speed_combo.setCurrentText(self.processing_speed)
        self.speed_combo.currentTextChanged.connect(self.update_processing_speed)
        speed_layout.addWidget(self.speed_combo)
        
        buttons_layout.addLayout(speed_layout)
        
        # Process button
        self.process_button = StyledButton("Process Files", primary=True)
        self.process_button.setMinimumHeight(40)
        self.process_button.clicked.connect(self.start_processing)
        self.process_button.setEnabled(False)
        buttons_layout.addWidget(self.process_button)
        
        # Add simple CPU usage bar - same size as process button
        cpu_layout = QHBoxLayout()
        cpu_layout.addWidget(QLabel("CPU:"))
        
        self.cpu_usage_bar = QProgressBar()
        self.cpu_usage_bar.setRange(0, 100)
        self.cpu_usage_bar.setValue(0)
        self.cpu_usage_bar.setFixedHeight(25)
        self.cpu_usage_bar.setTextVisible(True)
        self.cpu_usage_bar.setFormat("%v%")
        cpu_layout.addWidget(self.cpu_usage_bar)
        
        buttons_layout.addLayout(cpu_layout)
        
        # Add in init_ui method, near CPU usage bar:
        memory_layout = QHBoxLayout()
        memory_layout.addWidget(QLabel("Memory:"))

        self.memory_usage_bar = QProgressBar()
        self.memory_usage_bar.setRange(0, 100)
        self.memory_usage_bar.setValue(0)
        self.memory_usage_bar.setFixedHeight(25)
        self.memory_usage_bar.setTextVisible(True)
        self.memory_usage_bar.setFormat("%v%")
        memory_layout.addWidget(self.memory_usage_bar)

        buttons_layout.addLayout(memory_layout)
        
        # Stop button
        self.stop_button = StyledButton("Stop Processing")
        self.stop_button.setMinimumHeight(40)
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("background-color: #FFB6C1;")  # Light pink
        buttons_layout.addWidget(self.stop_button)
        
        # Exit button
        exit_button = StyledButton("Exit")
        exit_button.setMinimumHeight(40)
        exit_button.clicked.connect(self.close)
        buttons_layout.addWidget(exit_button)
        
        bottom_layout.addLayout(buttons_layout, 3)
        main_layout.addLayout(bottom_layout)
        
        # Start CPU and Memroy monitor update timer
        self.resource_timer = QTimer()
        self.resource_timer.timeout.connect(self.update_resource_usage)
        self.resource_timer.start(1000)  # Update every second
        
    def create_main_tab(self, tab_widget):
        main_tab = QWidget()
        layout = QVBoxLayout(main_tab)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Welcome message
        welcome_label = QLabel("Kirtan Processor")
        header_font = QFont("Arial", 10, QFont.Weight.Bold)
        welcome_label.setFont(header_font)
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(welcome_label)
        
        # Directory selection
        dir_frame = QFrame()
        dir_frame.setFrameShape(QFrame.Shape.StyledPanel)
        dir_layout = QHBoxLayout(dir_frame)
        dir_layout.setContentsMargins(10, 10, 10, 10)
        
        dir_label = QLabel("Select Directory:")
        dir_label.setMinimumWidth(100)
        dir_layout.addWidget(dir_label)
        
        self.dir_path = QLineEdit()
        self.dir_path.setReadOnly(True)
        self.dir_path.setMinimumHeight(30)
        self.dir_path.setMinimumWidth(200)
        dir_layout.addWidget(self.dir_path, 3)
        
        self.browse_button = StyledButton("Browse...")
        self.browse_button.clicked.connect(self.select_directory)
        self.browse_button.setMinimumHeight(30)
        dir_layout.addWidget(self.browse_button)
        
        # Reset button
        self.reset_button = StyledButton("Reset")
        self.reset_button.clicked.connect(self.reset_track_selection)
        self.reset_button.setMinimumHeight(30)
        dir_layout.addWidget(self.reset_button)
        
        layout.addWidget(dir_frame)
        
        # Input assignments frame
        assignments_frame = QFrame()
        assignments_frame.setFrameShape(QFrame.Shape.StyledPanel)
        assignments_layout = QVBoxLayout(assignments_frame)
        assignments_layout.setContentsMargins(10, 10, 10, 10)

        # Title for assignments section
        assignments_title = QLabel("Input Channel Assignments")
        assignments_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        assignments_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        assignments_layout.addWidget(assignments_title)

        # Instructions
        assignments_info = QLabel("Assign audio profiles to each input channel detected in your recordings.")
        assignments_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        assignments_layout.addWidget(assignments_info)

        # Track assignments table
        self.track_assignments_table = QTableWidget(0, 2)
        self.track_assignments_table.setHorizontalHeaderLabels(["Input Channel", "Assigned Profile"])
        self.track_assignments_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.track_assignments_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.track_assignments_table.setColumnWidth(0, 150)

        # Set vertical scroll bar policy and size policy
        self.track_assignments_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.track_assignments_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.track_assignments_table.setMinimumHeight(200)
        self.track_assignments_table.verticalHeader().setVisible(False)

        assignments_layout.addWidget(self.track_assignments_table)

        layout.addWidget(assignments_frame)
        
        # Files display section
        files_frame = QFrame()
        files_frame.setFrameShape(QFrame.Shape.StyledPanel)
        files_layout = QVBoxLayout(files_frame)
        files_layout.setContentsMargins(10, 10, 10, 10)
        
        files_label = QLabel("Available Tracks:")
        files_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        files_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        files_layout.addWidget(files_label)
        
        # Files table
        self.files_table = QTableWidget(0, 3)
        self.files_table.setHorizontalHeaderLabels(["Track", "Inputs Found", "Status"])
        self.files_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.files_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.files_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.files_table.setColumnWidth(1, 150)
        self.files_table.setColumnWidth(2, 150)
        self.files_table.setMinimumHeight(200)
        self.files_table.verticalHeader().setVisible(False)
        files_layout.addWidget(self.files_table)
        
        # Add buttons to open output folder and play files
        post_process_layout = QHBoxLayout()
        post_process_layout.setSpacing(10)

        self.open_folder_button = StyledButton("Open Output Folder")
        self.open_folder_button.clicked.connect(self.open_output_folder)
        post_process_layout.addWidget(self.open_folder_button)

        self.play_files_button = StyledButton("Play Processed Files")
        self.play_files_button.clicked.connect(self.play_processed_files)
        post_process_layout.addWidget(self.play_files_button)

        # Add new button for importing settings
        self.import_settings_button = StyledButton("Import Track Settings")
        self.import_settings_button.clicked.connect(self.import_track_settings_dialog)
        post_process_layout.addWidget(self.import_settings_button)

        files_layout.addLayout(post_process_layout)
        
        layout.addWidget(files_frame)
        
        tab_widget.addTab(main_tab, "Main")

    def create_profiles_tab(self, tab_widget):
        profiles_tab = QWidget()
        layout = QVBoxLayout(profiles_tab)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Profiles header section - at the very top
        header_frame = QFrame()
        header_frame.setFrameShape(QFrame.Shape.StyledPanel)
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(10, 10, 10, 10)
        
        # Profiles header
        profiles_header = QLabel("Audio Profiles")
        profiles_header.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        profiles_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(profiles_header)
        
        # Profiles description
        profiles_desc = QLabel(
            "Create and customize audio profiles for different types of inputs.\n"
            "Adjust gain, EQ settings, and panning for each profile."
        )
        profiles_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(profiles_desc)
        
        # Add "Set as Default" button
        set_default_button = StyledButton("Set Current Profile as Default")
        set_default_button.clicked.connect(self.save_as_default_settings)
        header_layout.addWidget(set_default_button)
        
        layout.addWidget(header_frame)
        
        # Main content area - will expand to fill available space
        content_frame = QFrame()
        content_frame.setFrameShape(QFrame.Shape.StyledPanel)
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create a horizontal splitter for the profiles view
        self.profiles_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Profiles table
        profiles_frame = QFrame()
        profiles_frame.setFrameShape(QFrame.Shape.StyledPanel)
        profiles_layout = QVBoxLayout(profiles_frame)
        profiles_layout.setContentsMargins(10, 10, 10, 10)
        
        # Modified: Add normalization target column
        self.profiles_table = QTableWidget(0, 5)
        self.profiles_table.setHorizontalHeaderLabels([
            "Profile Name", "Gain (dB)", "Norm Target", "Enabled Features", "Features"
        ])
        
        # Set column widths for the table
        self.profiles_table.setColumnWidth(0, 150)  # Profile name
        self.profiles_table.setColumnWidth(1, 80)   # Gain
        self.profiles_table.setColumnWidth(2, 100)   # Normalization Target
        self.profiles_table.setColumnWidth(3, 180)  # Enabled features
        self.profiles_table.setColumnWidth(4, 120)   # Features
        
        # Table settings
        self.profiles_table.setMinimumHeight(200)
        self.profiles_table.verticalHeader().setVisible(False)
        self.profiles_table.verticalHeader().setDefaultSectionSize(40)  # Reduced row height for compact view
        
        # Connect selection signal
        self.profiles_table.itemSelectionChanged.connect(self.profile_selection_changed)
        
        profiles_layout.addWidget(self.profiles_table)
        
        # Right panel: Detail view container
        self.profile_detail_frame = QFrame()
        self.profile_detail_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.profile_detail_layout = QVBoxLayout(self.profile_detail_frame)
        self.profile_detail_layout.setContentsMargins(10, 10, 10, 10)
        
        # Add empty label initially
        self.empty_detail_label = QLabel("Select a profile to view and edit its settings")
        self.empty_detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.profile_detail_layout.addWidget(self.empty_detail_label)
        
        # Add panels to splitter
        self.profiles_splitter.addWidget(profiles_frame)
        self.profiles_splitter.addWidget(self.profile_detail_frame)
        
        # Set initial sizes (400 pixels for table, 600 pixels for details)
        self.profiles_splitter.setSizes([470, 530])
        
        content_layout.addWidget(self.profiles_splitter)
        
        # Add new profile section at the bottom
        add_profile_frame = QFrame()
        add_profile_frame.setFrameShape(QFrame.Shape.StyledPanel)
        add_profile_layout = QVBoxLayout(add_profile_frame)
        add_profile_layout.setContentsMargins(10, 10, 10, 10)
        
        add_profile_label = QLabel("Add New Profile")
        add_profile_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        add_profile_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        add_profile_layout.addWidget(add_profile_label)
        
        add_profile_input_layout = QHBoxLayout()
        add_profile_input_layout.setSpacing(10)
        
        self.new_profile_name = QLineEdit()
        self.new_profile_name.setPlaceholderText("New Profile Name")
        self.new_profile_name.setMinimumHeight(35)
        add_profile_input_layout.addWidget(self.new_profile_name, 3)
        
        self.add_profile_button = StyledButton("Add Profile")
        self.add_profile_button.clicked.connect(self.add_new_profile)
        self.add_profile_button.setMinimumHeight(35)
        add_profile_input_layout.addWidget(self.add_profile_button)
        
        add_profile_layout.addLayout(add_profile_input_layout)
        content_layout.addWidget(add_profile_frame)
        
        # Add the content frame to the main layout with stretch
        layout.addWidget(content_frame, 1)  # The '1' sets stretch factor to make it expand
        
        # Store the currently selected profile name
        self.selected_profile = None
        
        tab_widget.addTab(profiles_tab, "Profiles")
    
    def profile_selection_changed(self):
        """Handle selection change in profiles table"""
        selected_rows = self.profiles_table.selectedItems()
        if not selected_rows:
            # If nothing selected, clear detail panel
            self.clear_profile_detail_panel()
            self.selected_profile = None
            return
            
        # Get the profile name from the first column of the selected row
        row = selected_rows[0].row()
        profile_name = self.profiles_table.item(row, 0).text()
        
        # Save the selected profile
        self.selected_profile = profile_name
        
        # Update the detail panel
        self.update_profile_detail_panel(profile_name)
    
    def clear_profile_detail_panel(self):
        """Clear the profile detail panel"""
        # Remove all widgets from the layout
        while self.profile_detail_layout.count():
            item = self.profile_detail_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
        
        # Add empty label
        self.profile_detail_layout.addWidget(self.empty_detail_label)
    
    def update_profile_detail_panel(self, profile_name):
        """Update the detail panel with the selected profile's settings"""
        if profile_name not in self.profiles:
            self.clear_profile_detail_panel()
            return
            
        settings = self.profiles[profile_name]
        
        # Clear the current detail panel
        self.clear_profile_detail_panel()
        
        # Create detail widget
        detail_widget = self.create_profile_detail_widget(profile_name, settings)
        
        # Add to layout
        self.profile_detail_layout.addWidget(detail_widget)
    
    def create_profile_detail_widget(self, profile_name, settings):
        """Creates a widget containing all detailed profile settings"""
        detail_container = QWidget()
        main_layout = QVBoxLayout(detail_container)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Profile header
        header = QLabel(f"Edit Profile: {profile_name}")
        header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)
        
        # Create tabs for different setting categories
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)
        
        # EQ Tab
        eq_tab = QWidget()
        eq_layout = QGridLayout(eq_tab)
        eq_layout.setContentsMargins(10, 10, 10, 10)
        
        # Low Pass controls
        low_pass_check = QCheckBox("Low Pass Filter")
        low_pass_check.setChecked(settings.get('low_pass', False))
        low_pass_check.stateChanged.connect(
            lambda state, name=profile_name: self.update_profile_low_pass(name, state)
        )
        eq_layout.addWidget(low_pass_check, 0, 0)
        
        # Low pass frequency
        low_pass_freq_spin = QSpinBox()
        low_pass_freq_spin.setRange(1000, 20000)
        low_pass_freq_spin.setSingleStep(500)
        low_pass_freq_spin.setValue(settings.get('low_pass_freq', 8000))
        low_pass_freq_spin.valueChanged.connect(
            lambda value, name=profile_name: self.update_profile_low_pass_freq(name, value)
        )
        eq_layout.addWidget(QLabel("Frequency (Hz):"), 1, 0)
        eq_layout.addWidget(low_pass_freq_spin, 1, 1)
        
        # Low pass dB reduction
        low_pass_db_spin = QSpinBox()
        low_pass_db_spin.setRange(-24, 0)
        low_pass_db_spin.setValue(settings.get('low_pass_db', -12))
        low_pass_db_spin.valueChanged.connect(
            lambda value, name=profile_name: self.update_profile_low_pass_db(name, value)
        )
        eq_layout.addWidget(QLabel("Reduction (dB):"), 2, 0)
        eq_layout.addWidget(low_pass_db_spin, 2, 1)
        
        # Add spacer
        eq_layout.addWidget(QLabel(""), 3, 0)
        
        # High Pass controls
        high_pass_check = QCheckBox("High Pass Filter")
        high_pass_check.setChecked(settings.get('high_pass', False))
        high_pass_check.stateChanged.connect(
            lambda state, name=profile_name: self.update_profile_high_pass(name, state)
        )
        eq_layout.addWidget(high_pass_check, 4, 0)
        
        # High pass frequency
        high_pass_freq_spin = QSpinBox()
        high_pass_freq_spin.setRange(20, 2000)
        high_pass_freq_spin.setSingleStep(10)
        high_pass_freq_spin.setValue(settings.get('high_pass_freq', 200))
        high_pass_freq_spin.valueChanged.connect(
            lambda value, name=profile_name: self.update_profile_high_pass_freq(name, value)
        )
        eq_layout.addWidget(QLabel("Frequency (Hz):"), 5, 0)
        eq_layout.addWidget(high_pass_freq_spin, 5, 1)
        
        # High pass dB reduction
        high_pass_db_spin = QSpinBox()
        high_pass_db_spin.setRange(-24, 0)
        high_pass_db_spin.setValue(settings.get('high_pass_db', -12))
        high_pass_db_spin.valueChanged.connect(
            lambda value, name=profile_name: self.update_profile_high_pass_db(name, value)
        )
        eq_layout.addWidget(QLabel("Reduction (dB):"), 6, 0)
        eq_layout.addWidget(high_pass_db_spin, 6, 1)
        
        # Pan control
        pan_value_label = QLabel(f"{int(settings.get('pan', 0) * 100)}")
        pan_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        pan_slider = QSlider(Qt.Orientation.Horizontal)
        pan_slider.setRange(-100, 100)
        pan_slider.setValue(int(settings.get('pan', 0) * 100))
        pan_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        pan_slider.setTickInterval(50)
        
        # Update label when slider moves
        pan_slider.valueChanged.connect(lambda value, label=pan_value_label: label.setText(f"{value}"))
        pan_slider.valueChanged.connect(
            lambda value, name=profile_name: self.update_profile_pan(name, value / 100)
        )
        
        pan_layout = QHBoxLayout()
        pan_layout.addWidget(QLabel("L"))
        pan_layout.addWidget(pan_slider, 1)
        pan_layout.addWidget(QLabel("R"))
        pan_layout.addWidget(pan_value_label)
        
        eq_layout.addWidget(QLabel("Pan:"), 7, 0)
        eq_layout.addLayout(pan_layout, 7, 1)
        
        tabs.addTab(eq_tab, "EQ & Pan")
        
        # Normalize Tab
        normalize_tab = QWidget()
        normalize_widget = self.create_normalize_widget(0, profile_name, settings.get('normalize', True))
        normalize_layout = QVBoxLayout(normalize_tab)
        normalize_layout.addWidget(normalize_widget)
        tabs.addTab(normalize_tab, "Normalization")
        
        # Dynamic Processing Tab
        dynamic_tab = QWidget()
        dynamic_widget = self.create_dynamic_processing_widget(0, profile_name, settings.get('dynamic_processing', {"enabled": False}))
        dynamic_layout = QVBoxLayout(dynamic_tab)
        dynamic_layout.addWidget(dynamic_widget)
        tabs.addTab(dynamic_tab, "Dynamic Processing")
        
        main_layout.addWidget(tabs)
        
        return detail_container

    def update_profiles_table(self):
        """Update the profiles table with current profile settings"""
        # Save current selection
        current_selection = self.selected_profile
        
        self.profiles_table.setRowCount(0)
        row = 0
        
        for name, settings in self.profiles.items():
            self.profiles_table.insertRow(row)
            
            # Profile name
            profile_item = QTableWidgetItem(name)
            profile_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.profiles_table.setItem(row, 0, profile_item)
            
            # Gain
            gain_spin = QSpinBox()
            gain_spin.setRange(-20, 10)
            gain_spin.setValue(settings.get('gain', 0))
            gain_spin.valueChanged.connect(lambda value, name=name: self.update_profile_gain(name, value))
            
            # Create container widget with proper layout
            gain_container = QWidget()
            gain_layout = QHBoxLayout(gain_container)
            gain_layout.setContentsMargins(5, 2, 5, 2)
            gain_layout.addWidget(gain_spin)
            gain_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            self.profiles_table.setCellWidget(row, 1, gain_container)
            
            # Normalization Target
            norm_target_spin = QDoubleSpinBox()
            norm_target_spin.setRange(-30.0, 0.0)
            norm_target_spin.setSingleStep(0.5)
            norm_target_spin.setSuffix(" dB")
            
            # Get the normalization settings or set default
            if isinstance(settings.get('normalize'), dict):
                norm_settings = settings.get('normalize', {})
                norm_target_spin.setValue(norm_settings.get("target_level", -1.0))
            else:
                norm_target_spin.setValue(-1.0)
                
            norm_target_spin.valueChanged.connect(
                lambda value, name=name: self.update_profile_normalize_target(name, value)
            )
            
            # Create container for normalization target
            norm_target_container = QWidget()
            norm_target_layout = QHBoxLayout(norm_target_container)
            norm_target_layout.setContentsMargins(5, 2, 5, 2)
            norm_target_layout.addWidget(norm_target_spin)
            norm_target_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            self.profiles_table.setCellWidget(row, 2, norm_target_container)
            
            # Enabled features summary
            enabled_features = []
            if settings.get('normalize') and isinstance(settings.get('normalize'), dict) and settings.get('normalize').get('enabled', True):
                method = settings.get('normalize', {}).get('method', 'peak')
                enabled_features.append(f"Normalize ({method})")
            if settings.get('low_pass', False):
                enabled_features.append(f"LP {settings.get('low_pass_freq', 8000)}Hz")
            if settings.get('high_pass', False):
                enabled_features.append(f"HP {settings.get('high_pass_freq', 200)}Hz")
            if settings.get('pan', 0) != 0:
                pan_value = int(settings.get('pan', 0) * 100)
                enabled_features.append(f"Pan {pan_value:+d}")
            if settings.get('dynamic_processing', {}).get('enabled', False):
                enabled_features.append("Dynamic")
                
            features_item = QTableWidgetItem(", ".join(enabled_features) if enabled_features else "No processing")
            self.profiles_table.setItem(row, 3, features_item)
            
            # Actions column with edit button
            actions_container = QWidget()
            actions_layout = QHBoxLayout(actions_container)
            actions_layout.setContentsMargins(5, 2, 5, 2)
            
            edit_button = QPushButton("Edit")
            edit_button.clicked.connect(lambda checked, name=name: self.select_profile_for_edit(name))
            actions_layout.addWidget(edit_button)
            
            # Add delete button only for non-default profiles
            if name not in DEFAULT_PROFILES:
                delete_button = QPushButton("Delete")
                delete_button.clicked.connect(lambda checked, name=name: self.delete_profile(name))
                delete_button.setStyleSheet("QPushButton { color: red; }")
                actions_layout.addWidget(delete_button)
            
            actions_container.setLayout(actions_layout)
            self.profiles_table.setCellWidget(row, 4, actions_container)
            
            row += 1
        
        # Restore selection if possible
        if current_selection and current_selection in self.profiles:
            for row in range(self.profiles_table.rowCount()):
                if self.profiles_table.item(row, 0).text() == current_selection:
                    self.profiles_table.selectRow(row)
                    break
    
    def select_profile_for_edit(self, profile_name):
        """Select a profile for editing (programmatically select the row)"""
        for row in range(self.profiles_table.rowCount()):
            if self.profiles_table.item(row, 0).text() == profile_name:
                self.profiles_table.selectRow(row)
                break
    
    def create_settings_tab(self, tab_widget):
        settings_tab = QWidget()
        layout = QVBoxLayout(settings_tab)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Settings header
        settings_header = QLabel("Settings")
        settings_header.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        settings_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(settings_header)
        
        # Add a QScrollArea to contain all settings sections
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_container = QWidget()
        scroll_layout = QVBoxLayout(scroll_container)
        scroll_layout.setSpacing(15)
        
    # Section 1: App settings
        app_settings_group = QGroupBox("Application Settings")
        app_settings_layout = QVBoxLayout()
        app_settings_layout.setContentsMargins(10, 10, 10, 10)
        
        # Save/load settings
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)
        
        self.save_settings_btn = StyledButton("Save Custom Settings")
        self.save_settings_btn.clicked.connect(self.save_settings)
        self.save_settings_btn.setMinimumHeight(35)
        buttons_layout.addWidget(self.save_settings_btn)
        
        self.load_settings_btn = StyledButton("Load Custom Settings")
        self.load_settings_btn.clicked.connect(self.load_settings)
        self.load_settings_btn.setMinimumHeight(35)
        buttons_layout.addWidget(self.load_settings_btn)
        
        self.reset_settings_btn = StyledButton("Reset to Factory Defaults")
        self.reset_settings_btn.clicked.connect(self.reset_settings)
        self.reset_settings_btn.setMinimumHeight(35)
        buttons_layout.addWidget(self.reset_settings_btn)
        
        app_settings_layout.addLayout(buttons_layout)
        app_settings_group.setLayout(app_settings_layout)
        scroll_layout.addWidget(app_settings_group)
        
    # Section 2: Export settings
        export_group = QGroupBox("MP3 Export Settings")
        export_layout = QGridLayout()
        export_layout.setContentsMargins(10, 15, 10, 10)
        export_layout.setHorizontalSpacing(10)
        export_layout.setVerticalSpacing(10)
        
        # Bitrate
        export_layout.addWidget(QLabel("MP3 Bitrate (kbps):"), 0, 0)
        self.bitrate_combo = QComboBox()
        self.bitrate_combo.addItems(["64", "96", "128", "192", "256", "320"])
        self.bitrate_combo.setCurrentText(str(self.export_bitrate))
        self.bitrate_combo.currentTextChanged.connect(self.update_bitrate)
        self.bitrate_combo.setMinimumHeight(30)
        export_layout.addWidget(self.bitrate_combo, 0, 1)
        
        # Album
        export_layout.addWidget(QLabel("Album Name:"), 2, 0)
        self.album_edit = QLineEdit(self.export_album)
        self.album_edit.textChanged.connect(self.update_album)
        self.album_edit.setMinimumHeight(30)
        export_layout.addWidget(self.album_edit, 2, 1)
        
        # Fade in/out
        export_layout.addWidget(QLabel("Fade In (seconds):"), 3, 0)
        self.fade_in_spin = QSpinBox()
        self.fade_in_spin.setRange(0, 10)
        self.fade_in_spin.setValue(self.fade_in)
        self.fade_in_spin.valueChanged.connect(self.update_fade_in)
        self.fade_in_spin.setMinimumHeight(30)
        export_layout.addWidget(self.fade_in_spin, 3, 1)
        
        export_layout.addWidget(QLabel("Fade Out (seconds):"), 4, 0)
        self.fade_out_spin = QSpinBox()
        self.fade_out_spin.setRange(0, 10)
        self.fade_out_spin.setValue(self.fade_out)
        self.fade_out_spin.valueChanged.connect(self.update_fade_out)
        self.fade_out_spin.setMinimumHeight(30)
        export_layout.addWidget(self.fade_out_spin, 4, 1)
        
        # Save unsegmented
        self.save_unsegmented_check = QCheckBox("Save unsegmented version")
        self.save_unsegmented_check.setChecked(self.save_unsegmented)
        self.save_unsegmented_check.stateChanged.connect(self.update_save_unsegmented)
        self.save_unsegmented_check.setMinimumHeight(30)
        export_layout.addWidget(self.save_unsegmented_check, 5, 0, 1, 2)
        
        # Add UI checkbox for saving unsegmented track
        self.trim_only_check = QCheckBox("Trim silence at start/end only (no segmentation)")
        self.trim_only_check.setChecked(self.trim_only)
        self.trim_only_check.stateChanged.connect(self.update_trim_only)
        self.trim_only_check.setMinimumHeight(30)
        export_layout.addWidget(self.trim_only_check, 6, 0, 1, 2)
        
        export_group.setLayout(export_layout)
        scroll_layout.addWidget(export_group)
        
        # Add to the export settings section
        self.batch_normalize_check = QCheckBox("Apply batch normalization (consistent levels across tracks)")
        self.batch_normalize_check.setChecked(self.batch_normalize)
        self.batch_normalize_check.stateChanged.connect(self.update_batch_normalize)
        export_layout.addWidget(self.batch_normalize_check, 8, 0, 1, 2)
        
        # Add in create_settings_tab method, after batch normalize checkbox:
        help_button = QPushButton("Normalization Help")
        help_button.clicked.connect(self.show_normalization_help)
        export_layout.addWidget(help_button, 9, 0, 1, 2)
        
    # Section 3: Segmentation parameters
        seg_group = QGroupBox("Segmentation Parameters")
        seg_layout = QGridLayout()
        seg_layout.setContentsMargins(8, 8, 8, 8)
        seg_layout.setHorizontalSpacing(8)
        seg_layout.setVerticalSpacing(8)
        
        # Helper function to create parameter rows with tooltips
        def add_param_row(layout, row, label_text, widget, tooltip_text):
            container = QWidget()
            h_layout = QHBoxLayout(container)
            h_layout.setContentsMargins(0, 0, 0, 0)
            h_layout.setSpacing(5)
            
            label = QLabel(label_text)
            h_layout.addWidget(label)
            
            tooltip_icon = QLabel("ℹ️")
            tooltip_icon.setStyleSheet("color: blue;")
            tooltip_icon.setToolTip(tooltip_text)
            h_layout.addWidget(tooltip_icon)
            
            h_layout.addStretch()
            
            layout.addWidget(container, row, 0)
            layout.addWidget(widget, row, 1)
        
        # Padding settings - user configurable
        self.pre_padding_spin = QSpinBox()
        self.pre_padding_spin.setRange(-300, 300)  # Allow -30 to +30 seconds
        self.pre_padding_spin.setValue(self.pre_segment_padding)  # Use the current value
        self.pre_padding_spin.setEnabled(True)  # Enable user configuration
        self.pre_padding_spin.valueChanged.connect(self.update_pre_padding)
        self.pre_padding_spin.setMinimumHeight(30)
        self.pre_padding_spin.setMinimumWidth(80)
        add_param_row(seg_layout, 1, "Pre-segment Padding (seconds):", self.pre_padding_spin,
                     "Negative values: Move start point EARLIER (pulls back start time before vocals)\nPositive values: Move start point LATER (adds delay before vocals)")
        
        self.post_padding_spin = QSpinBox()
        self.post_padding_spin.setRange(-300, 300)  # Allow -30 to +30 seconds
        self.post_padding_spin.setValue(self.post_segment_padding)  # Use the current value
        self.post_padding_spin.setEnabled(True)  # Enable user configuration
        self.post_padding_spin.valueChanged.connect(self.update_post_padding)
        self.post_padding_spin.setMinimumHeight(30)
        self.post_padding_spin.setMinimumWidth(80)
        add_param_row(seg_layout, 2, "Post-segment Padding (seconds):", self.post_padding_spin,
                     "Positive values: Add time AFTER segment (move end later)\nNegative values: Remove time AFTER segment (move end earlier)")
        
        # Silence threshold
        self.silence_threshold_spin = QSpinBox()
        self.silence_threshold_spin.setRange(-60, 30)  # Negative dB values are typical for silence thresholds
        self.silence_threshold_spin.setValue(self.silence_threshold if self.silence_threshold < 0 else -self.silence_threshold)
        self.silence_threshold_spin.valueChanged.connect(self.update_silence_threshold)
        self.silence_threshold_spin.setMinimumHeight(30)
        self.silence_threshold_spin.setMinimumWidth(80)
        add_param_row(seg_layout, 3, "Silence Threshold (dB) (default=-21):", self.silence_threshold_spin, 
                  "How many dB below the audio's average level to consider as silence. Lower values = more sensitive to quiet sounds")
        
        # Min silence length
        self.min_silence_spin = QSpinBox()
        self.min_silence_spin.setRange(1000, 10000)
        self.min_silence_spin.setSingleStep(500)
        self.min_silence_spin.setValue(self.min_silence)
        self.min_silence_spin.valueChanged.connect(self.update_min_silence)
        self.min_silence_spin.setMinimumHeight(30)
        self.min_silence_spin.setMinimumWidth(80)
        add_param_row(seg_layout, 4, "Minimum Silence Length (ms) (default=4000):", self.min_silence_spin,
                     "How long silence must last to count as a break (higher = fewer breaks)")
        
        # Seek step
        self.seek_step_spin = QSpinBox()
        self.seek_step_spin.setRange(500, 5000)
        self.seek_step_spin.setSingleStep(500)
        self.seek_step_spin.setValue(self.seek_step)
        self.seek_step_spin.valueChanged.connect(self.update_seek_step)
        self.seek_step_spin.setMinimumHeight(30)
        self.seek_step_spin.setMinimumWidth(80)
        add_param_row(seg_layout, 5, "Seek Step (ms) (default=2000):", self.seek_step_spin,
                     "How frequently to check for silence (smaller = more precise but slower)")
        
        # Min time between segments
        self.min_time_spin = QSpinBox()
        self.min_time_spin.setRange(5000, 30000)
        self.min_time_spin.setSingleStep(1000)
        self.min_time_spin.setValue(self.min_time_between_segments)
        self.min_time_spin.valueChanged.connect(self.update_min_time)
        self.min_time_spin.setMinimumHeight(30)
        self.min_time_spin.setMinimumWidth(80)
        add_param_row(seg_layout, 6, "Min Time Between Segments (ms) (default=10000):", self.min_time_spin,
                     "How long silence must last to start a new segment")
        
        # Min segment length
        self.min_segment_spin = QSpinBox()
        self.min_segment_spin.setRange(5, 30)
        self.min_segment_spin.setValue(self.min_segment_length)
        self.min_segment_spin.valueChanged.connect(self.update_min_segment)
        self.min_segment_spin.setMinimumHeight(30)
        self.min_segment_spin.setMinimumWidth(80)
        add_param_row(seg_layout, 7, "Min Segment Length (minutes) (default=15):", self.min_segment_spin,
                     "Minimum length of a valid segment in minutes")
        
        # Dropout length
        self.dropout_spin = QSpinBox()
        self.dropout_spin.setRange(0, 5)
        self.dropout_spin.setValue(self.dropout)
        self.dropout_spin.valueChanged.connect(self.update_dropout)
        self.dropout_spin.setMinimumHeight(30)
        self.dropout_spin.setMinimumWidth(80)
        add_param_row(seg_layout, 8, "Dropout Length (minutes) (default=1):", self.dropout_spin,
                     "Discard segments shorter than this length (minutes)")
        
        seg_group.setLayout(seg_layout)
        scroll_layout.addWidget(seg_group)
        
        # Add a stretcher at the end to push everything up
        scroll_layout.addStretch(1)
        
        # Set the scroll content
        scroll_area.setWidget(scroll_container)
        layout.addWidget(scroll_area)
        
        # Add the tab
        tab_widget.addTab(settings_tab, "Settings")
        
    def select_directory(self):
        """Open directory selection dialog"""
        # Start from last directory if available
        start_dir = self.working_dir if self.working_dir else ""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory", start_dir)
        
        if dir_path:
            self.working_dir = dir_path
            self.dir_path.setText(dir_path)
            self.scan_directory(dir_path)
            self.process_button.setEnabled(True)
            
            # Save the selected directory for next time
            self.settings.setValue("last_directory", dir_path)
    
    
    def reset_track_selection(self):
        """Reset the track selection and clear found tracks"""
        self.working_dir = ""
        self.dir_path.setText("")
        self.files_table.setRowCount(0)
        self.track_assignments_table.setRowCount(0)
        self.detected_inputs = []
        self.process_button.setEnabled(False)
        self.log("Track selection reset")
        
    def clear_log(self):
        """Clear the progress log"""
        self.progress_text.clear()
        self.log("Log cleared")
    
    def scan_directory(self, dir_path):
        """Scan directory for audio files and update UI"""
        if hasattr(self, 'just_processed') and self.just_processed:
            self.just_processed = False
            return
        self.files_table.setRowCount(0)
        self.detected_inputs = []
        
        # Get all files in directory and subdirectories
        all_subdirs = tuple(os.walk(dir_path))
        
        # Get all tracks and their input channels
        track_input_map = {}
        
        for subdir, _, filenames in all_subdirs:
            # Skip "edited" directory
            if "/edited" in subdir.replace("\\", "/"):
                continue
                
            # Filter for valid audio files
            audio_files = [f for f in filenames if f[-4:].lower() == INPUT_FORMAT.lower()]
            
            # Group by track
            for file in audio_files:
                if "_" not in file:
                    continue
                    
                # Get track prefix (before last underscore)
                last_underscore = file.rfind("_")
                if last_underscore <= 0:
                    continue
                    
                track_prefix = file[:last_underscore]
                input_channel = file[last_underscore+1:file.rfind(".")]
                
                # Handle split files (for large recordings)
                if "-" in input_channel and input_channel[-5:-1].isdigit():
                    base_input = input_channel.split("-")[0]
                    input_channel = base_input
                
                # Add to map
                full_path = os.path.join(subdir, track_prefix)
                if full_path not in track_input_map:
                    track_input_map[full_path] = set()
                
                track_input_map[full_path].add(input_channel)
                
                # Add to detected inputs if not already there
                if input_channel not in self.detected_inputs:
                    self.detected_inputs.append(input_channel)
            
        # Check for existing metadata files to auto-load settings
        processed_tracks_exist = False
        latest_metadata_file = None
        latest_timestamp = 0
        
        edited_dir = os.path.join(dir_path, "edited")
        if os.path.exists(edited_dir):
            try:
                # Look for both regular and versioned metadata files
                metadata_files = [f for f in os.listdir(edited_dir) 
                                  if f.endswith("_metadata.json") or 
                                     "_metadata_v" in f and f.endswith(".json")]
                
                # Find the most recent metadata file based on file modification time
                for metadata_file in metadata_files:
                    metadata_path = os.path.join(edited_dir, metadata_file)
                    file_timestamp = os.path.getmtime(metadata_path)
                    
                    # Try to get the processing time from metadata
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            if 'processed_time' in metadata:
                                # Convert processed_time string to timestamp for comparison
                                process_time = time.strptime(metadata['processed_time'], "%Y-%m-%d %H:%M:%S")
                                process_timestamp = time.mktime(process_time)
                                file_timestamp = process_timestamp  # Use processed time instead of file modification time
                    except:
                        # If can't parse the file or get processed_time, just use file modification time
                        pass
                    
                    # Update if this is the most recent file
                    if file_timestamp > latest_timestamp:
                        latest_timestamp = file_timestamp
                        latest_metadata_file = metadata_path
                        processed_tracks_exist = True
                
                # Auto-load settings from the most recent processed track
                if processed_tracks_exist and latest_metadata_file:
                    self.log(f"Auto-loading settings from most recent processed track...")
                    self.import_track_settings(latest_metadata_file)
            except Exception as e:
                self.log(f"Warning: Could not auto-load settings: {str(e)}")
        
        # Update files table
        row = 0
        for track_path, inputs in track_input_map.items():
            self.files_table.insertRow(row)
            
            # Track name
            track_name = os.path.basename(track_path)
            track_item = QTableWidgetItem(track_name)
            self.files_table.setItem(row, 0, track_item)
            
            # Inputs found
            inputs_text = ", ".join(sorted(inputs))
            inputs_item = QTableWidgetItem(inputs_text)
            self.files_table.setItem(row, 1, inputs_item)
            
            # Status
            status = "Ready to process"
            
            # Check if already processed
            edited_path = os.path.join(dir_path, "edited", os.path.basename(track_path))
            versions = []
            version = 1
            try:
                # First check if the base version exists (no version number)
                if os.path.exists(f"{edited_path} - Segment 1.mp3"):
                    versions.append(1)
                
                # Then check for versioned files
                while os.path.exists(f"{edited_path}_v{version} - Segment 1.mp3"):
                    versions.append(version)
                    version += 1
                    # Safety limit to prevent infinite loops
                    if version > 100:  # Set a reasonable maximum number of versions
                        break
            except Exception as e:
                # If any error occurs during version checking, log it and continue
                self.log(f"Warning: Error checking versions for {os.path.basename(track_path)}: {str(e)}")
                versions = []

            if versions:
                if len(versions) == 1:
                    status = "Processed (1 version)"
                else:
                    status = f"Processed ({len(versions)} versions)"
                # Set background to pastel green for processed tracks
                for col in range(3):
                    item = self.files_table.item(row, col) if col < 2 else QTableWidgetItem(status)
                    if col == 2:
                        self.files_table.setItem(row, col, item)
                    item.setBackground(QBrush(QColor(200, 255, 200)))  # Pastel green
            else:
                status_item = QTableWidgetItem(status)
                self.files_table.setItem(row, 2, status_item)
            
            row += 1
            
        # Make sure the files table has a scroll bar
        self.files_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Update track assignments table
        self.update_track_assignments_table()
        
        # Also update profiles table with detected inputs
        self.update_profiles_table()
        
        # Log
        self.log(f"Found {row} tracks with {len(self.detected_inputs)} different input channels")
    
    def open_output_folder(self):
        """Open the main directory in file explorer"""
        if not self.working_dir:
            self.log("Please select a directory first")
            return
            
        # Use the main directory directly
        output_dir = self.working_dir
            
        # Open folder in file explorer based on platform
        if sys.platform == 'win32':
            os.startfile(output_dir)
        elif sys.platform == 'darwin':  # macOS
            subprocess.run(['open', output_dir])
        else:  # Linux
            subprocess.run(['xdg-open', output_dir])
    
    def play_processed_files(self):
        """Open the first processed file in default media player"""
        if not self.working_dir:
            self.log("Please select a directory first")
            return
            
        # Use the main directory directly
        output_dir = self.working_dir
            
        # Find MP3 files that look like processed files (with version numbers)
        mp3_files = [f for f in os.listdir(output_dir) if f.endswith('.mp3') and ('_v' in f or ' - Segment ' in f)]
        if not mp3_files:
            self.log("No processed files found")
            return
            
        # Open the first file in default media player
        track_path = os.path.join(output_dir, mp3_files[0])
        if sys.platform == 'win32':
            os.startfile(track_path)
        elif sys.platform == 'darwin':  # macOS
            subprocess.run(['open', track_path])
        else:  # Linux
            subprocess.run(['xdg-open', track_path])
            
        self.log(f"Playing {os.path.basename(track_path)}")
    
    def update_track_assignments_table(self):
        """Update the track assignments table with detected inputs"""
        self.track_assignments_table.setRowCount(0)
        
        # Always auto-assign profiles for any unassigned channels
        for input_channel in self.detected_inputs:
            if input_channel not in self.track_profile_assignments:
                # Default assignment logic
                if input_channel in ["Tr3", "3"]:
                    self.track_profile_assignments[input_channel] = "Kirtan (Vocals)"
                elif input_channel in ["Tr1", "5"]:
                    self.track_profile_assignments[input_channel] = "Tabla"
                elif input_channel in ["LR", "1-2"]:
                    self.track_profile_assignments[input_channel] = "Sangat (Ambient)"
                else:
                    # Default to Sangat for other channels
                    self.track_profile_assignments[input_channel] = "Sangat (Ambient)"
        
        # Set the table height to fit the content
        row = 0
        for input_channel in sorted(self.detected_inputs):
            self.track_assignments_table.insertRow(row)
            
            # Input channel
            input_item = QTableWidgetItem(input_channel)
            input_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.track_assignments_table.setItem(row, 0, input_item)
            
            # Profile selector
            profile_combo = QComboBox()
            profile_combo.addItems(list(self.profiles.keys()))
            profile_combo.setMinimumHeight(30)
            
            # Set current assignment
            if input_channel in self.track_profile_assignments:
                profile_combo.setCurrentText(self.track_profile_assignments[input_channel])
            
            # Connect change signal
            profile_combo.currentTextChanged.connect(
                lambda text, input=input_channel: self.update_track_assignment(input, text)
            )
            
            self.track_assignments_table.setCellWidget(row, 1, profile_combo)
            
            row += 1
        
        # Adjust table height based on content
        if row > 0:
            row_height = self.track_assignments_table.rowHeight(0)
            header_height = self.track_assignments_table.horizontalHeader().height()
            total_height = (row_height * min(row, 8)) + header_height + 2  # +2 for borders
            self.track_assignments_table.setMinimumHeight(total_height)
            
            # If there are more than 8 rows, enable scrolling
            if row > 8:
                self.track_assignments_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
            else:
                self.track_assignments_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    
    def update_profiles_table(self):
        """Update the profiles table with current profile settings"""
        # Save current selection
        current_selection = self.selected_profile
        
        self.profiles_table.setRowCount(0)
        row = 0
        
        for name, settings in self.profiles.items():
            self.profiles_table.insertRow(row)
            
            # Profile name
            profile_item = QTableWidgetItem(name)
            profile_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.profiles_table.setItem(row, 0, profile_item)
            
            # Gain
            gain_spin = QSpinBox()
            gain_spin.setRange(-20, 10)
            gain_spin.setValue(settings.get('gain', 0))
            gain_spin.valueChanged.connect(lambda value, name=name: self.update_profile_gain(name, value))
            
            # Create container widget with proper layout
            gain_container = QWidget()
            gain_layout = QHBoxLayout(gain_container)
            gain_layout.setContentsMargins(5, 2, 5, 2)
            gain_layout.addWidget(gain_spin)
            gain_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            self.profiles_table.setCellWidget(row, 1, gain_container)
            
            # Normalization Target
            norm_target_spin = QDoubleSpinBox()
            norm_target_spin.setRange(-30.0, 0.0)
            norm_target_spin.setSingleStep(0.5)
            norm_target_spin.setSuffix(" dB")
            
            # Get the normalization settings or set default
            if isinstance(settings.get('normalize'), dict):
                norm_settings = settings.get('normalize', {})
                norm_target_spin.setValue(norm_settings.get("target_level", -1.0))
            else:
                norm_target_spin.setValue(-1.0)
                
            norm_target_spin.valueChanged.connect(
                lambda value, name=name: self.update_profile_normalize_target(name, value)
            )
            
            # Create container for normalization target
            norm_target_container = QWidget()
            norm_target_layout = QHBoxLayout(norm_target_container)
            norm_target_layout.setContentsMargins(5, 2, 5, 2)
            norm_target_layout.addWidget(norm_target_spin)
            norm_target_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            self.profiles_table.setCellWidget(row, 2, norm_target_container)
            
            # Enabled features summary
            enabled_features = []
            if settings.get('normalize') and isinstance(settings.get('normalize'), dict) and settings.get('normalize').get('enabled', True):
                method = settings.get('normalize', {}).get('method', 'peak')
                enabled_features.append(f"Normalize ({method})")
            if settings.get('low_pass', False):
                enabled_features.append(f"LP {settings.get('low_pass_freq', 8000)}Hz")
            if settings.get('high_pass', False):
                enabled_features.append(f"HP {settings.get('high_pass_freq', 200)}Hz")
            if settings.get('pan', 0) != 0:
                pan_value = int(settings.get('pan', 0) * 100)
                enabled_features.append(f"Pan {pan_value:+d}")
            if settings.get('dynamic_processing', {}).get('enabled', False):
                enabled_features.append("Dynamic")
                
            features_item = QTableWidgetItem(", ".join(enabled_features) if enabled_features else "No processing")
            self.profiles_table.setItem(row, 3, features_item)
            
            # Actions column with edit button
            actions_container = QWidget()
            actions_layout = QHBoxLayout(actions_container)
            actions_layout.setContentsMargins(5, 2, 5, 2)
            
            edit_button = QPushButton("Edit")
            edit_button.clicked.connect(lambda checked, name=name: self.select_profile_for_edit(name))
            actions_layout.addWidget(edit_button)
            
            # Add delete button only for non-default profiles
            if name not in DEFAULT_PROFILES:
                delete_button = QPushButton("Delete")
                delete_button.clicked.connect(lambda checked, name=name: self.delete_profile(name))
                delete_button.setStyleSheet("QPushButton { color: red; }")
                actions_layout.addWidget(delete_button)
            
            actions_container.setLayout(actions_layout)
            self.profiles_table.setCellWidget(row, 4, actions_container)
            
            row += 1
        
        # Restore selection if possible
        if current_selection and current_selection in self.profiles:
            for row in range(self.profiles_table.rowCount()):
                if self.profiles_table.item(row, 0).text() == current_selection:
                    self.profiles_table.selectRow(row)
                    break
    
    def select_profile_for_edit(self, profile_name):
        """Select a profile for editing (programmatically select the row)"""
        for row in range(self.profiles_table.rowCount()):
            if self.profiles_table.item(row, 0).text() == profile_name:
                self.profiles_table.selectRow(row)
                break
    
    def add_new_profile(self):
        """Add a new profile with default settings"""
        profile_name = self.new_profile_name.text().strip()
        if not profile_name:
            QMessageBox.warning(self, "Warning", "Please enter a profile name")
            return
            
        if profile_name in self.profiles:
            QMessageBox.warning(self, "Warning", "Profile name already exists")
            return
            
        # Create new profile with default settings
        self.profiles[profile_name] = {
            "gain": 0,
            "normalize": {
                "enabled": True,
                "target_level": -3.0,
                "headroom": 2.0,
                "method": "peak"
            },
            "low_pass": False,
            "low_pass_freq": 8000,
            "low_pass_db": -12,
            "high_pass": False,
            "high_pass_freq": 200,
            "high_pass_db": -12,
            "pan": 0,
            "dynamic_processing": {
                "enabled": False,
                "compressor": {
                    "threshold": -18.0,
                    "ratio": 2.5,
                    "attack": 20.0,
                    "release": 250.0,
                },
                "limiter": {
                    "threshold": -1.0,
                    "release": 50.0,
                }
            }
        }
        
        # Update UI
        self.update_profiles_table()
        self.update_track_assignments_table()
        self.new_profile_name.clear()
        
        self.log(f"Added new profile: {profile_name}")
    
    
    def auto_assign_profiles(self):
        """Auto-assign profiles to input channels based on common naming conventions"""
        for input_channel in self.detected_inputs:
            # Default assignment logic
            if input_channel in ["Tr3", "3"]:
                self.track_profile_assignments[input_channel] = "Kirtan (Vocals)"
            elif input_channel in ["Tr1", "5"]:
                self.track_profile_assignments[input_channel] = "Tabla"
            elif input_channel in ["LR", "1-2"]:
                self.track_profile_assignments[input_channel] = "Sangat (Ambient)"
            else:
                # Default to Sangat for other channels
                self.track_profile_assignments[input_channel] = "Sangat (Ambient)"
        
        # Update UI
        self.update_track_assignments_table()
        self.log("Auto-assigned profiles to input channels")
    
    def reset_track_assignments(self):
        """Reset all track assignments"""
        self.track_profile_assignments = {}
        self.update_track_assignments_table()
        self.log("Reset all input channel assignments")
    
    def update_track_assignment(self, input_channel, profile_name):
        """Update the profile assignment for an input channel"""
        self.track_profile_assignments[input_channel] = profile_name
    
    def update_profile_gain(self, profile_name, value):
        if profile_name in self.profiles:
            self.profiles[profile_name]['gain'] = value
    
    def update_profile_normalize(self, profile_name, state):
        if profile_name in self.profiles:
            self.profiles[profile_name]['normalize'] = bool(state)
    
    def update_profile_low_pass(self, profile_name, state):
        if profile_name in self.profiles:
            self.profiles[profile_name]['low_pass'] = bool(state)
    
    def update_profile_low_pass_freq(self, profile_name, value):
        if profile_name in self.profiles:
            self.profiles[profile_name]['low_pass_freq'] = value
            
    def update_profile_low_pass_db(self, profile_name, value):
        if profile_name in self.profiles:
            self.profiles[profile_name]['low_pass_db'] = value
    
    def update_profile_high_pass(self, profile_name, state):
        if profile_name in self.profiles:
            self.profiles[profile_name]['high_pass'] = bool(state)
    
    def update_profile_high_pass_freq(self, profile_name, value):
        if profile_name in self.profiles:
            self.profiles[profile_name]['high_pass_freq'] = value
            
    def update_profile_high_pass_db(self, profile_name, value):
        if profile_name in self.profiles:
            self.profiles[profile_name]['high_pass_db'] = value
    
    def update_profile_pan(self, profile_name, value):
        if profile_name in self.profiles:
            self.profiles[profile_name]['pan'] = value
    
    def update_silence_threshold(self, value):
        """Update silence threshold value
        
        The value represents how many dB below the audio's average level to consider as silence.
        A positive value means the threshold will be that many dB below the average level.
        A negative value means the threshold will be that many dB above the average level.
        """
        self.silence_threshold = value
    
    def update_min_silence(self, value):
        self.min_silence = value
    
    def update_seek_step(self, value):
        self.seek_step = value
    
    def update_min_time(self, value):
        self.min_time_between_segments = value
    
    def update_min_segment(self, value):
        self.min_segment_length = value
    
    def update_dropout(self, value):
        self.dropout = value
    
    def update_pre_padding(self, value):
        """Update pre-segment padding in seconds
        
        Negative value: Move start point EARLIER (pulls back start time before vocals)
        Positive value: Move start point LATER (adds delay before vocals)
        """
        self.pre_segment_padding = value
        
    def update_post_padding(self, value):
        """Update post-segment padding in seconds
        
        Positive value: Add time AFTER segment (move end later)
        Negative value: Remove time AFTER segment (move end earlier)
        """
        self.post_segment_padding = value
    
    def update_bitrate(self, value):
        self.export_bitrate = int(value)
    
    def update_album(self, value):
        self.export_album = value
    
    def update_fade_in(self, value):
        self.fade_in = value
    
    def update_fade_out(self, value):
        self.fade_out = value
    
    def update_save_unsegmented(self, state):
        self.save_unsegmented = bool(state)
        
    def update_trim_only(self, state):
        self.trim_only = bool(state)
    
    def update_show_waveform(self, state):
        self.show_waveform = bool(state)
    
    def update_batch_normalize(self, state):
        self.batch_normalize = bool(state)
        
    def update_profile_normalize_enabled(self, profile_name, state):
        """Update whether normalization is enabled for this profile"""
        if profile_name in self.profiles:
            # Ensure normalize is a dictionary
            if not isinstance(self.profiles[profile_name].get('normalize'), dict):
                self.profiles[profile_name]['normalize'] = {
                    "enabled": bool(state),
                    "target_level": -1.0,
                    "headroom": 2.0,
                    "method": "peak"
                }
            else:
                self.profiles[profile_name]['normalize']["enabled"] = bool(state)
                
            # Update the enabled features cell
            self.update_profile_enabled_features(profile_name)

    def update_profile_enabled_features(self, profile_name):
        """Update the enabled features cell for a specific profile"""
        # Find the row for this profile
        for row in range(self.profiles_table.rowCount()):
            if self.profiles_table.item(row, 0) and self.profiles_table.item(row, 0).text() == profile_name:
                # Get settings for this profile
                settings = self.profiles[profile_name]
                
                # Build enabled features list
                enabled_features = []
                if settings.get('normalize') and isinstance(settings.get('normalize'), dict) and settings.get('normalize').get('enabled', True):
                    method = settings.get('normalize', {}).get('method', 'peak')
                    enabled_features.append(f"Normalize ({method})")
                if settings.get('low_pass', False):
                    enabled_features.append(f"LP {settings.get('low_pass_freq', 8000)}Hz")
                if settings.get('high_pass', False):
                    enabled_features.append(f"HP {settings.get('high_pass_freq', 200)}Hz")
                if settings.get('pan', 0) != 0:
                    pan_value = int(settings.get('pan', 0) * 100)
                    enabled_features.append(f"Pan {pan_value:+d}")
                if settings.get('dynamic_processing', {}).get('enabled', False):
                    enabled_features.append("Dynamic")
                
                # Update the cell
                features_item = QTableWidgetItem(", ".join(enabled_features) if enabled_features else "No processing")
                self.profiles_table.setItem(row, 3, features_item)
                break

    def update_profile_low_pass(self, profile_name, state):
        """Update whether low pass filter is enabled"""
        if profile_name in self.profiles:
            self.profiles[profile_name]['low_pass'] = bool(state)
            
            # Update the enabled features cell
            self.update_profile_enabled_features(profile_name)

    def update_profile_high_pass(self, profile_name, state):
        """Update whether high pass filter is enabled"""
        if profile_name in self.profiles:
            self.profiles[profile_name]['high_pass'] = bool(state)
            
            # Update the enabled features cell
            self.update_profile_enabled_features(profile_name)

    def update_profile_dynamic_enabled(self, profile_name, state):
        """Update whether dynamic processing is enabled"""
        if profile_name in self.profiles:
            if 'dynamic_processing' not in self.profiles[profile_name]:
                self.profiles[profile_name]['dynamic_processing'] = {
                    "enabled": bool(state),
                    "compressor": {
                        "threshold": -18.0,
                        "ratio": 2.5,
                        "attack": 20.0,
                        "release": 250.0,
                    },
                    "limiter": {
                        "threshold": -1.0,
                        "release": 50.0,
                    }
                }
            else:
                self.profiles[profile_name]['dynamic_processing']["enabled"] = bool(state)
                
            # Update the enabled features cell
            self.update_profile_enabled_features(profile_name)

    def update_profile_pan(self, profile_name, value):
        """Update pan value for this profile"""
        if profile_name in self.profiles:
            self.profiles[profile_name]['pan'] = value
            
            # Update the enabled features cell
            self.update_profile_enabled_features(profile_name)

    def update_profile_normalize_method(self, profile_name, method):
        """Update normalization method for this profile"""
        if profile_name in self.profiles:
            if not isinstance(self.profiles[profile_name].get('normalize'), dict):
                self.profiles[profile_name]['normalize'] = {
                    "enabled": True,
                    "target_level": -1.0,
                    "headroom": 2.0,
                    "method": method
                }
            else:
                self.profiles[profile_name]['normalize']["method"] = method
                
            # Update the enabled features cell
            self.update_profile_enabled_features(profile_name)

    def update_profile_low_pass_freq(self, profile_name, value):
        """Update low pass frequency for this profile"""
        if profile_name in self.profiles:
            self.profiles[profile_name]['low_pass_freq'] = value
            
            # Update the enabled features cell if low pass is enabled
            if self.profiles[profile_name].get('low_pass', False):
                self.update_profile_enabled_features(profile_name)

    def update_profile_high_pass_freq(self, profile_name, value):
        """Update high pass frequency for this profile"""
        if profile_name in self.profiles:
            self.profiles[profile_name]['high_pass_freq'] = value
            
            # Update the enabled features cell if high pass is enabled
            if self.profiles[profile_name].get('high_pass', False):
                self.update_profile_enabled_features(profile_name)

    def show_normalization_help(self):
        """Show help dialog for normalization and dynamic processing features"""
        help_text = """
        <h3>Enhanced Audio Normalization</h3>
        
        <p>The Kirtan Processor supports three different normalization methods:</p>
        
        <ul>
            <li><b>Peak Normalization</b> (Default): Adjusts audio based on peak levels</li>
            <li><b>RMS Normalization</b>: Adjusts audio based on average loudness</li>
            <li><b>LUFS Normalization</b>: Adjusts audio based on perceived loudness (broadcast standard)</li>
        </ul>
        
        <p><b>Settings Explained:</b></p>
        
        <ul>
            <li><b>Method</b>: The normalization technique to use</li>
            <li><b>Target Level</b>: The desired output level in dB</li>
            <li><b>Headroom</b>: Amount of space (in dB) to leave below 0dB to prevent clipping</li>
        </ul>
        
        <p><b>Batch Normalization</b>: When enabled, all tracks will be normalized to consistent levels relative to each other, 
        preserving their natural dynamics and relative loudness differences.</p>
        
        <h3>Dynamic Processing (Compression & Limiting)</h3>
        
        <p>The dynamic processor combines compression and limiting for smooth, musical control of audio dynamics:</p>
        
        <p><b>Compressor Settings:</b></p>
        <ul>
            <li><b>Threshold</b>: Level at which compression begins (lower = more compression)</li>
            <li><b>Ratio</b>: Strength of compression (higher = more aggressive volume reduction)</li>
            <li><b>Attack</b>: How quickly compression engages (lower = faster reaction)</li>
            <li><b>Release</b>: How quickly compression disengages (higher = smoother transitions)</li>
        </ul>
        
        <p><b>Limiter Settings:</b></p>
        <ul>
            <li><b>Threshold</b>: Maximum allowed volume level (absolute ceiling)</li>
            <li><b>Release</b>: How quickly limiting disengages after peaks</li>
        </ul>
        
        <p><b>Recommended Settings:</b></p>
        
        <p><b>For Vocals:</b><br>
        Compressor: Threshold -18dB, Ratio 2-3:1, Attack 20ms, Release 250ms<br>
        Limiter: Threshold -1dB, Release 50ms</p>
        
        <p><b>For Tabla:</b><br>
        Compressor: Threshold -20dB, Ratio 3-4:1, Attack 10ms, Release 150ms<br>
        Limiter: Threshold -0.5dB, Release 40ms</p>
        
        <p><b>For Ambient/Sangat:</b><br>
        Compressor: Threshold -24dB, Ratio 2:1, Attack 40ms, Release 400ms<br>
        Limiter: Threshold -1.5dB, Release 100ms</p>
        
        <p><b>Note:</b> LUFS normalization requires the pyloudnorm package (pip install pyloudnorm).</p>
        """
        
        msg = QMessageBox()
        msg.setWindowTitle("Audio Processing Help")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(help_text)
        msg.exec()
        
    def update_processing_speed(self, value):
        """Update processing speed setting"""
        self.processing_speed = value
        # Update the label in the CPU monitor section
        if hasattr(self, 'processing_speed_label'):
            # Current processing speed info
            self.processing_speed_label = QLabel(f"Current Processing Speed: {self.processing_speed}")
            self.processing_speed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            #cpu_monitor_layout.addWidget(self.processing_speed_label)
        self.log(f"Processing speed set to: {value}")
    
    # Enhanced CPU and Memroy monitoring
    def update_resource_usage(self):
        """Update both CPU and memory usage displays with enhanced monitoring"""
        try:
            # Update CPU
            cpu_percent = psutil.cpu_percent()
            self.cpu_usage_bar.setValue(int(cpu_percent))
            
            # Update Memory with more details
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.memory_usage_bar.setValue(int(memory_percent))
            
            # Add memory usage details to tooltip
            memory_details = (f"Total: {memory.total/1024**3:.1f} GB\n"
                            f"Available: {memory.available/1024**3:.1f} GB\n"
                            f"Used: {(memory.total - memory.available)/1024**3:.1f} GB")
            self.memory_usage_bar.setToolTip(memory_details)
            
            # Set CPU bar color based on usage
            if cpu_percent < 50:
                self.cpu_usage_bar.setStyleSheet("QProgressBar::chunk { background-color: #90EE90; }")  # Light green
            elif cpu_percent < 80:
                self.cpu_usage_bar.setStyleSheet("QProgressBar::chunk { background-color: #FFD700; }")  # Gold
            else:
                self.cpu_usage_bar.setStyleSheet("QProgressBar::chunk { background-color: #FF6347; }")  # Tomato red
            
            # Set Memory bar color based on usage
            if memory_percent < 70:
                self.memory_usage_bar.setStyleSheet("QProgressBar::chunk { background-color: #90EE90; }")  # Light green
            elif memory_percent < 85:
                self.memory_usage_bar.setStyleSheet("QProgressBar::chunk { background-color: #FFD700; }")  # Gold
            else:
                self.memory_usage_bar.setStyleSheet("QProgressBar::chunk { background-color: #FF6347; }")  # Tomato red
                
            # Trigger memory optimization if usage is high and worker is running
            if hasattr(self, 'worker') and self.worker.isRunning() and memory_percent > 85:
                self.worker.optimize_memory_usage()
                
        except Exception as e:
            # Silent failure for resource monitoring to avoid disrupting the main process
            print(f"Error updating resource usage: {e}")
    
    def save_settings(self):
        """Save current settings to JSON file"""
        settings = {
            'profiles': self.profiles,
            'track_profile_assignments': self.track_profile_assignments,
            'segmentation': {
                'silence_threshold': self.silence_threshold,
                'min_silence': self.min_silence,
                'seek_step': self.seek_step,
                'min_time_between_segments': self.min_time_between_segments,
                'min_segment_length': self.min_segment_length,
                'dropout': self.dropout,
                'pre_segment_padding': self.pre_segment_padding,
                'post_segment_padding': self.post_segment_padding
            },
            'export': {
                'bitrate': self.export_bitrate,
                'album': self.export_album,
                'fade_in': self.fade_in,
                'fade_out': self.fade_out,
                'save_unsegmented': self.save_unsegmented,
                'trim_only': self.trim_only,
                'batch_normalize': self.batch_normalize
            },
            'visualization': {
                'show_waveform': self.show_waveform
            },
            'processing': {
                'processing_speed': self.processing_speed
            }
        }
        
        # Ask for save location
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Settings", "", "JSON Files (*.json)"
        )
        
        if save_path:
            if not save_path.endswith('.json'):
                save_path += '.json'
                
            try:
                with open(save_path, 'w') as f:
                    json.dump(settings, f, indent=2)
                self.log(f"Settings saved to {save_path}")
            except Exception as e:
                self.log(f"Error saving settings: {str(e)}")
                
    def load_settings(self):
        """Load settings from JSON file"""
        # Ask for load location - always show the dialog when using the button
        load_path, _ = QFileDialog.getOpenFileName(
            self, "Load Settings", "", "JSON Files (*.json)"
        )
        
        if load_path and os.path.exists(load_path):
            try:
                with open(load_path, 'r') as f:
                    settings = json.load(f)
                    
                # Apply loaded settings
                if 'profiles' in settings:
                    self.profiles = settings['profiles']
                    
                    # Make sure dB settings exist
                    for profile_name, profile in self.profiles.items():
                        if 'low_pass_db' not in profile:
                            profile['low_pass_db'] = -12
                        if 'high_pass_db' not in profile:
                            profile['high_pass_db'] = -12
                
                if 'track_profile_assignments' in settings:
                    self.track_profile_assignments = settings['track_profile_assignments']
                
                if 'segmentation' in settings:
                    seg = settings['segmentation']
                    self.silence_threshold = seg.get('silence_threshold', self.silence_threshold)
                    self.min_silence = seg.get('min_silence', self.min_silence)
                    self.seek_step = seg.get('seek_step', self.seek_step)
                    self.min_time_between_segments = seg.get('min_time_between_segments', self.min_time_between_segments)
                    self.min_segment_length = seg.get('min_segment_length', self.min_segment_length)
                    self.dropout = seg.get('dropout', self.dropout)
                    self.pre_segment_padding = seg.get('pre_segment_padding', self.pre_segment_padding)
                    self.post_segment_padding = seg.get('post_segment_padding', self.post_segment_padding)
                
                if 'export' in settings:
                    exp = settings['export']
                    self.export_bitrate = exp.get('bitrate', self.export_bitrate)
                    self.export_album = exp.get('album', self.export_album)
                    self.fade_in = exp.get('fade_in', self.fade_in)
                    self.fade_out = exp.get('fade_out', self.fade_out)
                    self.save_unsegmented = exp.get('save_unsegmented', self.save_unsegmented)
                    self.trim_only = exp.get('trim_only', self.trim_only)
                    self.batch_normalize = exp.get('batch_normalize', self.batch_normalize)
                
                if 'visualization' in settings:
                    vis = settings['visualization']
                    self.show_waveform = vis.get('show_waveform', self.show_waveform)
                    
                if 'processing' in settings:
                    proc = settings['processing']
                    self.processing_speed = proc.get('processing_speed', self.processing_speed)
                
                # Update UI
                self.update_profiles_table()
                self.update_track_assignments_table()
                self.update_segmentation_ui()
                self.update_export_ui()
                self.speed_combo.setCurrentText(self.processing_speed)
                
                self.log(f"Settings loaded from {load_path}")
            except Exception as e:
                self.log(f"Error loading settings: {str(e)}")
                
    def load_default_settings(self):
        """Load default settings from default_settings.py"""
        try:
            # Get the path to default_settings.py
            default_settings_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                'default_settings.py'
            )
            
            # Import default_settings using importlib
            import importlib.util
            spec = importlib.util.spec_from_file_location("default_settings", default_settings_path)
            default_settings = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(default_settings)
            
            # Get all settings
            settings = default_settings.get_default_settings()
            
            # Apply profiles
            if 'profiles' in settings:
                self.profiles = settings['profiles']
            
            # Apply segmentation settings
            if 'segmentation' in settings:
                seg = settings['segmentation']
                self.silence_threshold = seg.get('silence_threshold', 21)
                self.min_silence = seg.get('min_silence', 4000)
                self.seek_step = seg.get('seek_step', 2000)
                self.min_time_between_segments = seg.get('min_time_between_segments', 10000)
                self.min_segment_length = seg.get('min_segment_length', 15)
                self.dropout = seg.get('dropout', 1)
                self.pre_segment_padding = seg.get('pre_segment_padding', 5)
                self.post_segment_padding = seg.get('post_segment_padding', 5)
            
            # Apply export settings
            if 'export' in settings:
                exp = settings['export']
                self.export_bitrate = exp.get('bitrate', 192)
                self.export_album = exp.get('album', "")
                self.fade_in = exp.get('fade_in', 0)
                self.fade_out = exp.get('fade_out', 0)
                self.save_unsegmented = exp.get('save_unsegmented', False)
                self.trim_only = exp.get('trim_only', False)
                self.batch_normalize = exp.get('batch_normalize', False)
            
            # Apply visualization settings
            if 'visualization' in settings:
                vis = settings['visualization']
                self.show_waveform = vis.get('show_waveform', False)
            
            # Apply processing settings
            if 'processing' in settings:
                proc = settings['processing']
                self.processing_speed = proc.get('speed', "Auto")
            
            self.log("Loaded settings from default_settings.py")
            
            # Update UI to reflect loaded settings
            self.update_profiles_table()
            self.update_track_assignments_table()
            self.update_segmentation_ui()
            self.update_export_ui()
            
        except Exception as e:
            self.log(f"Error loading default settings: {str(e)}")
            self.log("Using factory defaults")
            QMessageBox.warning(
                self,
                "Settings Error",
                f"Could not load default settings. Using factory defaults instead.\n\nError details: {str(e)}"
            )

    def update_segmentation_ui(self):
        """Update segmentation UI to reflect current settings"""
        self.silence_threshold_spin.setValue(self.silence_threshold)
        self.min_silence_spin.setValue(self.min_silence)
        self.seek_step_spin.setValue(self.seek_step)
        self.min_time_spin.setValue(self.min_time_between_segments)
        self.min_segment_spin.setValue(self.min_segment_length)
        self.dropout_spin.setValue(self.dropout)
        
        # Update padding controls
        self.pre_padding_spin.setValue(self.pre_segment_padding)
        self.post_padding_spin.setValue(self.post_segment_padding)

    def update_export_ui(self):
        """Update export UI to reflect current settings"""
        self.bitrate_combo.setCurrentText(str(self.export_bitrate))
        self.album_edit.setText(self.export_album)
        self.fade_in_spin.setValue(self.fade_in)
        self.fade_out_spin.setValue(self.fade_out)
        self.save_unsegmented_check.setChecked(self.save_unsegmented)
        self.trim_only_check.setChecked(self.trim_only)
        self.batch_normalize_check.setChecked(self.batch_normalize)

    def reset_settings(self):
        """Reset all settings to Factory defaults"""
        # Default profiles
        self.profiles = {
            "Kirtan (Vocals)": {
                "gain": 0,
                "normalize": {
                    "enabled": True,         # Whether normalization is active
                    "target_level": -3.0,    # Target dB level for peaks (negative number)
                    "headroom": 2.0,         # dB of headroom to preserve
                    "method": "peak"         # 'peak', 'rms', or 'lufs'
                },
                "dynamic_processing": {
                    "enabled": True,  # Enable for vocals
                    "compressor": {
                        "threshold": -18.0,
                        "ratio": 2.5,
                        "attack": 20.0,
                        "release": 250.0,
                    },
                    "limiter": {
                        "threshold": -1.0,
                        "release": 50.0,
                    }
                },
                "low_pass": False,
                "low_pass_freq": 8000,
                "low_pass_db": -12,
                "high_pass": True,
                "high_pass_freq": 120,
                "high_pass_db": -12,
                "pan": 0
            },
            "Tabla": {
                "gain": 0,
                "normalize": {
                    "enabled": True,         # Whether normalization is active
                    "target_level": -4.0,    # Target dB level for peaks (negative number)
                    "headroom": 2.0,         # dB of headroom to preserve
                    "method": "peak"         # 'peak', 'rms', or 'lufs'
                },
                "dynamic_processing": {
                    "enabled": True,  # Enable for tabla
                    "compressor": {
                        "threshold": -20.0,
                        "ratio": 3.5,
                        "attack": 10.0,
                        "release": 150.0,
                    },
                    "limiter": {
                        "threshold": -0.5,
                        "release": 40.0,
                    }
                },
                "low_pass": False,
                "low_pass_freq": 10000,
                "low_pass_db": -12,
                "high_pass": True,
                "high_pass_freq": 80,
                "high_pass_db": -12,
                "pan": 0
            },
            "Sangat (Ambient)": {
                "gain": 0,
                "normalize": {
                    "enabled": True,         # Whether normalization is active
                    "target_level": -2.0,    # Target dB level for peaks (negative number)
                    "headroom": 2.0,         # dB of headroom to preserve
                    "method": "peak"         # 'peak', 'rms', or 'lufs'
                },
                "dynamic_processing": {
                    "enabled": True,  # Enable for ambient
                    "compressor": {
                        "threshold": -24.0,
                        "ratio": 2.0,
                        "attack": 40.0,
                        "release": 400.0,
                    },
                    "limiter": {
                        "threshold": -1.5,
                        "release": 100.0,
                    }
                },
                "low_pass": True,
                "low_pass_freq": 8000,
                "low_pass_db": -12,
                "high_pass": False,
                "high_pass_freq": 60,
                "high_pass_db": -12,
                "pan": 0
            }
        }
        
        # Reset track profile assignments but keep detected inputs
        self.track_profile_assignments = {}
        
        # Default segmentation settings
        self.silence_threshold = 21
        self.min_silence = 4000
        self.seek_step = 2000
        self.min_time_between_segments = 10000
        self.min_segment_length = 15
        self.dropout = 1
        
        # Default padding settings - set to 0 for neutral behavior
        self.pre_segment_padding = 0  # No adjustment to start point by default
        self.post_segment_padding = 0 # No adjustment to end point by default
        
        # Default export settings
        self.export_bitrate = 128
        self.export_artist = ""
        self.export_album = ""
        self.fade_in = 0
        self.fade_out = 0
        self.save_unsegmented = False
        self.trim_only = False
        self.batch_normalize = False
        
        # Default visualization settings
        self.show_waveform = False
        
        # Default processing settings
        self.processing_speed = "Full Speed"
        
        # Update UI
        self.update_profiles_table()
        self.update_track_assignments_table()
        self.update_segmentation_ui()
        self.update_export_ui()
        self.speed_combo.setCurrentText(self.processing_speed)
        
        self.log("Settings reset to defaults")

    def disable_ui_during_processing(self):
        """Disable UI elements during processing"""
        # Main controls
        self.process_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.browse_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        
        # Tab-specific controls
        self.open_folder_button.setEnabled(False)
        self.play_files_button.setEnabled(False)
        self.add_profile_button.setEnabled(False)
        self.save_settings_btn.setEnabled(False)
        self.load_settings_btn.setEnabled(False)
        self.reset_settings_btn.setEnabled(False)
        
        # Mark as processing
        self.is_processing = True

    def enable_ui_after_processing(self):
        """Re-enable UI elements after processing"""
        # Main controls
        self.process_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.browse_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        
        # Tab-specific controls
        self.open_folder_button.setEnabled(True)
        self.play_files_button.setEnabled(True)
        self.add_profile_button.setEnabled(True)
        self.save_settings_btn.setEnabled(True)
        self.load_settings_btn.setEnabled(True)
        self.reset_settings_btn.setEnabled(True)
        
        # Mark as not processing
        self.is_processing = False

    def start_processing(self):
        """Start audio processing"""
        if not self.working_dir:
            self.log("Please select a directory first")
            return
            
        # Check for unassigned input channels
        unassigned_inputs = [input_channel for input_channel in self.detected_inputs 
                            if input_channel not in self.track_profile_assignments]
        
        # Set up a QTimer to periodically update the UI without blocking
        self.ui_update_timer = QTimer()
        self.ui_update_timer.timeout.connect(self.update_ui_during_processing)
        self.ui_update_timer.start(500)  # Update every 500ms
        
        if unassigned_inputs:
            unassigned_str = ", ".join(unassigned_inputs)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText(f"The following input channels have no profile assigned: {unassigned_str}")
            msg.setInformativeText("Do you want to auto-assign profiles to these channels?")
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            result = msg.exec()
            
            if result == QMessageBox.StandardButton.Yes:
                # Auto-assign only unassigned inputs
                for input_channel in unassigned_inputs:
                    if input_channel in ["Tr3", "3"]:
                        self.track_profile_assignments[input_channel] = "Kirtan (Vocals)"
                    elif input_channel in ["Tr1", "5"]:
                        self.track_profile_assignments[input_channel] = "Tabla"
                    elif input_channel in ["LR", "1-2"]:
                        self.track_profile_assignments[input_channel] = "Sangat (Ambient)"
                    else:
                        # Default to Sangat for other channels
                        self.track_profile_assignments[input_channel] = "Sangat (Ambient)"
                self.update_track_assignments_table()
            elif result == QMessageBox.StandardButton.Cancel:
                return
        
        # Disable UI during processing
        self.disable_ui_during_processing()
        self.progress_bar.setValue(0)
        
        # Start button blinking
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.blink_process_button)
        self.blink_timer.start(500)  # Blink every 500ms
        
        # Set initial color
        self.process_button.setStyleSheet("background-color: #ADD8E6;")  # Light blue
        
        # Set initial state
        self.process_button.setProperty("primary", "true")
        self.process_button.setProperty("processing", "true")
        self.process_button.style().unpolish(self.process_button)
        self.process_button.style().polish(self.process_button)
        
        # Start worker thread
        self.worker = ProcessingWorker(self, self.working_dir)
        self.worker.progress_update.connect(self.log)
        self.worker.progress_bar.connect(self.progress_bar.setValue)
        self.worker.processing_finished.connect(self.processing_done)
        self.worker.track_finished.connect(self.mark_track_as_processed)
        self.worker.start()
        
    def blink_process_button(self):
        """Toggle button background color for blinking effect"""
        if not self.is_processing:
            return
            
        # Toggle between processing states
        is_highlighted = self.process_button.property("highlighted") == "true"
        
        if is_highlighted:
            self.process_button.setProperty("highlighted", "false")
        else:
            self.process_button.setProperty("highlighted", "true")
        
        # Force style refresh
        self.process_button.style().unpolish(self.process_button)
        self.process_button.style().polish(self.process_button)
            
    def mark_track_as_processed(self, track_name):
        """Mark a track as processed in the files table"""
        for row in range(self.files_table.rowCount()):
            track_item = self.files_table.item(row, 0)
            if track_item and track_item.text() == track_name:
                # Set all cells in this row to green background
                for col in range(3):
                    item = self.files_table.item(row, col)
                    if item:
                        item.setBackground(QBrush(QColor(200, 255, 200)))  # Pastel green
                
                # Update status cell
                status_item = self.files_table.item(row, 2)
                if status_item:
                    status_item.setText("Processed")
                break

    def stop_processing(self):
        """Stop the processing worker"""
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.stop_processing()
            self.log("Stopping processing... Aborting current operation.")
            self.stop_button.setEnabled(False)
            
    def processing_done(self):
        """Called when processing is complete"""
        # Re-enable UI
        self.enable_ui_after_processing()
        
        # Stop the UI update timer if it exists
        if hasattr(self, 'ui_update_timer') and self.ui_update_timer.isActive():
            self.ui_update_timer.stop()
        
        # Stop blinking and set to green if not stopped by user
        if hasattr(self, 'blink_timer'):
            self.blink_timer.stop()
            
        if not hasattr(self, 'worker') or not self.worker.stop_requested:
            # Set success state
            self.process_button.setProperty("processing", "false")
            self.process_button.setProperty("success", "true")
            self.process_button.setProperty("highlighted", "false")
            self.process_button.style().unpolish(self.process_button)
            self.process_button.style().polish(self.process_button)
            
            # Only log completion once here, not in worker thread
            if not hasattr(self, '_completion_logged'):
                self.log("Processing complete!")
                self._completion_logged = True
                # Reset the flag after a short delay
                QTimer.singleShot(5000, lambda: setattr(self, '_completion_logged', False))
            
            # Show completion dialog with options
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("Processing complete!")
            msg.setInformativeText("What would you like to do next?")
            
            open_folder_button = msg.addButton("Open Output Folder", QMessageBox.ButtonRole.ActionRole)
            play_button = msg.addButton("Play First Track", QMessageBox.ButtonRole.ActionRole)
            close_button = msg.addButton("Close", QMessageBox.ButtonRole.RejectRole)
            
            msg.exec()
            
            if msg.clickedButton() == open_folder_button:
                self.open_output_folder()
            elif msg.clickedButton() == play_button:
                self.play_processed_files()
        else:
            # Reset button state
            self.process_button.setProperty("processing", "false")
            self.process_button.setProperty("highlighted", "false")
            self.process_button.style().unpolish(self.process_button)
            self.process_button.style().polish(self.process_button)
            self.log("Processing stopped by user.")
            
        # Reset button state after a delay if it was successful
        if not hasattr(self, 'worker') or not self.worker.stop_requested:
            QTimer.singleShot(3000, lambda: self.reset_process_button_state())
        
        self.just_processed = True
        
        # Rescan directory to update status
        if self.working_dir:
            self.scan_directory(self.working_dir)
            
    def reset_process_button_state(self):
        """Reset the process button to its default state"""
        self.process_button.setProperty("processing", "false")
        self.process_button.setProperty("success", "false")
        self.process_button.setProperty("highlighted", "false")
        self.process_button.style().unpolish(self.process_button)
        self.process_button.style().polish(self.process_button)

    def log(self, message):
        """Add message to log with better formatting and color coding"""
        # Use 12-hour time format
        timestamp = time.strftime("%I:%M:%S %p", time.localtime())
        
        # Determine message type and apply appropriate color
        if "Error" in message or "error" in message:
            formatted_message = f'<span style="color:red;">[{timestamp}] {message}</span>'
        elif "complete" in message.lower() or "finished" in message.lower() or "completed" in message.lower():
            formatted_message = f'<span style="color:green;">[{timestamp}] {message}</span>'
        else:
            formatted_message = f'[{timestamp}] {message}'
        
        # Add separators only at beginning of track processing or when exporting
        if "Processing " in message and "_" in message and not "track:" in message:
            separator = "-" * 50
            self.progress_text.append(separator)
            self.progress_text.append(formatted_message)
        elif "Exporting " in message or "Created " in message and "segments" in message:
            self.progress_text.append(formatted_message)
            separator = "-" * 50
            self.progress_text.append(separator)
        else:
            self.progress_text.append(formatted_message)
        
        # Scroll to bottom
        self.progress_text.verticalScrollBar().setValue(
            self.progress_text.verticalScrollBar().maximum()
        )
    
    def check_performance_dependencies(self):
        """Check for optional dependencies that enhance performance"""
        missing_deps = []
        performance_tips = []
        
        # Check for NumPy
        try:
            import numpy
            performance_tips.append("NumPy is available - vectorized processing enabled")
        except ImportError:
            missing_deps.append("numpy")
            
        # Check for SciPy
        try:
            from scipy import signal
            performance_tips.append("SciPy is available - optimized filters enabled")
        except ImportError:
            missing_deps.append("scipy")
            
        # Check for PyLoudNorm
        try:
            import pyloudnorm
            performance_tips.append("PyLoudNorm is available - LUFS normalization enabled")
        except ImportError:
            missing_deps.append("pyloudnorm")
        
        # Display performance info
        if performance_tips:
            self.log("Performance optimizations available:")
            for tip in performance_tips:
                self.log(f"  - {tip}")
                
        if missing_deps:
            self.log("Optional performance dependencies not found:")
            self.log(f"  - Missing: {', '.join(missing_deps)}")
            self.log(f"  - Install with: pip install {' '.join(missing_deps)}")

    def update_dropout(self, value):
        self.dropout = value
    
    def update_pre_padding(self, value):
        """Update pre-segment padding in seconds
        
        Negative value: Move start point EARLIER (pulls back start time before vocals)
        Positive value: Move start point LATER (adds delay before vocals)
        """
        self.pre_segment_padding = value
        
    def update_post_padding(self, value):
        """Update post-segment padding in seconds
        
        Positive value: Add time AFTER segment (move end later)
        Negative value: Remove time AFTER segment (move end earlier)
        """
        self.post_segment_padding = value
    
    def update_bitrate(self, value):
        self.export_bitrate = int(value)

    # Add a new method to create the collapsible detail widget
    def create_profile_detail_widget(self, profile_name, settings):
        """Creates a collapsible widget containing all detailed profile settings"""
        detail_container = QWidget()
        main_layout = QVBoxLayout(detail_container)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create tabs for different setting categories
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)
        
        # EQ Tab
        eq_tab = QWidget()
        eq_layout = QGridLayout(eq_tab)
        eq_layout.setContentsMargins(10, 10, 10, 10)
        
        # Low Pass controls
        low_pass_check = QCheckBox("Low Pass Filter")
        low_pass_check.setChecked(settings.get('low_pass', False))
        low_pass_check.stateChanged.connect(
            lambda state, name=profile_name: self.update_profile_low_pass(name, state)
        )
        eq_layout.addWidget(low_pass_check, 0, 0)
        
        # Low pass frequency
        low_pass_freq_spin = QSpinBox()
        low_pass_freq_spin.setRange(1000, 20000)
        low_pass_freq_spin.setSingleStep(500)
        low_pass_freq_spin.setValue(settings.get('low_pass_freq', 8000))
        low_pass_freq_spin.valueChanged.connect(
            lambda value, name=profile_name: self.update_profile_low_pass_freq(name, value)
        )
        eq_layout.addWidget(QLabel("Frequency (Hz):"), 1, 0)
        eq_layout.addWidget(low_pass_freq_spin, 1, 1)
        
        # Low pass dB reduction
        low_pass_db_spin = QSpinBox()
        low_pass_db_spin.setRange(-24, 0)
        low_pass_db_spin.setValue(settings.get('low_pass_db', -12))
        low_pass_db_spin.valueChanged.connect(
            lambda value, name=profile_name: self.update_profile_low_pass_db(name, value)
        )
        eq_layout.addWidget(QLabel("Reduction (dB):"), 2, 0)
        eq_layout.addWidget(low_pass_db_spin, 2, 1)
        
        # Add spacer
        eq_layout.addWidget(QLabel(""), 3, 0)
        
        # High Pass controls
        high_pass_check = QCheckBox("High Pass Filter")
        high_pass_check.setChecked(settings.get('high_pass', False))
        high_pass_check.stateChanged.connect(
            lambda state, name=profile_name: self.update_profile_high_pass(name, state)
        )
        eq_layout.addWidget(high_pass_check, 4, 0)
        
        # High pass frequency
        high_pass_freq_spin = QSpinBox()
        high_pass_freq_spin.setRange(20, 2000)
        high_pass_freq_spin.setSingleStep(10)
        high_pass_freq_spin.setValue(settings.get('high_pass_freq', 200))
        high_pass_freq_spin.valueChanged.connect(
            lambda value, name=profile_name: self.update_profile_high_pass_freq(name, value)
        )
        eq_layout.addWidget(QLabel("Frequency (Hz):"), 5, 0)
        eq_layout.addWidget(high_pass_freq_spin, 5, 1)
        
        # High pass dB reduction
        high_pass_db_spin = QSpinBox()
        high_pass_db_spin.setRange(-24, 0)
        high_pass_db_spin.setValue(settings.get('high_pass_db', -12))
        high_pass_db_spin.valueChanged.connect(
            lambda value, name=profile_name: self.update_profile_high_pass_db(name, value)
        )
        eq_layout.addWidget(QLabel("Reduction (dB):"), 6, 0)
        eq_layout.addWidget(high_pass_db_spin, 6, 1)
        
        # Pan control
        pan_value_label = QLabel(f"{int(settings.get('pan', 0) * 100)}")
        pan_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        pan_slider = QSlider(Qt.Orientation.Horizontal)
        pan_slider.setRange(-100, 100)
        pan_slider.setValue(int(settings.get('pan', 0) * 100))
        pan_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        pan_slider.setTickInterval(50)
        
        # Update label when slider moves
        pan_slider.valueChanged.connect(lambda value, label=pan_value_label: label.setText(f"{value}"))
        pan_slider.valueChanged.connect(
            lambda value, name=profile_name: self.update_profile_pan(name, value / 100)
        )
        
        pan_layout = QHBoxLayout()
        pan_layout.addWidget(QLabel("L"))
        pan_layout.addWidget(pan_slider, 1)
        pan_layout.addWidget(QLabel("R"))
        pan_layout.addWidget(pan_value_label)
        
        eq_layout.addWidget(QLabel("Pan:"), 7, 0)
        eq_layout.addLayout(pan_layout, 7, 1)
        
        tabs.addTab(eq_tab, "EQ & Pan")
        
        # Normalize Tab
        normalize_tab = QWidget()
        normalize_widget = self.create_normalize_widget(0, profile_name, settings.get('normalize', True))
        normalize_layout = QVBoxLayout(normalize_tab)
        normalize_layout.addWidget(normalize_widget)
        tabs.addTab(normalize_tab, "Normalization")
        
        # Dynamic Processing Tab
        dynamic_tab = QWidget()
        dynamic_widget = self.create_dynamic_processing_widget(0, profile_name, settings.get('dynamic_processing', {"enabled": False}))
        dynamic_layout = QVBoxLayout(dynamic_tab)
        dynamic_layout.addWidget(dynamic_widget)
        tabs.addTab(dynamic_tab, "Dynamic Processing")
        
        main_layout.addWidget(tabs)
        
        return detail_container
    
    def update_profiles_table(self):
        """Update the profiles table with current profile settings"""
        # Save current selection
        current_selection = self.selected_profile
        
        self.profiles_table.setRowCount(0)
        row = 0
        
        for name, settings in self.profiles.items():
            self.profiles_table.insertRow(row)
            
            # Profile name
            profile_item = QTableWidgetItem(name)
            profile_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.profiles_table.setItem(row, 0, profile_item)
            
            # Gain
            gain_spin = QSpinBox()
            gain_spin.setRange(-20, 10)
            gain_spin.setValue(settings.get('gain', 0))
            gain_spin.valueChanged.connect(lambda value, name=name: self.update_profile_gain(name, value))
            
            # Create container widget with proper layout
            gain_container = QWidget()
            gain_layout = QHBoxLayout(gain_container)
            gain_layout.setContentsMargins(5, 2, 5, 2)
            gain_layout.addWidget(gain_spin)
            gain_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            self.profiles_table.setCellWidget(row, 1, gain_container)
            
            # Normalization Target
            norm_target_spin = QDoubleSpinBox()
            norm_target_spin.setRange(-30.0, 0.0)
            norm_target_spin.setSingleStep(0.5)
            norm_target_spin.setSuffix(" dB")
            
            # Get the normalization settings or set default
            if isinstance(settings.get('normalize'), dict):
                norm_settings = settings.get('normalize', {})
                norm_target_spin.setValue(norm_settings.get("target_level", -1.0))
            else:
                norm_target_spin.setValue(-1.0)
                
            norm_target_spin.valueChanged.connect(
                lambda value, name=name: self.update_profile_normalize_target(name, value)
            )
            
            # Create container for normalization target
            norm_target_container = QWidget()
            norm_target_layout = QHBoxLayout(norm_target_container)
            norm_target_layout.setContentsMargins(5, 2, 5, 2)
            norm_target_layout.addWidget(norm_target_spin)
            norm_target_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            self.profiles_table.setCellWidget(row, 2, norm_target_container)
            
            # Enabled features summary
            enabled_features = []
            if settings.get('normalize') and isinstance(settings.get('normalize'), dict) and settings.get('normalize').get('enabled', True):
                method = settings.get('normalize', {}).get('method', 'peak')
                enabled_features.append(f"Normalize ({method})")
            if settings.get('low_pass', False):
                enabled_features.append(f"LP {settings.get('low_pass_freq', 8000)}Hz")
            if settings.get('high_pass', False):
                enabled_features.append(f"HP {settings.get('high_pass_freq', 200)}Hz")
            if settings.get('pan', 0) != 0:
                pan_value = int(settings.get('pan', 0) * 100)
                enabled_features.append(f"Pan {pan_value:+d}")
            if settings.get('dynamic_processing', {}).get('enabled', False):
                enabled_features.append("Dynamic")
                
            features_item = QTableWidgetItem(", ".join(enabled_features) if enabled_features else "No processing")
            self.profiles_table.setItem(row, 3, features_item)
            
            # Actions column with edit button
            actions_container = QWidget()
            actions_layout = QHBoxLayout(actions_container)
            actions_layout.setContentsMargins(5, 2, 5, 2)
            
            edit_button = QPushButton("Edit")
            edit_button.clicked.connect(lambda checked, name=name: self.select_profile_for_edit(name))
            actions_layout.addWidget(edit_button)
            
            # Add delete button only for non-default profiles
            if name not in DEFAULT_PROFILES:
                delete_button = QPushButton("Delete")
                delete_button.clicked.connect(lambda checked, name=name: self.delete_profile(name))
                delete_button.setStyleSheet("QPushButton { color: red; }")
                actions_layout.addWidget(delete_button)
            
            actions_container.setLayout(actions_layout)
            self.profiles_table.setCellWidget(row, 4, actions_container)
            
            row += 1
        
        # Restore selection if possible
        if current_selection and current_selection in self.profiles:
            for row in range(self.profiles_table.rowCount()):
                if self.profiles_table.item(row, 0).text() == current_selection:
                    self.profiles_table.selectRow(row)
                    break
    
    def select_profile_for_edit(self, profile_name):
        """Select a profile for editing (programmatically select the row)"""
        for row in range(self.profiles_table.rowCount()):
            if self.profiles_table.item(row, 0).text() == profile_name:
                self.profiles_table.selectRow(row)
                break
    
    def delete_profile(self, profile_name):
        """Delete a user-created profile"""
        if profile_name in DEFAULT_PROFILES:
            QMessageBox.warning(self, "Warning", "Cannot delete default profiles")
            return
            
        # Check if profile is in use
        for channel, assigned_profile in self.track_profile_assignments.items():
            if assigned_profile == profile_name:
                QMessageBox.warning(self, "Warning", f"Cannot delete profile '{profile_name}' as it is assigned to channel '{channel}'")
                return
            
        # Ask for confirmation
        reply = QMessageBox.question(self, 'Delete Profile',
                                   f'Are you sure you want to delete the profile "{profile_name}"?',
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                   QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            # Delete the profile
            if profile_name in self.profiles:
                del self.profiles[profile_name]
                self.log(f"Deleted profile: {profile_name}")
                
                # Clear selection if deleted profile was selected
                if self.selected_profile == profile_name:
                    self.selected_profile = None
                    self.clear_profile_detail_panel()
                
                # Update UI
                self.update_profiles_table()
                self.update_track_assignments_table()
        else:
            # Update UI
            self.update_profiles_table()

    def update_profile_normalize_target(self, profile_name, value):
        """Update the normalization target level for a profile"""
        if profile_name in self.profiles:
            # Ensure normalize is a dictionary
            if not isinstance(self.profiles[profile_name].get('normalize'), dict):
                self.profiles[profile_name]['normalize'] = {
                    "enabled": True,
                    "target_level": value,
                    "headroom": 2.0,
                    "method": "peak"
                }
            else:
                self.profiles[profile_name]['normalize']["target_level"] = value
                
            # Update the enabled features cell
            self.update_profile_enabled_features(profile_name)

    def save_as_default_settings(self):
        """Save current settings as defaults"""
        try:
            # Get the path to default_settings.py
            default_settings_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                'default_settings.py'
            )
            
            # Ensure all profiles have the correct normalize structure
            profiles_to_save = self.profiles.copy()
            for profile_name, profile in profiles_to_save.items():
                if not isinstance(profile.get('normalize'), dict):
                    profile['normalize'] = {
                        "enabled": bool(profile.get('normalize', True)),
                        "target_level": -3.0,
                        "headroom": 2.0,
                        "method": "peak"
                    }
            
            # Prepare all settings to save
            new_settings = {
                "DEFAULT_PROFILES": profiles_to_save,
                "DEFAULT_SEGMENTATION": {
                    "silence_threshold": self.silence_threshold,
                    "min_silence": self.min_silence,
                    "seek_step": self.seek_step,
                    "min_time_between_segments": self.min_time_between_segments,
                    "min_segment_length": self.min_segment_length,
                    "dropout": self.dropout,
                    "pre_segment_padding": self.pre_segment_padding,
                    "post_segment_padding": self.post_segment_padding
                },
                "DEFAULT_EXPORT": {
                    "bitrate": self.export_bitrate,
                    "album": self.export_album,
                    "fade_in": self.fade_in,
                    "fade_out": self.fade_out,
                    "save_unsegmented": self.save_unsegmented,
                    "trim_only": self.trim_only,
                    "batch_normalize": self.batch_normalize
                },
                "DEFAULT_VISUALIZATION": {
                    "show_waveform": self.show_waveform
                },
                "DEFAULT_PROCESSING": {
                    "speed": self.processing_speed,
                    "resource_usage": "auto"
                }
            }
            
            # Convert the settings to a formatted string with Python-style booleans
            def format_value(v):
                if isinstance(v, bool):
                    return str(v)
                elif isinstance(v, (int, float)):
                    return str(v)
                elif isinstance(v, str):
                    return f'"{v}"'
                elif isinstance(v, dict):
                    items = [f'"{k}": {format_value(val)}' for k, val in v.items()]
                    return "{" + ", ".join(items) + "}"
                return str(v)

            settings_str = "# This file is auto-generated. Manual changes may be overwritten.\n\n"
            settings_str += 'SETTINGS_VERSION = "1.0"\n\n'
            for key, value in new_settings.items():
                settings_str += f"{key} = {format_value(value)}\n\n"
            
            # Add the get_default_settings function at the end
            settings_str += """
# Function to get a deep copy of all default settings
def get_default_settings():
    \"\"\"Return a deep copy of all default settings.\"\"\"
    import copy
    
    settings = {
        "profiles": copy.deepcopy(DEFAULT_PROFILES),
        "segmentation": copy.deepcopy(DEFAULT_SEGMENTATION),
        "export": copy.deepcopy(DEFAULT_EXPORT),
        "visualization": copy.deepcopy(DEFAULT_VISUALIZATION),
        "processing": copy.deepcopy(DEFAULT_PROCESSING),
        "version": SETTINGS_VERSION
    }
    
    return settings
"""
            
            # Write the new content
            with open(default_settings_path, 'w') as f:
                f.write(settings_str)
            
            QMessageBox.information(self, "Success", "Default settings have been updated successfully!")
            self.log("Default settings updated successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save default settings: {str(e)}")
            self.log(f"Error saving default settings: {str(e)}")

def slash(dir_path) -> str:
    return "/"

def main():
    # Force proper multiprocessing method for maximum performance
    if sys.platform == 'win32':  # Windows
        multiprocessing.set_start_method('spawn')
    else:  # macOS or Linux
        multiprocessing.set_start_method('fork')
    
    # Rest of main function...
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for consistent look
    
    # Set application-wide font
    app_font = QFont("Arial", 10)
    app.setFont(app_font)
    
    # Load style sheet
    try:
        style_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'style.qss')
        if os.path.exists(style_file):
            with open(style_file, "r") as f:
                app.setStyleSheet(f.read())
    except Exception as e:
        print(f"Error loading style sheet: {e}")
    
    window = KirtanProcessorApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()