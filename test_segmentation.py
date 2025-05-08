#!/usr/bin/env python
"""
Enhanced test script to verify audio segmentation functionality in Kirtan Processor
with real audio files and GUI-equivalent segmentation logic
"""

import os
import sys
import array
import math
import argparse
import time
from pydub import AudioSegment, silence

# Import PyQt for GUI
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QPushButton, QLabel, QFileDialog, QSpinBox, QGroupBox, 
                           QTextEdit, QSlider, QCheckBox, QProgressBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize

# Import GUI-specific segmentation components
from audio.detection import detect_silence_efficiently
from utils.file_utils import format_time

def generate_sine_wave(frequency=440, duration=1000, sample_rate=44100, amplitude=0.5):
    """Generate a sine wave at the given frequency and duration"""
    num_samples = int(sample_rate * (duration / 1000.0))
    samples = array.array('h', [0] * num_samples)
    
    for i in range(num_samples):
        sample = int(amplitude * 32767.0 * math.sin(2 * math.pi * frequency * i / sample_rate))
        samples[i] = sample
    
    return AudioSegment(
        data=samples.tobytes(),
        sample_width=2,  # 2 bytes = 16 bits
        frame_rate=sample_rate,
        channels=1
    )

class SegmentationWorker(QThread):
    progress_update = pyqtSignal(str)
    finished = pyqtSignal(list)
    
    def __init__(self, audio_file, silence_settings):
        super().__init__()
        self.audio_file = audio_file
        self.silence_settings = silence_settings
        
    def run(self):
        try:
            # Load the audio file
            self.progress_update.emit(f"Loading audio file: {self.audio_file}")
            start_time = time.time()
            
            audio = AudioSegment.from_file(self.audio_file)
            load_time = time.time() - start_time
            self.progress_update.emit(f"Loaded audio file in {load_time:.2f} seconds")
            
            # Run the segmentation
            self.progress_update.emit("\nRunning segmentation...")
            segments = self.detect_segments(audio)
            
            # Final summary
            self.progress_update.emit("\nFINAL SEGMENTATION SUMMARY")
            self.progress_update.emit("=========================")
            for i, (start, end) in enumerate(segments):
                duration_ms = end - start
                self.progress_update.emit(f"Segment {i+1}: {format_time(start)} - {format_time(end)} (duration: {format_time(duration_ms)})")
            
            self.finished.emit(segments)
            
        except Exception as e:
            self.progress_update.emit(f"ERROR processing audio file: {str(e)}")
            import traceback
            self.progress_update.emit(traceback.format_exc())
            self.finished.emit([])
    
    def detect_segments(self, audio):
        """
        Detect segments using the same logic as in the GUI ProcessingWorker class
        """
        self.progress_update.emit(f"Audio length: {format_time(len(audio))} ({len(audio)}ms)")
        self.progress_update.emit(f"Audio dBFS: {audio.dBFS:.2f}dB")
        
        dBFS = audio.dBFS
        min_silence = self.silence_settings.get('min_silence', 4000)
        silence_thresh = dBFS - self.silence_settings.get('silence_threshold', 21)
        seek_step = self.silence_settings.get('seek_step', 2000)
        min_time_between = self.silence_settings.get('min_time_between_segments', 10000)  # ms
        min_segment_length_min = self.silence_settings.get('min_segment_length', 15)  # minutes
        min_segment_length = int(min_segment_length_min * 60_000)  # ms
        dropout = self.silence_settings.get('dropout', 60_000)  # ms, drop segments shorter than this

        self.progress_update.emit(f"Using segmentation parameters:")
        self.progress_update.emit(f"- silence_thresh: {silence_thresh:.2f}dB (audio dBFS {dBFS:.2f}dB - threshold offset {self.silence_settings.get('silence_threshold', 21)})")
        self.progress_update.emit(f"- min_silence: {min_silence}ms")
        self.progress_update.emit(f"- seek_step: {seek_step}ms")
        self.progress_update.emit(f"- min_time_between_segments: {min_time_between}ms")
        self.progress_update.emit(f"- min_segment_length: {min_segment_length}ms ({min_segment_length_min} minutes)")
        self.progress_update.emit(f"- dropout: {dropout}ms")
        
        # Step 1: Detect nonsilent regions
        self.progress_update.emit("\nDetecting non-silent regions...")
        raw_segments = silence.detect_nonsilent(
            audio,
            min_silence_len=min_silence,
            silence_thresh=silence_thresh,
            seek_step=seek_step
        )
        
        self.progress_update.emit(f"Initial detected segments: {len(raw_segments)}")
        for idx, (start, end) in enumerate(raw_segments):
            self.progress_update.emit(f"  Segment {idx+1}: {format_time(start)} - {format_time(end)} (duration: {format_time(end-start)})")
        
        # Step 2: Drop short segments before merging
        filtered = [s for s in raw_segments if (s[1] - s[0]) >= dropout]
        self.progress_update.emit(f"\nSegments after dropout filter: {len(filtered)}")
        for idx, (start, end) in enumerate(filtered):
            self.progress_update.emit(f"  Filtered Segment {idx+1}: {format_time(start)} - {format_time(end)} (duration: {format_time(end-start)})")

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
                    self.progress_update.emit(f"  Merged segments: gap={format_time(gap_between)}, prev_duration={format_time(prev_duration)}, curr_duration={format_time(segment_duration)}")
                else:
                    # Otherwise add as a new segment
                    final_segments.append([start, end])
                    self.progress_update.emit(f"  Kept as separate segment: gap={format_time(gap_between)}, prev_duration={format_time(prev_duration)}, curr_duration={format_time(segment_duration)}")
            else:
                final_segments.append([start, end])
                self.progress_update.emit(f"  Added first segment: duration={format_time(segment_duration)}")

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
                        self.progress_update.emit(f"  Merged last segment with previous due to short duration and small gap")

        # Step 5: Final segment debug
        self.progress_update.emit(f"\nFinal segments after merging: {len(final_segments)}")
        for idx, (start, end) in enumerate(final_segments):
            self.progress_update.emit(f"  Final Segment {idx+1}: {format_time(start)} - {format_time(end)} (duration: {format_time(end-start)})")

        # Step 6: Apply padding
        padded_segments = []
        pre_pad = self.silence_settings.get('pre_segment_padding', -3)  # Default 3 seconds pre-padding
        post_pad = self.silence_settings.get('post_segment_padding', 3)  # Default 3 seconds post-padding
        
        self.progress_update.emit(f"\nApplying segment padding: pre={pre_pad}s, post={post_pad}s")
        
        for idx, (start, end) in enumerate(final_segments):
            # Calculate padded values with safety checks
            if pre_pad < 0:
                # Negative pre-padding means move start time forward (add to start)
                padded_start = min(end - 5000, start + abs(pre_pad*1000))
            else:
                # Positive pre-padding means move start time backward (subtract from start)
                padded_start = max(0, start - pre_pad*1000)
                
            if post_pad < 0:
                # Negative post-padding means move end time backward (subtract from end)
                padded_end = max(start + 5000, end - abs(post_pad*1000))
            else:
                # Positive post-padding means move end time forward (add to end)
                padded_end = min(len(audio), end + post_pad*1000)
            
            padded_segments.append((int(padded_start), int(padded_end)))
            self.progress_update.emit(f"  Padded Segment {idx+1}: {format_time(padded_start)} - {format_time(padded_end)} (duration: {format_time(padded_end-padded_start)})")
        
        return padded_segments

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.current_file = None
        self.worker = None
    
    def init_ui(self):
        self.setWindowTitle('Kirtan Processor Segmentation Test')
        self.setMinimumSize(900, 700)
        
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # File selection
        file_group = QGroupBox("Audio File")
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_button = QPushButton("Select File")
        self.file_button.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_button)
        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)
        
        # Parameters
        params_group = QGroupBox("Segmentation Parameters")
        params_layout = QVBoxLayout()
        
        # Parameter: silence_threshold
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Silence Threshold:"))
        self.thresh_spin = QSpinBox()
        self.thresh_spin.setRange(1, 40)
        self.thresh_spin.setValue(21)
        self.thresh_spin.setToolTip("Silence threshold offset from dBFS")
        thresh_layout.addWidget(self.thresh_spin)
        params_layout.addLayout(thresh_layout)
        
        # Parameter: min_silence
        min_silence_layout = QHBoxLayout()
        min_silence_layout.addWidget(QLabel("Min Silence (ms):"))
        self.min_silence_spin = QSpinBox()
        self.min_silence_spin.setRange(100, 10000)
        self.min_silence_spin.setValue(4000)
        self.min_silence_spin.setSingleStep(100)
        self.min_silence_spin.setToolTip("Minimum silence duration in ms")
        min_silence_layout.addWidget(self.min_silence_spin)
        params_layout.addLayout(min_silence_layout)
        
        # Parameter: seek_step
        seek_layout = QHBoxLayout()
        seek_layout.addWidget(QLabel("Seek Step (ms):"))
        self.seek_spin = QSpinBox()
        self.seek_spin.setRange(100, 5000)
        self.seek_spin.setValue(2000)
        self.seek_spin.setSingleStep(100)
        self.seek_spin.setToolTip("Seek step in ms")
        seek_layout.addWidget(self.seek_spin)
        params_layout.addLayout(seek_layout)
        
        # Parameter: min_time_between
        time_between_layout = QHBoxLayout()
        time_between_layout.addWidget(QLabel("Min Time Between Segments (ms):"))
        self.time_between_spin = QSpinBox()
        self.time_between_spin.setRange(1000, 60000)
        self.time_between_spin.setValue(10000)
        self.time_between_spin.setSingleStep(1000)
        self.time_between_spin.setToolTip("Minimum time between segments in ms")
        time_between_layout.addWidget(self.time_between_spin)
        params_layout.addLayout(time_between_layout)
        
        # Parameter: min_segment_length
        min_length_layout = QHBoxLayout()
        min_length_layout.addWidget(QLabel("Min Segment Length (minutes):"))
        self.min_length_spin = QSpinBox()
        self.min_length_spin.setRange(1, 30)
        self.min_length_spin.setValue(15)
        self.min_length_spin.setToolTip("Minimum segment length in minutes")
        min_length_layout.addWidget(self.min_length_spin)
        params_layout.addLayout(min_length_layout)
        
        # Parameter: dropout
        dropout_layout = QHBoxLayout()
        dropout_layout.addWidget(QLabel("Dropout (seconds):"))
        self.dropout_spin = QSpinBox()
        self.dropout_spin.setRange(1, 300)
        self.dropout_spin.setValue(60)
        self.dropout_spin.setToolTip("Drop segments shorter than this (in seconds)")
        dropout_layout.addWidget(self.dropout_spin)
        params_layout.addLayout(dropout_layout)

        # Parameter: pre_segment_padding
        pre_pad_layout = QHBoxLayout()
        pre_pad_layout.addWidget(QLabel("Pre-Segment Padding (seconds):"))
        self.pre_pad_spin = QSpinBox()
        self.pre_pad_spin.setRange(-10, 10)
        self.pre_pad_spin.setValue(-3)
        self.pre_pad_spin.setToolTip("Padding before segment (negative values trim)")
        pre_pad_layout.addWidget(self.pre_pad_spin)
        params_layout.addLayout(pre_pad_layout)
        
        # Parameter: post_segment_padding
        post_pad_layout = QHBoxLayout()
        post_pad_layout.addWidget(QLabel("Post-Segment Padding (seconds):"))
        self.post_pad_spin = QSpinBox()
        self.post_pad_spin.setRange(-10, 10)
        self.post_pad_spin.setValue(3)
        self.post_pad_spin.setToolTip("Padding after segment (negative values trim)")
        post_pad_layout.addWidget(self.post_pad_spin)
        params_layout.addLayout(post_pad_layout)
        
        # Add all parameters to group
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)
        
        # Process button
        self.process_button = QPushButton("Process Audio")
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self.process_audio)
        main_layout.addWidget(self.process_button)
        
        # Progress indicator
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Log output
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        self.setCentralWidget(main_widget)
    
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.ogg *.flac);;All Files (*)"
        )
        
        if file_path:
            self.current_file = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.process_button.setEnabled(True)
            self.log_text.clear()
            self.log_text.append(f"Selected file: {file_path}")
    
    def process_audio(self):
        if not self.current_file:
            self.log_text.append("No file selected!")
            return
        
        # Disable UI during processing
        self.process_button.setEnabled(False)
        self.file_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log_text.clear()
        
        # Get parameters from UI
        silence_settings = {
            'silence_threshold': self.thresh_spin.value(),
            'min_silence': self.min_silence_spin.value(),
            'seek_step': self.seek_spin.value(),
            'min_time_between_segments': self.time_between_spin.value(),
            'min_segment_length': self.min_length_spin.value(),
            'dropout': self.dropout_spin.value() * 1000,  # Convert to ms
            'pre_segment_padding': self.pre_pad_spin.value(),
            'post_segment_padding': self.post_pad_spin.value()
        }
        
        # Create and start worker thread
        self.worker = SegmentationWorker(self.current_file, silence_settings)
        self.worker.progress_update.connect(self.update_log)
        self.worker.finished.connect(self.processing_finished)
        self.worker.start()
    
    def update_log(self, message):
        self.log_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def processing_finished(self, segments):
        # Re-enable UI
        self.progress_bar.setVisible(False)
        self.process_button.setEnabled(True)
        self.file_button.setEnabled(True)
        
        # Final message
        if segments:
            self.log_text.append(f"\nProcessing complete! Found {len(segments)} segments.")
        else:
            self.log_text.append("\nProcessing failed or no segments found.")

def main_cli():
    parser = argparse.ArgumentParser(description='Test Kirtan Processor segmentation with real audio files')
    parser.add_argument('audio_file', type=str, help='Path to audio file for segmentation testing')
    parser.add_argument('--silence-threshold', type=int, default=21, help='Silence threshold offset from dBFS (default: 21)')
    parser.add_argument('--min-silence', type=int, default=4000, help='Minimum silence duration in ms (default: 4000)')
    parser.add_argument('--seek-step', type=int, default=2000, help='Seek step in ms (default: 2000)')
    parser.add_argument('--min-time-between', type=int, default=10000, help='Minimum time between segments in ms (default: 10000)')
    parser.add_argument('--min-segment-length', type=int, default=15, help='Minimum segment length in minutes (default: 15)')
    parser.add_argument('--dropout', type=int, default=60, help='Drop segments shorter than this (in seconds, default: 60)')
    
    args = parser.parse_args()
    
    print("Kirtan Processor Segmentation Test")
    print("=================================")
    print(f"Python version: {sys.version}")
    
    # Check if file exists
    if not os.path.exists(args.audio_file):
        print(f"ERROR: Audio file '{args.audio_file}' not found")
        return
    
    # Create silence settings from arguments
    silence_settings = {
        'silence_threshold': args.silence_threshold,
        'min_silence': args.min_silence,
        'seek_step': args.seek_step,
        'min_time_between_segments': args.min_time_between,
        'min_segment_length': args.min_segment_length,
        'dropout': args.dropout * 1000,  # Convert to ms
    }
    
    print(f"\nLoading audio file: {args.audio_file}")
    start_time = time.time()
    
    try:
        # Load the audio file
        audio = AudioSegment.from_file(args.audio_file)
        load_time = time.time() - start_time
        print(f"Loaded audio file in {load_time:.2f} seconds")
        
        # Create a worker and run detection
        worker = SegmentationWorker(args.audio_file, silence_settings)
        segments = worker.detect_segments(audio)
        
        # Final summary
        print("\nFINAL SEGMENTATION SUMMARY")
        print("=========================")
        for i, (start, end) in enumerate(segments):
            duration_ms = end - start
            print(f"Segment {i+1}: {format_time(start)} - {format_time(end)} (duration: {format_time(duration_ms)})")
        
    except Exception as e:
        print(f"ERROR processing audio file: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check if we should run in GUI mode or CLI mode
    if len(sys.argv) > 1 and sys.argv[1] == '--gui':
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    elif len(sys.argv) > 1:
        main_cli()
    else:
        # Default to GUI mode with no arguments
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())