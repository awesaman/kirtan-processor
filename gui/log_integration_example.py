#!/usr/bin/env python
"""
Example integration of the HTML Log Viewer into Kirtan Processor
"""
import sys
import os
import time
import random

# Add the project root to the Python path to find the gui module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, 
    QWidget, QPushButton, QTabWidget
)
from gui.html_log_viewer import HTMLLogPanel

class LogDemoWindow(QMainWindow):
    """Demo window to show how to integrate the HTML log viewer into Kirtan Processor."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HTML Log Viewer Integration Example")
        self.resize(1000, 700)
        
        # Create central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget for different log views
        self.tab_widget = QTabWidget(self)
        
        # Create HTML log panel for standard logs
        self.standard_log = HTMLLogPanel(self)
        
        # Create HTML log panel for detailed logs
        self.detailed_log = HTMLLogPanel(self)
        
        # Add tabs for standard and detailed logs
        self.tab_widget.addTab(self.standard_log, "Standard Log")
        self.tab_widget.addTab(self.detailed_log, "Detailed Log")
        main_layout.addWidget(self.tab_widget)
        
        # Add buttons to simulate different operations
        button_layout = QVBoxLayout()
        
        # Button to simulate processing a track
        process_track_btn = QPushButton("Simulate Processing Track")
        process_track_btn.clicked.connect(self.simulate_process_track)
        button_layout.addWidget(process_track_btn)
        
        # Button to simulate trimming segments
        trim_btn = QPushButton("Simulate Trim Operation")
        trim_btn.clicked.connect(self.simulate_trim_operation)
        button_layout.addWidget(trim_btn)
        
        # Button to simulate applying fade
        fade_btn = QPushButton("Simulate Fade Operation")
        fade_btn.clicked.connect(self.simulate_fade_operation)
        button_layout.addWidget(fade_btn)
        
        # Button to simulate errors
        error_btn = QPushButton("Simulate Error")
        error_btn.clicked.connect(self.simulate_error)
        button_layout.addWidget(error_btn)
        
        # Add button layout to main layout
        main_layout.addLayout(button_layout)
    
    def log(self, message, level="INFO", details=None, detailed=False):
        """Log a message to both standard and detailed logs."""
        timestamp = time.strftime("%H:%M:%S")
        
        # Always add to detailed log
        entry = self.detailed_log.log(message, level)
        if details:
            if isinstance(details, list):
                for detail in details:
                    self.detailed_log.add_detail(detail)
            else:
                self.detailed_log.add_detail(details)
        
        # Only add to standard log if not detailed or if detailed is specifically requested
        if not detailed:
            self.standard_log.log(message, level)
    
    def log_from_worker(self, message):
        """Receive a log message from a worker thread."""
        # This method would handle messages from a worker thread
        # For demo purposes, we'll just add it to both logs
        self.log(message, "INFO")
    
    def simulate_process_track(self):
        """Simulate processing a track."""
        self.log("Starting track processing", "INFO")
        
        track_name = f"Track {random.randint(1, 10)}.mp3"
        self.log(f"Processing {track_name}", "INFO", [
            f"File size: {random.randint(10, 50)}.{random.randint(1, 9)} MB",
            f"Duration: {random.randint(1, 10)}:{random.randint(10, 59)}",
            "Loading audio data..."
        ])
        
        # Simulate processing steps
        steps = ["Applying normalization", "Detecting silence", "Identifying segments", "Analyzing audio quality"]
        for step in steps:
            # Add a slight delay to simulate processing
            time.sleep(0.5)
            self.log(step, "INFO", f"Processing time: {random.randint(100, 500)} ms")
        
        # Final result
        segments = random.randint(1, 5)
        self.log(f"Processing complete! Found {segments} segments.", "SUCCESS", [
            f"Total processing time: {random.randint(1, 5)}.{random.randint(1, 9)} seconds",
            f"Average segment length: {random.randint(30, 180)} seconds",
            f"Peak audio level: -{random.randint(1, 20)} dB"
        ])
    
    def simulate_trim_operation(self):
        """Simulate trimming segments."""
        segments = random.randint(1, 3)
        self.log(f"Trimming {segments} segment(s)", "INFO")
        
        for i in range(segments):
            segment_name = f"Segment_{i+1}.mp3"
            trim_start = random.randint(1000, 5000)
            trim_end = random.randint(1000, 5000)
            
            self.log(f"Trimming {segment_name}", "INFO", [
                f"Original duration: {random.randint(60, 300)}s",
                f"Trim start: {trim_start/1000}s",
                f"Trim end: {trim_end/1000}s"
            ])
            
            time.sleep(0.5)  # Simulate processing time
            
            new_name = f"Segment_{i+1}_TrimPre-{int(trim_start/1000)}_TrimPost-{int(trim_end/1000)}.mp3"
            self.log(f"Trimmed {segment_name} ({trim_start}ms from start, {trim_end}ms from end)", "INFO")
            self.log(f"Exported new file: {new_name}", "SUCCESS")
        
        self.log(f"Successfully trimmed {segments} segments and saved as new files.", "SUCCESS")
    
    def simulate_fade_operation(self):
        """Simulate applying fade effects to segments."""
        segments = random.randint(1, 3)
        fade_type = random.choice(["Linear", "Logarithmic", "Exponential"])
        fade_in = random.randint(1000, 3000)
        fade_out = random.randint(1000, 3000)
        
        self.log(f"Applying {fade_type} fade to {segments} segment(s)", "INFO", [
            f"Fade in: {fade_in}ms",
            f"Fade out: {fade_out}ms",
            f"Type: {fade_type}"
        ])
        
        for i in range(segments):
            segment_name = f"Segment_{i+1}.mp3"
            
            self.log(f"Processing: {segment_name}", "INFO")
            time.sleep(0.5)  # Simulate processing time
            
            new_name = f"Segment_{i+1}_FadeIn-{int(fade_in/1000)}_FadeOut-{int(fade_out/1000)}_{fade_type}.mp3"
            self.log(f"Applied {fade_type} fade to {segment_name}", "INFO", [
                f"Fade in: {fade_in}ms",
                f"Fade out: {fade_out}ms"
            ])
            self.log(f"Exported new file: {new_name}", "SUCCESS")
        
        self.log(f"Successfully applied fade effects to {segments} segments and saved as new files.", "SUCCESS")
    
    def simulate_error(self):
        """Simulate an error condition."""
        error_types = [
            "File not found",
            "Invalid audio format",
            "Insufficient disk space",
            "Memory allocation error"
        ]
        error_type = random.choice(error_types)
        
        self.log(f"Error detected during operation", "ERROR", [
            f"Error type: {error_type}",
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "Check logs for more details"
        ])
        
        # Add a detailed error message
        error_msg = f"Exception: {error_type}Error - Failed to process file at line 42 in processing_thread.py"
        self.log("Detailed error information", "ERROR", error_msg, detailed=True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LogDemoWindow()
    window.show()
    sys.exit(app.exec())
