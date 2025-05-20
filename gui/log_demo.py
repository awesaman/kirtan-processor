#!/usr/bin/env python
"""
Standalone demo for the HTML Log Viewer
Run this file directly to see the HTML log viewer in action.
"""
import sys
import os
import random
import time

# Add the project root to the Python path to find the gui module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from gui.html_log_viewer import HTMLLogPanel

class LogViewerDemo(QMainWindow):
    """Demo window for the HTML log viewer."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kirtan Processor - HTML Log Viewer Demo")
        self.resize(800, 600)
        
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        self.log_panel = HTMLLogPanel(self)
        layout.addWidget(self.log_panel)
        
        # Populate with example log entries
        self.populate_example_logs()
    
    def populate_example_logs(self):
        """Add some example logs to demonstrate the viewer."""
        # Application startup sequence
        self.log_panel.log("Kirtan Processor v1.0 starting...", "INFO")
        self.log_panel.add_detail("Operating system: Windows 10")
        self.log_panel.add_detail("Python version: 3.12.1")
        self.log_panel.add_detail("Loading configuration...")
        
        # Settings loaded
        settings_entry = self.log_panel.log("Settings loaded from default.json", "INFO")
        self.log_panel.add_detail("Debug mode: OFF")
        self.log_panel.add_detail("Output directory: C:\\Users\\user\\Music\\Kirtan")
        self.log_panel.add_detail("MP3 bitrate: 128kbps")
        
        # Processing a track
        track_name = "Arshdeep Singh - Aarti.mp3"
        self.log_panel.log(f"Processing track: {track_name}", "INFO")
        self.log_panel.add_detail(f"File path: C:\\Kirtan\\{track_name}")
        self.log_panel.add_detail(f"File size: 45.7 MB")
        self.log_panel.add_detail(f"Duration: 48:23")
        
        # Segmentation details
        seg_entry = self.log_panel.log("Running segmentation algorithm...", "INFO")
        self.log_panel.add_detail("[DEBUG][Segmentation] Initial detected segments: 6")
        self.log_panel.add_detail("[DEBUG][Segmentation] Merged segments: gap=6000ms, prev_duration=64000ms, curr_duration=296000ms")
        self.log_panel.add_detail("[DEBUG][Segmentation] Merged segments: gap=10000ms, prev_duration=366000ms, curr_duration=10000ms")
        self.log_panel.add_detail("[DEBUG][Segmentation] Merged segments: gap=8000ms, prev_duration=386000ms, curr_duration=48000ms")
        self.log_panel.add_detail("[DEBUG][Segmentation] Merged segments: gap=4000ms, prev_duration=442000ms, curr_duration=554000ms")
        self.log_panel.add_detail("[DEBUG][Segmentation] Kept as separate segment: gap=4000ms, prev_duration=1000000ms, curr_duration=1714000ms")
        self.log_panel.add_detail("[DEBUG][Segmentation] Detected 12.2 minutes of silence (20.1%) at the end of the track")
        
        # Segment results
        self.log_panel.log("Segmentation complete", "SUCCESS")
        self.log_panel.add_detail("Found 2 segments")
        self.log_panel.add_detail("[DEBUG] Segment 1: 196000ms to 1204000ms (duration: 1008000ms)")
        self.log_panel.add_detail("[DEBUG] Segment 2: 1200000ms to 2922000ms (duration: 1722000ms)")
        
        # Exporting segments
        self.log_panel.log("Exporting segments...", "INFO")
        self.log_panel.add_detail("Format: MP3")
        self.log_panel.add_detail("Bitrate: 128kbps")
        
        # Segment 1 export
        self.log_panel.log("Exporting segment 1...", "INFO")
        self.log_panel.add_detail("Creating file: Arshdeep Singh_v1 - Segment 1.mp3")
        self.log_panel.add_detail("Duration: 16:48")
        self.log_panel.add_detail("Export successful")
        
        # Segment 2 export
        self.log_panel.log("Exporting segment 2...", "INFO")
        self.log_panel.add_detail("Creating file: Arshdeep Singh_v1 - Segment 2.mp3")
        self.log_panel.add_detail("Duration: 28:42")
        self.log_panel.add_detail("Export successful")
        
        # Save complete track
        self.log_panel.log("Exporting unsegmented version...", "INFO")
        self.log_panel.add_detail("Creating file: Arshdeep Singh_v1_unsegmented.mp3")
        self.log_panel.add_detail("Duration: 48:23")
        self.log_panel.add_detail("Export successful")
        
        # Processing complete
        self.log_panel.log("Processing complete", "SUCCESS")
        self.log_panel.add_detail("Total segments: 2")
        self.log_panel.add_detail("Processing time: 12.3 seconds")
        self.log_panel.add_detail("Files saved to: C:\\Kirtan\\Arshdeep Singh")
        
        # Example warning
        self.log_panel.log("Warning: Large silence detected at the end of the track", "WARNING")
        self.log_panel.add_detail("Silence duration: 12.2 minutes")
        self.log_panel.add_detail("Suggestion: Consider using 'Trim Silence' option to remove trailing silence")
        
        # Example error (to show error styling)
        self.log_panel.log("Error in a hypothetical situation", "ERROR")
        self.log_panel.add_detail("This is just to demonstrate error styling")
        self.log_panel.add_detail("In a real app, this would contain error details and stack trace")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = LogViewerDemo()
    demo.show()
    sys.exit(app.exec())
