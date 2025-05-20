#!/usr/bin/env python
"""
HTML-based Log Viewer with accordion functionality for Kirtan Processor

This module provides an enhanced log viewer with collapsible sections,
color-coding for different log levels, and better readability.
"""
from PyQt6.QtWidgets import (
    QTextBrowser, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QComboBox, QLabel, QFrame
)
from PyQt6.QtGui import QColor, QTextCursor
from PyQt6.QtCore import Qt, QRegularExpression

import time
import re
import uuid

class LogEntry:
    """Represents a single log entry with collapsible details."""
    
    def __init__(self, message, level="INFO", parent_id=None, timestamp=None):
        self.id = str(uuid.uuid4())[:8]
        self.message = message
        self.level = level
        self.timestamp = timestamp or time.strftime("%H:%M:%S")
        self.parent_id = parent_id
        self.details = []
    
    def add_detail(self, message, level="DETAIL"):
        """Add a detail message to this log entry."""
        detail = LogEntry(message, level, parent_id=self.id)
        self.details.append(detail)
        return detail
    
    def to_html(self, collapsed=True):
        """Convert the log entry to HTML format."""
        level_colors = {
            "INFO": "#0066cc",
            "SUCCESS": "#28a745",
            "WARNING": "#ffc107",
            "ERROR": "#dc3545",
            "DEBUG": "#6c757d",
            "DETAIL": "#343a40"
        }
        color = level_colors.get(self.level, "#000000")
        
        if self.parent_id is None:  # This is a top-level entry
            # Determine icon based on collapsed state
            icon = "▶" if collapsed else "▼"
            
            # Main entry
            html = f"""
            <div class="log-entry" data-id="{self.id}">
                <div class="log-header" onclick="toggleDetails('{self.id}')">
                    <span class="toggle-icon">{icon}</span>
                    <span class="timestamp">[{self.timestamp}]</span>
                    <span class="level" style="color: {color};">[{self.level}]</span>
                    <span class="message">{self.message}</span>
                </div>
            """
            
            # Details section
            if self.details:
                display = "none" if collapsed else "block"
                html += f'<div id="details-{self.id}" class="details" style="display: {display};">'
                for detail in self.details:
                    html += f"""
                    <div class="detail-item">
                        <span class="timestamp">[{detail.timestamp}]</span>
                        <span class="level" style="color: {level_colors.get(detail.level, '#000000')};">[{detail.level}]</span>
                        <span class="message">{detail.message}</span>
                    </div>
                    """
                html += '</div>'
            
            html += '</div>'
            return html
        else:
            # This is already a detail item, so just return the message
            return self.message


class HTMLLogViewer(QTextBrowser):
    """An enhanced log viewer with HTML formatting and collapsible sections."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        
        # Enable rich text and HTML support
        self.setAcceptRichText(True)
        
        # Store log entries as objects
        self.log_entries = []
        
        # Current active entry for adding details
        self.current_entry = None
        
        # Initialize with styles and JavaScript
        self.init_html()
    
    def init_html(self):
        """Initialize the HTML document with styles and JavaScript."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            body {
                font-family: Consolas, Monaco, 'Courier New', monospace;
                font-size: 10pt;
                margin: 0;
                padding: 0;
                background-color: #ffffff;
            }
            .log-entry {
                border-bottom: 1px solid #e0e0e0;
                padding: 3px 0;
                margin: 0;
            }
            .log-header {
                cursor: pointer;
                padding: 5px;
            }
            .log-header:hover {
                background-color: #f0f0f0;
            }
            .toggle-icon {
                display: inline-block;
                width: 16px;
                color: #666;
                font-weight: bold;
            }
            .timestamp {
                color: #666;
                margin-right: 6px;
            }
            .level {
                font-weight: bold;
                margin-right: 6px;
            }
            .details {
                margin-left: 20px;
                padding-left: 10px;
                border-left: 1px solid #ccc;
                display: none;
            }
            .detail-item {
                padding: 3px 0;
                color: #333;
            }
        </style>
        <script>
            function toggleDetails(id) {
                var detailsDiv = document.getElementById("details-" + id);
                var logEntry = document.querySelector('.log-entry[data-id="' + id + '"]');
                var toggleIcon = logEntry.querySelector('.toggle-icon');
                
                if (detailsDiv) {
                    if (detailsDiv.style.display === "none") {
                        detailsDiv.style.display = "block";
                        toggleIcon.textContent = "▼";
                    } else {
                        detailsDiv.style.display = "none";
                        toggleIcon.textContent = "▶";
                    }
                }
            }
        </script>
        </head>
        <body>
        <div id="log-container">
        <!-- Log entries will be inserted here -->
        </div>
        </body>
        </html>
        """
        self.setHtml(html)
    
    def append_entry(self, message, level="INFO", timestamp=None):
        """Add a new top-level log entry."""
        entry = LogEntry(message, level, timestamp=timestamp)
        self.log_entries.append(entry)
        self.current_entry = entry
        
        # Insert the entry's HTML at the end of the log container
        self.append_html(entry.to_html())
        
        return entry
    
    def append_detail(self, message, level="DETAIL"):
        """Add a detail to the current entry."""
        if not self.current_entry:
            # If no current entry, create a new one
            return self.append_entry(message, level)
        
        detail = self.current_entry.add_detail(message, level)
        
        # Update the current entry in the display
        # This is a bit inefficient as we're replacing the whole entry,
        # but it's simple and works for demonstration purposes
        self.update_entry_html(self.current_entry)
        
        return detail
    
    def append_html(self, html):
        """Append HTML to the document while preserving scroll position."""
        scrollbar = self.verticalScrollBar()
        at_bottom = scrollbar.value() == scrollbar.maximum()
        
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(html)
        
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())
    
    def update_entry_html(self, entry):
        """Update an existing entry's HTML in the document."""
        # Find the entry in the document and replace it
        # For simplicity in this example, we'll just rerender everything
        # In a production app, you'd use a more efficient approach
        
        content = '<div id="log-container">'
        for log_entry in self.log_entries:
            content += log_entry.to_html(collapsed=False if log_entry.id == entry.id else True)
        content += '</div>'
        
        # Keep the scroll position
        scrollbar = self.verticalScrollBar()
        scroll_pos = scrollbar.value()
        
        # Update the content
        self.setHtml(f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            body {{
                font-family: Consolas, Monaco, 'Courier New', monospace;
                font-size: 10pt;
                margin: 0;
                padding: 0;
                background-color: #ffffff;
            }}
            .log-entry {{
                border-bottom: 1px solid #e0e0e0;
                padding: 3px 0;
                margin: 0;
            }}
            .log-header {{
                cursor: pointer;
                padding: 5px;
            }}
            .log-header:hover {{
                background-color: #f0f0f0;
            }}
            .toggle-icon {{
                display: inline-block;
                width: 16px;
                color: #666;
                font-weight: bold;
            }}
            .timestamp {{
                color: #666;
                margin-right: 6px;
            }}
            .level {{
                font-weight: bold;
                margin-right: 6px;
            }}
            .details {{
                margin-left: 20px;
                padding-left: 10px;
                border-left: 1px solid #ccc;
                display: none;
            }}
            .detail-item {{
                padding: 3px 0;
                color: #333;
            }}
        </style>
        <script>
            function toggleDetails(id) {{
                var detailsDiv = document.getElementById("details-" + id);
                var logEntry = document.querySelector('.log-entry[data-id="' + id + '"]');
                var toggleIcon = logEntry.querySelector('.toggle-icon');
                
                if (detailsDiv) {{
                    if (detailsDiv.style.display === "none") {{
                        detailsDiv.style.display = "block";
                        toggleIcon.textContent = "▼";
                    }} else {{
                        detailsDiv.style.display = "none";
                        toggleIcon.textContent = "▶";
                    }}
                }}
            }}
        </script>
        </head>
        <body>
        {content}
        </body>
        </html>
        """)
        
        # Restore scroll position
        self.verticalScrollBar().setValue(scroll_pos)
    
    def clear(self):
        """Clear all log entries."""
        super().clear()
        self.log_entries = []
        self.current_entry = None
        self.init_html()


class HTMLLogPanel(QWidget):
    """A complete log panel with filters and controls."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create the log viewer
        self.log_viewer = HTMLLogViewer(self)
        
        # Create filter and control elements
        controls_layout = QHBoxLayout()
        
        # Level filter
        self.level_filter = QComboBox(self)
        self.level_filter.addItem("All Levels")
        self.level_filter.addItems(["INFO", "SUCCESS", "WARNING", "ERROR", "DEBUG"])
        controls_layout.addWidget(QLabel("Filter:"))
        controls_layout.addWidget(self.level_filter)
        
        # Spacer
        controls_layout.addStretch(1)
        
        # Clear button
        self.clear_button = QPushButton("Clear Log", self)
        self.clear_button.clicked.connect(self.clear_log)
        controls_layout.addWidget(self.clear_button)
        
        # Separator line
        separator = QFrame(self)
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        
        # Add widgets to main layout
        layout.addLayout(controls_layout)
        layout.addWidget(separator)
        layout.addWidget(self.log_viewer, 1)
    
    def log(self, message, level="INFO", details=None):
        """Add a log entry, optionally with details."""
        entry = self.log_viewer.append_entry(message, level)
        
        # If details are provided, add them
        if details:
            if isinstance(details, str):
                entry.add_detail(details)
            elif isinstance(details, list):
                for detail in details:
                    if isinstance(detail, str):
                        entry.add_detail(detail)
                    elif isinstance(detail, tuple) and len(detail) == 2:
                        entry.add_detail(detail[0], detail[1])
            
            # Refresh the entry in the viewer
            self.log_viewer.update_entry_html(entry)
        
        return entry
    
    def add_detail(self, message, level="DETAIL"):
        """Add a detail message to the current log entry."""
        self.log_viewer.append_detail(message, level)
    
    def clear_log(self):
        """Clear the log viewer."""
        self.log_viewer.clear()


# Example usage
if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    log_panel = HTMLLogPanel()
    log_panel.setWindowTitle("HTML Log Viewer Example")
    log_panel.resize(800, 600)
    
    # Add some example log entries
    log_panel.log("Starting application", "INFO")
    log_panel.add_detail("Initializing components...", "DETAIL")
    log_panel.add_detail("Loading settings from config.json", "DETAIL")
    
    entry = log_panel.log("Processing audio file: track01.mp3", "INFO")
    log_panel.add_detail("File size: 15.2 MB", "DETAIL")
    log_panel.add_detail("Duration: 04:23", "DETAIL")
    log_panel.add_detail("Sample rate: 44100 Hz", "DETAIL")
    
    log_panel.log("Audio analysis complete", "SUCCESS")
    log_panel.add_detail("Found 3 segments", "DETAIL")
    log_panel.add_detail("Silence detection threshold: -40dB", "DETAIL")
    
    log_panel.log("Warning: High noise level detected", "WARNING")
    log_panel.add_detail("Average noise floor: -35dB", "DETAIL")
    log_panel.add_detail("Recommended: Apply noise reduction", "DETAIL")
    
    log_panel.log("Error processing file: track02.mp3", "ERROR")
    log_panel.add_detail("File not found or invalid format", "DETAIL")
    log_panel.add_detail("Exception: FileNotFoundError", "ERROR")
    
    log_panel.show()
    sys.exit(app.exec())
