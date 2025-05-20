## HTML Log Viewer Integration for Kirtan Processor

This document provides instructions and example code for integrating the HTML Log Viewer into the main Kirtan Processor application.

### Key Benefits

1. **Collapsible Sections**: Group related log entries together for cleaner display
2. **Color-Coding**: Different log levels are color-coded for better visibility
3. **Improved Readability**: Better formatting with timestamps and log levels
4. **Detail Management**: Expand only the sections you need to see
5. **Interactive UI**: Click to expand/collapse log details

### Integration Steps

To integrate the HTML Log Viewer into the Kirtan Processor, you'll need to:

1. Add the `html_log_viewer.py` module to your codebase
2. Replace the existing QTextEdit-based log system with the new HTML Log Panel
3. Update the log method to use the new system
4. Update any code that directly manipulates the log text

### Example Integration

Here's how you could modify the `main_window.py` file to use the HTML Log Viewer:

```python
# In gui/main_window.py
# Replace the standard and detailed log widgets with HTML Log Panels

# 1. Import the HTMLLogPanel
from gui.html_log_viewer import HTMLLogPanel

# 2. In the init_ui method, replace:
'''
self.standard_log_text = QTextEdit()
self.standard_log_text.setReadOnly(True)
self.detailed_log_text = QTextEdit()
self.detailed_log_text.setReadOnly(True)
self.log_tab_widget.addTab(self.standard_log_text, "Standard Log")
self.log_tab_widget.addTab(self.detailed_log_text, "Detailed Log")
self.log_tab_widget.setCurrentIndex(0)  # Standard log as default
log_layout.addWidget(self.log_tab_widget)
'''

# With:
'''
self.standard_log = HTMLLogPanel(self)
self.detailed_log = HTMLLogPanel(self)
self.log_tab_widget.addTab(self.standard_log, "Standard Log")
self.log_tab_widget.addTab(self.detailed_log, "Detailed Log")
self.log_tab_widget.setCurrentIndex(0)  # Standard log as default
log_layout.addWidget(self.log_tab_widget)
'''

# 3. Replace the clear log buttons:
'''
self.clear_standard_log_button = QPushButton("Clear Standard Log")
self.clear_detailed_log_button = QPushButton("Clear Detailed Log")
self.clear_standard_log_button.clicked.connect(lambda: self.standard_log_text.clear())
self.clear_detailed_log_button.clicked.connect(lambda: self.detailed_log_text.clear())
'''

# With:
'''
self.clear_standard_log_button = QPushButton("Clear Standard Log")
self.clear_detailed_log_button = QPushButton("Clear Detailed Log")
self.clear_standard_log_button.clicked.connect(lambda: self.standard_log.clear_log())
self.clear_detailed_log_button.clicked.connect(lambda: self.detailed_log.clear_log())
'''

# 4. Update the log method:
'''
def log(self, msg, detailed=False):
    """Add a message to the log."""
    # Format with timestamp
    timestamp = time.strftime("%H:%M:%S")
    msg = f"[{timestamp}] {msg}"
    
    # Always add to detailed log
    if hasattr(self, 'detailed_log_text') and self.detailed_log_text:
        self.detailed_log_text.append(msg)
        if hasattr(self, 'detailed_log_buffer'):
            self.detailed_log_buffer.append(msg)
    
    # Only add to standard log if not detailed or if detailed is specifically requested
    if not detailed and hasattr(self, 'standard_log_text') and self.standard_log_text:
        self.standard_log_text.append(msg)
        if hasattr(self, 'standard_log_buffer'):
            self.standard_log_buffer.append(msg)
'''

# With:
'''
def log(self, msg, detailed=False, level="INFO"):
    """Add a message to the log."""
    
    # Determine if this is an error, warning, or success message
    lower_msg = msg.lower()
    if "error" in lower_msg or "exception" in lower_msg or "failed" in lower_msg:
        log_level = "ERROR"
    elif "warning" in lower_msg:
        log_level = "WARNING"
    elif "success" in lower_msg or "complete" in lower_msg:
        log_level = "SUCCESS"
    else:
        log_level = level
    
    # Always add to detailed log
    if hasattr(self, 'detailed_log') and self.detailed_log:
        self.detailed_log.log(msg, log_level)
        if hasattr(self, 'detailed_log_buffer'):
            self.detailed_log_buffer.append(msg)
    
    # Only add to standard log if not detailed
    if not detailed and hasattr(self, 'standard_log') and self.standard_log:
        self.standard_log.log(msg, log_level)
        if hasattr(self, 'standard_log_buffer'):
            self.standard_log_buffer.append(msg)
'''

# 5. For log messages that should be grouped together, use the add_detail method:
'''
# Instead of:
self.log("Processing track: track01.mp3")
self.log("File size: 15.2 MB", detailed=True)
self.log("Duration: 04:23", detailed=True)

# Use:
self.log("Processing track: track01.mp3")
self.detailed_log.add_detail("File size: 15.2 MB")
self.detailed_log.add_detail("Duration: 04:23")
'''
```

### Code Structure

The HTML Log Viewer consists of these main components:

1. **LogEntry**: Represents a single log entry with optional details
2. **HTMLLogViewer**: The raw viewer that renders the HTML
3. **HTMLLogPanel**: A complete panel with the viewer and controls

### Advanced Usage

For more advanced logging needs, you can:

1. **Group Processing Steps**: Create a main log entry for a process and add all details under it
2. **Color-Code by Level**: Use "INFO", "SUCCESS", "WARNING", "ERROR", "DEBUG" levels
3. **Track Operations**: Create log entries for major operations with their details collapsed by default

### Try the Demo

Run the standalone demo to see the HTML Log Viewer in action:

```
python gui/log_demo.py
```

This will give you an idea of how the HTML Log Viewer looks and functions with data similar to your Kirtan Processor logs.
