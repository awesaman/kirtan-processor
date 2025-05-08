#!/usr/bin/env python
import os
import sys
import time
import json
import re
import platform
if platform.system() == 'Windows':
    import winsound
import subprocess

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, 
                             QComboBox, QSpinBox, QTextEdit, QTabWidget,
                             QScrollArea, QGroupBox, QSlider, QCheckBox, 
                             QProgressBar, QSplitter, QGridLayout, QLineEdit,
                             QMessageBox, QFrame, QTableWidget, QTableWidgetItem,
                             QListWidget, QListWidgetItem, QDoubleSpinBox, QMenu, QHeaderView, 
                             QInputDialog, QSizePolicy, QFormLayout, QDialog, 
                             QDialogButtonBox)
from PyQt6.QtCore import Qt, QSize, QSettings, QTimer, QEventLoop, QPoint
from PyQt6.QtGui import QFont, QPixmap, QIcon, QColor, QBrush, QPainter, QPen

from gui.processing_thread import ProcessingWorker
from gui.help_texts import HELP_TEXTS

from config.constants import INPUT_FORMAT, DEFAULT_SETTINGS
from utils.performance import PerformanceTracker
from core.processor import KirtanProcessor
from utils.app_logging import log_section_header, log_chunk_processing_summary

class ExpandableTrackWidget(QWidget):
    """Custom widget for tracks with expandable segment details"""
    
    def __init__(self, track_path, track_name, parent=None):
        super().__init__(parent)
        self.track_path = track_path
        self.track_name = track_name
        self.is_expanded = False
        self.segments = []
        self.segments_loaded = False  # Track if segments have been loaded
        
        # Main layout for the widget
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Header widget (always visible)
        self.header = QWidget()
        self.header_layout = QHBoxLayout(self.header)
        self.header_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create expand/collapse icons
        self.expand_icon = QPixmap(16, 16)
        self.expand_icon.fill(QColor(255, 255, 255, 0))  # Transparent background
        expand_painter = QPainter(self.expand_icon)
        expand_painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        expand_painter.setPen(QPen(QColor(80, 80, 80), 2))
        expand_painter.drawLine(4, 8, 12, 8)  # Horizontal line
        expand_painter.drawLine(8, 4, 8, 12)  # Vertical line (for + icon)
        expand_painter.end()
        
        self.collapse_icon = QPixmap(16, 16)
        self.collapse_icon.fill(QColor(255, 255, 255, 0))  # Transparent background
        collapse_painter = QPainter(self.collapse_icon)
        collapse_painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        collapse_painter.setPen(QPen(QColor(80, 80, 80), 2))
        collapse_painter.drawLine(4, 8, 12, 8)  # Horizontal line only (for - icon)
        collapse_painter.end()
        
        # Expand/collapse button
        self.expand_btn = QPushButton()
        self.expand_btn.setFixedSize(24, 24)  # Slightly larger for better clickability
        self.expand_btn.setIcon(QIcon(self.expand_icon))
        self.expand_btn.setIconSize(QSize(16, 16))
        self.expand_btn.setStyleSheet("QPushButton { border: none; background: transparent; }")
        self.expand_btn.clicked.connect(self.toggle_expand)
        self.header_layout.addWidget(self.expand_btn)
        
        # Track name label
        self.name_label = QLabel(track_name)
        self.name_label.setStyleSheet("font-weight: bold;")
        self.name_label.setWordWrap(True)  # Allow track names to wrap
        self.header_layout.addWidget(self.name_label, 1)  # Stretch factor 1
        
        # Add header to main layout
        self.layout.addWidget(self.header)
        
        # Container for segments (initially hidden)
        self.segments_container = QWidget()
        self.segments_layout = QVBoxLayout(self.segments_container)
        self.segments_layout.setContentsMargins(25, 8, 10, 8)
        self.segments_layout.setSpacing(8)
        self.segments_container.setVisible(False)
        self.layout.addWidget(self.segments_container)
        
        # Set size policies for better resizing behavior
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.segments_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        
        # Create loading indicator for deferred loading
        self.loading_indicator = QLabel("Loading segments...")
        self.loading_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_indicator.setStyleSheet("color: #666; font-style: italic; padding: 10px;")
        
        # Create progress bar for segment loading
        self.loading_progress = QProgressBar()
        self.loading_progress.setRange(0, 100)
        self.loading_progress.setFixedHeight(10)
        self.loading_progress.setTextVisible(False)
        self.loading_progress.setStyleSheet("""
            QProgressBar {
                background-color: #f0f0f0;
                border-radius: 5px;
            }
            QProgressBar::chunk {
                background-color: #4a86e8;
                border-radius: 5px;
            }
        """)
        
        # Add them to layout, but hide initially
        loading_layout = QVBoxLayout()
        loading_layout.addWidget(self.loading_indicator)
        loading_layout.addWidget(self.loading_progress)
        loading_layout.setSpacing(5)
        
        self.loading_widget = QWidget()
        self.loading_widget.setLayout(loading_layout)
        self.segments_layout.addWidget(self.loading_widget)
        self.loading_widget.hide()
        
        # Fast list widget for MP3 segments
        self.segments_list = QListWidget(self.segments_container)
        self.segments_list.setStyleSheet("padding: 5px;")
        self.segments_list.itemDoubleClicked.connect(
            lambda item: self.play_segment(os.path.join(self.track_path, item.text()))
        )
        self.segments_layout.addWidget(self.segments_list)
        self.segments_list.hide()
    
    def sizeHint(self):
        """Provide better size hint for layout managers"""
        if self.is_expanded:
            # When expanded, calculate height based on content plus extra space for padding
            base_height = self.header.sizeHint().height()
            content_height = self.segments_container.sizeHint().height()
            return QSize(self.width(), base_height + content_height + 10)  # Add extra padding
        else:
            # When collapsed, just use header height
            return QSize(self.width(), self.header.sizeHint().height())
    
    def minimumSizeHint(self):
        """Minimum size hint to prevent squeezing"""
        if self.is_expanded:
            # When expanded, ensure minimum height for content visibility
            base_height = self.header.minimumSizeHint().height()
            # Calculate minimum height needed for all segments
            content_height = 0
            for i in range(self.segments_layout.count()):
                widget = self.segments_layout.itemAt(i).widget()
                if widget:
                    content_height += widget.minimumSizeHint().height() + self.segments_layout.spacing()
            
            return QSize(200, base_height + content_height + 16)  # Extra padding
        else:
            # When collapsed, ensure minimum height for header
            return QSize(200, self.header.minimumSizeHint().height())
    
    def toggle_expand(self):
        """Toggle the expanded state of the widget"""
        self.is_expanded = not self.is_expanded
        if self.is_expanded:
            self.segments_container.setVisible(True)
            if not self.segments_loaded:
                self.loading_widget.show()
                QApplication.processEvents()  # Show spinner ASAP
                # Use QTimer to defer loading so UI can update
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(0, self._async_segment_load)
            self._resize_row_to_contents()
        else:
            self.segments_container.setVisible(False)
            self._resize_row_to_contents()
        self.expand_btn.setIcon(QIcon(self.collapse_icon if self.is_expanded else self.expand_icon))
        self.updateGeometry()
        QApplication.processEvents()

    def _async_segment_load(self):
        self.start_segment_loading()
        self._resize_row_to_contents()

    def _resize_row_to_contents(self):
        parent = self.parent()
        while parent:
            if hasattr(parent, "indexAt") and hasattr(parent, "resizeRowToContents"):
                row = parent.indexAt(self.mapTo(parent, QPoint(0, 0))).row()
                if row >= 0:
                    parent.resizeRowToContents(row)
                break
            parent = parent.parent()
    
    def start_segment_loading(self):
        """Start loading segments - optimized for speed by just listing files"""
        if not os.path.exists(self.track_path):
            error_label = QLabel(f"Error: Path not found")
            error_label.setStyleSheet("color: #d32f2f; background-color: #ffebee; padding: 8px; border-radius: 4px; margin: 5px;")
            error_label.setWordWrap(True)
            self.segments_layout.addWidget(error_label)
            self.loading_widget.hide()
            self.segments_loaded = True
            self._resize_row_to_contents()
            return
        try:
            segment_files = [f for f in os.listdir(self.track_path) if f.lower().endswith('.mp3') and 'segment' in f.lower()]
            segment_files.sort()
            self.segments_list.clear()
            for seg in segment_files:
                self.segments_list.addItem(seg)
            self.segments_list.show() if segment_files else self.segments_list.hide()
            self.loading_widget.hide()
            self.segments_loaded = True
            self._resize_row_to_contents()
        except Exception as e:
            error_label = QLabel(f"Error loading segments: {str(e)}")
            error_label.setStyleSheet("color: #d32f2f; background-color: #ffebee; padding: 8px; border-radius: 4px; margin: 5px;")
            error_label.setWordWrap(True)
            self.segments_layout.addWidget(error_label)
            self.loading_widget.hide()
            self.segments_loaded = True
            self._resize_row_to_contents()
    
    def play_segment(self, file_path):
        """Play an audio segment"""
        try:
            if platform.system() == 'Windows':
                os.startfile(file_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.call(['open', file_path])
            else:  # Linux and other Unix-like systems
                subprocess.call(['xdg-open', file_path])
        except Exception as e:
            print(f"Error playing segment: {e}")

class CompletionPopup(QDialog):
    """Dialog that shows when processing is complete"""
    
    def __init__(self, parent=None, processing_time=0, export_path=None):
        super().__init__(parent)
        self.export_path = export_path
        
        # Configure dialog
        self.setWindowTitle("Processing Complete")
        self.setMinimumWidth(400)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Simple success message
        message_label = QLabel("Processing completed successfully!")
        message_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #28a745;")
        message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Format time as mm:ss
        minutes = int(processing_time // 60)
        seconds = int(processing_time % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        # Time taken label
        time_label = QLabel(f"Time taken: {time_str}")
        time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add widgets to layout
        layout.addWidget(message_label)
        layout.addWidget(time_label)
        
        # Add separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Open folder button (only if path provided)
        if self.export_path and os.path.exists(self.export_path):
            open_folder_btn = QPushButton("Open Folder")
            open_folder_btn.clicked.connect(self.open_export_folder)
            button_layout.addWidget(open_folder_btn)
        
        # Open folder button (only if path provided)
        if self.export_path and os.path.exists(self.export_path):
            open_folder_btn = QPushButton("Open Export Folder")
            open_folder_btn.setStyleSheet("background-color: #4a86e8; color: white;")
            open_folder_btn.clicked.connect(self.open_export_folder)
            button_layout.addWidget(open_folder_btn)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
    def open_export_folder(self):
        """Open the export folder in file explorer"""
        if self.export_path and os.path.exists(self.export_path):
            try:
                if platform.system() == 'Windows':
                    os.startfile(self.export_path)
                elif platform.system() == 'Darwin':  # macOS
                    subprocess.call(['open', self.export_path])
                else:  # Linux and other Unix-like systems
                    subprocess.call(['xdg-open', self.export_path])
                # Close the dialog after opening the folder
                self.accept()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not open folder: {str(e)}")

class ProgressDialog(QDialog):
    """Dialog that shows progress during operations like trim and fade"""
    
    def __init__(self, parent=None, title="Processing", message="Processing files..."):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(400)
        self.setMinimumHeight(150)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setModal(True)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Message label
        self.message_label = QLabel(message)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.message_label)
        
        # Current file label
        self.file_label = QLabel("")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.file_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        
        # Status message
        self.status_label = QLabel("Starting...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # This is required to properly display the dialog while processing
        self.show()
        QApplication.processEvents()
    
    def update_progress(self, value, max_value=100):
        """Update progress bar with current value"""
        percent = int((value / max_value) * 100)
        self.progress_bar.setValue(percent)
        QApplication.processEvents()
    
    def update_message(self, message):
        """Update the message text"""
        self.message_label.setText(message)
        QApplication.processEvents()
    
    def update_file(self, filename):
        """Update the current file being processed"""
        self.file_label.setText(filename)
        QApplication.processEvents()
    
    def update_status(self, status):
        """Update status message"""
        self.status_label.setText(status)
        QApplication.processEvents()

class TrackInfoPanel(QGroupBox):
    """Panel showing detailed information about the selected track"""
    
    def __init__(self, parent=None):
        super().__init__("Track Information", parent)
        self.setStyleSheet("QGroupBox { margin-top: 10px; margin-bottom: 5px; border: 1px solid #cccccc; border-radius: 6px; padding: 10px; background: #ffffff; }")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.segment_files = []
        self.ignored_files = []  # Store ignored files separately
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the UI components"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        
        # Create segment management section directly (no tabs)
        segments_layout = QVBoxLayout()
        
        # Move browse controls to the top of the tab
        path_layout = QHBoxLayout()
        self.browse_path_edit = QLineEdit()
        self.browse_path_edit.setReadOnly(True)
        self.browse_path_edit.setPlaceholderText("Select folder containing MP3 segments")
        
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_segments_folder)
        
        path_layout.addWidget(self.browse_path_edit)
        path_layout.addWidget(browse_button)
        segments_layout.addLayout(path_layout)
        
        # Include subfolders checkbox and scan button in horizontal layout
        subfolder_layout = QHBoxLayout()
        self.include_subfolders_checkbox = QCheckBox("Include MP3 files from subfolders")
        scan_button = QPushButton("Scan for MP3 Files")
        scan_button.clicked.connect(self.scan_for_segments)
        subfolder_layout.addWidget(self.include_subfolders_checkbox)
        subfolder_layout.addWidget(scan_button)
        segments_layout.addLayout(subfolder_layout)
        
        # Album and Year common fields
        common_fields_layout = QFormLayout()
        self.common_album_edit = QLineEdit()
        self.common_album_edit.setPlaceholderText("Common album name for all segments")
        self.common_album_edit.textChanged.connect(self.update_common_album)
        
        self.common_year_edit = QLineEdit()
        self.common_year_edit.setPlaceholderText("Common year for all segments")
        self.common_year_edit.textChanged.connect(self.update_common_year)
        
        common_fields_layout.addRow("Album:", self.common_album_edit)
        common_fields_layout.addRow("Year:", self.common_year_edit)
        segments_layout.addLayout(common_fields_layout)
        
        # Track name (moved down but keep it for internal reference)
        self.track_name_label = QLabel("")
        self.track_name_label.setVisible(False)
        segments_layout.addWidget(self.track_name_label)
        
        # Segments table
        self.segments_table = QTableWidget()
        self.segments_table.setColumnCount(5)
        self.segments_table.setHorizontalHeaderLabels(["Filename", "Title", "Artist", "Album", "Year"])
        
        # Set column properties
        self.segments_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)  # Filename
        self.segments_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # Title
        self.segments_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)  # Artist
        self.segments_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)  # Album
        self.segments_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)  # Year
        
        # Connect to the cell change event for filename editing
        self.segments_table.cellChanged.connect(self.on_segment_cell_changed)
        
        # Set context menu policy for right-click menu
        self.segments_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.segments_table.customContextMenuRequested.connect(self.show_segment_context_menu)
        
        segments_layout.addWidget(self.segments_table)
        
        # Create collapsible group for ignored files
        self.ignored_group = QGroupBox("Ignored Files")
        self.ignored_group.setCheckable(True)
        self.ignored_group.setChecked(False)  # Initially collapsed
        
        # Connect the toggle signal to visibility
        self.ignored_group.toggled.connect(self.toggle_ignored_section)
        
        ignored_layout = QVBoxLayout(self.ignored_group)
        
        # Table for ignored files
        self.ignored_table = QTableWidget()
        self.ignored_table.setColumnCount(5)
        self.ignored_table.setHorizontalHeaderLabels(["Filename", "Title", "Artist", "Album", "Year"])
        
        # Set column properties
        self.ignored_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.ignored_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.ignored_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.ignored_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.ignored_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        
        # Set context menu policy for right-click menu on ignored files
        self.ignored_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.ignored_table.customContextMenuRequested.connect(self.show_ignored_context_menu)
        
        ignored_layout.addWidget(self.ignored_table)
        segments_layout.addWidget(self.ignored_group)
        
        # Initial state of ignored section
        self.toggle_ignored_section(False)
        
        # Action button
        action_layout = QHBoxLayout()
        
        # Apply metadata button
        apply_metadata_btn = QPushButton("Apply Metadata to Files")
        apply_metadata_btn.clicked.connect(self.apply_segment_metadata)
        action_layout.addWidget(apply_metadata_btn)
        
        segments_layout.addLayout(action_layout)
        
        # Add segment management layout to main layout with maximized margins
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.addLayout(segments_layout)
        
        # Set initial state
        self.clear_panel()
    
    def toggle_ignored_section(self, checked):
        """Toggle the visibility of the ignored section content"""
        # If unchecked, hide the internal widgets but keep the group box visible
        self.ignored_table.setVisible(checked)
        
        # Update the title with count
        count = self.ignored_table.rowCount()
        self.ignored_group.setTitle(f"Ignored Files ({count})")
    
    def clear_panel(self):
        """Reset the panel to its default state"""
        # Overview information
        self.track_name_label.setText("")
        
        # Segment management fields
        self.browse_path_edit.clear()
        self.common_album_edit.clear()
        self.common_year_edit.clear()
        self.include_subfolders_checkbox.setChecked(False)
        self.segments_table.setRowCount(0)
        self.ignored_table.setRowCount(0)
        self.segment_files = []
        self.ignored_files = []
        
    def update_panel(self, track_path):
        """Update the panel with information about the selected track"""
        if not track_path or not os.path.exists(track_path):
            self.clear_panel()
            return
        
        # Update track name
        self.track_name_label.setText(os.path.basename(track_path))
        
        # Update metadata fields (still needed for other functions)
        self.update_metadata_info(track_path)
        
        # Set browse path to current track path
        self.browse_path_edit.setText(track_path)
        
    def browse_segments_folder(self):
        """Open folder dialog to select a folder containing MP3 segments"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder with MP3 Segments",
            self.browse_path_edit.text() or ""
        )
        
        if folder_path:
            self.browse_path_edit.setText(folder_path)
            
    def scan_for_segments(self):
        """Scan the selected folder for MP3 segment files"""
        folder_path = self.browse_path_edit.text()
        if not folder_path or not os.path.exists(folder_path):
            QMessageBox.warning(self, "Invalid Path", "Please select a valid folder path.")
            return
        
        include_subfolders = self.include_subfolders_checkbox.isChecked()
        
        # Clear the table first
        self.segments_table.setRowCount(0)
        self.ignored_table.setRowCount(0)
        self.segment_files = []
        self.ignored_files = []
        
        # Function to collect MP3 files
        def collect_mp3_files(folder, include_subfolders=False):
            mp3_files = []
            
            try:
                # Get all MP3 files in the current folder
                for filename in os.listdir(folder):
                    filepath = os.path.join(folder, filename)
                    
                    if os.path.isfile(filepath) and filename.lower().endswith('.mp3'):
                        mp3_files.append(filepath)
                    elif include_subfolders and os.path.isdir(filepath):
                        # Recursively scan subfolders if option is enabled
                        mp3_files.extend(collect_mp3_files(filepath, True))
            except Exception as e:
                self.parent().log(f"Error scanning folder: {str(e)}")
                
            return mp3_files
        
        # Collect all MP3 files
        self.segment_files = collect_mp3_files(folder_path, include_subfolders)
        
        if not self.segment_files:
            QMessageBox.information(self, "No MP3 Files", "No MP3 files found in the selected folder.")
            return
        
        # Disconnect signal temporarily to avoid triggering filename updates
        self.segments_table.cellChanged.disconnect(self.on_segment_cell_changed)
        
        try:
            # Import required library for MP3 metadata
            import mutagen
            from mutagen.easyid3 import EasyID3
            mutagen_available = True
        except ImportError:
            mutagen_available = False
            self.parent().log("Warning: Mutagen library not available. Metadata extraction will be limited.")
        
        # Now populate the table
        for row, filepath in enumerate(self.segment_files):
            self.segments_table.insertRow(row)
            
            # Filename (without extension)
            filename = os.path.basename(filepath)
            name_without_ext = os.path.splitext(filename)[0]
            self.segments_table.setItem(row, 0, QTableWidgetItem(name_without_ext))
            
            # Set title to be exactly like the filename
            title = name_without_ext
            artist = ""
            
            # Parse artist from filename if it contains " - " (but keep title exactly as filename)
            if " - " in name_without_ext:
                parts = name_without_ext.split(" - ", 1)
                if len(parts) == 2:
                    artist = parts[0].strip()
            
            # Extract album/year metadata from MP3 file if possible
            album = self.common_album_edit.text()
            year = self.common_year_edit.text()
            
            if mutagen_available:
                try:
                    audio = EasyID3(filepath)
                    # Only use file metadata for artist, album and year but not for title
                    if not artist and not " - " in name_without_ext:
                        artist = audio.get('artist', [""])[0]
                    album = audio.get('album', [album])[0] if not album else album
                    year = audio.get('date', [year])[0] if not year else year
                except Exception as e:
                    # Couldn't read metadata, use parsed values
                    pass
            
            # Add metadata to table
            self.segments_table.setItem(row, 1, QTableWidgetItem(title))
            self.segments_table.setItem(row, 2, QTableWidgetItem(artist))
            self.segments_table.setItem(row, 3, QTableWidgetItem(album))
            self.segments_table.setItem(row, 4, QTableWidgetItem(year))
            
            # Store the full filepath as item data
            self.segments_table.item(row, 0).setData(Qt.ItemDataRole.UserRole, filepath)
        
        # Reconnect signal
        self.segments_table.cellChanged.connect(self.on_segment_cell_changed)
        
        # Show message
        self.parent().log(f"Found {len(self.segment_files)} MP3 files")
    
    def on_segment_cell_changed(self, row, column):
        """Handle edits to segment table cells, including auto-updating title/artist from filename"""
        item = self.segments_table.item(row, column)
        if not item:
            return
        
        # Handle filename column changes
        if column == 0:
            new_name = item.text().strip()
            filepath = item.data(Qt.ItemDataRole.UserRole)
            
            if not filepath or not os.path.exists(filepath):
                self.parent().log(f"Error: File path not found: {filepath}")
                return
                
            # Update title and artist based on filename if it contains " - "
            if " - " in new_name:
                # Temporarily disconnect to avoid triggering this handler again
                self.segments_table.cellChanged.disconnect(self.on_segment_cell_changed)
                
                parts = new_name.split(" - ", 1)
                if len(parts) == 2:
                    artist = parts[0].strip()
                    title = parts[1].strip()
                    
                    # Set title same as filename
                    self.segments_table.setItem(row, 1, QTableWidgetItem(new_name))
                    
                    # Update artist cell
                    self.segments_table.setItem(row, 2, QTableWidgetItem(artist))
                
                # Reconnect signal
                self.segments_table.cellChanged.connect(self.on_segment_cell_changed)
                
            # Handle file renaming
            self.rename_file(row, new_name, filepath)
        
        # Handle title column changes - sync with filename
        if column == 1:
            # Temporarily disconnect to avoid triggering this handler again
            self.segments_table.cellChanged.disconnect(self.on_segment_cell_changed)
            
            # Get the filename item and update it to match title
            filename_item = self.segments_table.item(row, 0)
            if filename_item:
                filepath = filename_item.data(Qt.ItemDataRole.UserRole)
                filename_item.setText(item.text())
                self.rename_file(row, item.text(), filepath)
            
            # Reconnect signal
            self.segments_table.cellChanged.connect(self.on_segment_cell_changed)
    
    def rename_file(self, row, new_name, filepath):
        """Rename a file and update its data in the table"""
        if not filepath or not os.path.exists(filepath):
            self.parent().log(f"Error: File path not found: {filepath}")
            return
            
        # Get file directory and extension
        file_dir = os.path.dirname(filepath)
        _, ext = os.path.splitext(filepath)
        
        # Build new path
        new_path = os.path.join(file_dir, new_name + ext)
        
        # Don't do anything if path hasn't changed
        if filepath == new_path:
            return
            
        # Check if target already exists
        if os.path.exists(new_path):
            QMessageBox.warning(self, "Rename Error", 
                             f"Cannot rename to '{new_name}' - a file with this name already exists.")
            # Revert to original name
            orig_name = os.path.splitext(os.path.basename(filepath))[0]
            
            # Temporarily disconnect to avoid triggering this handler again
            self.segments_table.cellChanged.disconnect(self.on_segment_cell_changed)
            self.segments_table.item(row, 0).setText(orig_name)
            self.segments_table.cellChanged.connect(self.on_segment_cell_changed)
            return
            
        try:
            # Rename the file
            os.rename(filepath, new_path)
            
            # Update the stored path
            self.segments_table.item(row, 0).setData(Qt.ItemDataRole.UserRole, new_path)
            
            # Update the segment_files list
            for i, path in enumerate(self.segment_files):
                if path == filepath:
                    self.segment_files[i] = new_path
                    break
                    
            self.parent().log(f"Renamed file: {os.path.basename(filepath)} → {new_name + ext}")
        except Exception as e:
            QMessageBox.warning(self, "Rename Error", f"Failed to rename file: {str(e)}")
            
            # Revert to original name
            orig_name = os.path.splitext(os.path.basename(filepath))[0]
            
            # Temporarily disconnect to avoid triggering this handler again
            self.segments_table.cellChanged.disconnect(self.on_segment_cell_changed)
            self.segments_table.item(row, 0).setText(orig_name)
            self.segments_table.cellChanged.connect(self.on_segment_cell_changed)
    
    def update_common_album(self, album):
        """Update album field for all items in the table"""
        if self.segments_table.rowCount() == 0:
            return
            
        # Temporarily disconnect the cell changed signal
        self.segments_table.cellChanged.disconnect(self.on_segment_cell_changed)
        
        for row in range(self.segments_table.rowCount()):
            item = self.segments_table.item(row, 3)  # Album column
            if item:
                item.setText(album)
        
        # Update ignored files as well
        for row in range(self.ignored_table.rowCount()):
            item = self.ignored_table.item(row, 3)  # Album column
            if item:
                item.setText(album)
                
        # Reconnect the signal
        self.segments_table.cellChanged.connect(self.on_segment_cell_changed)
    
    def update_common_year(self, year):
        """Update year field for all items in the table"""
        if self.segments_table.rowCount() == 0:
            return
            
        # Temporarily disconnect the cell changed signal
        self.segments_table.cellChanged.disconnect(self.on_segment_cell_changed)
        
        for row in range(self.segments_table.rowCount()):
            item = self.segments_table.item(row, 4)  # Year column
            if item:
                item.setText(year)
        
        # Update ignored files as well
        for row in range(self.ignored_table.rowCount()):
            item = self.ignored_table.item(row, 4)  # Year column
            if item:
                item.setText(year)
                
        # Reconnect the signal
        self.segments_table.cellChanged.connect(self.on_segment_cell_changed)
    
    def show_segment_context_menu(self, position):
        """Show context menu for right-clicking on a segment in the table"""
        context_menu = QMenu(self)
        
        # Get the row under the cursor
        row = self.segments_table.rowAt(position.y())
        if row >= 0:
            self.segments_table.selectRow(row)
            
            # Add context menu options
            play_action = context_menu.addAction("Play Track")
            open_folder_action = context_menu.addAction("Open Containing Folder")
            export_info_action = context_menu.addAction("Export Track Info")
            context_menu.addSeparator()
            ignore_action = context_menu.addAction("Ignore File")
            
            # Execute the menu and get the selected action
            action = context_menu.exec(self.segments_table.viewport().mapToGlobal(position))
            
            if action == play_action:
                self.play_segment(row)
            elif action == open_folder_action:
                self.open_segment_folder(row)
            elif action == export_info_action:
                self.export_segment_info(row)
            elif action == ignore_action:
                self.ignore_segment(row)
    
    def show_ignored_context_menu(self, position):
        """Show context menu for right-clicking on an ignored file"""
        context_menu = QMenu(self)
        
        # Get the row under the cursor
        row = self.ignored_table.rowAt(position.y())
        if row >= 0:
            self.ignored_table.selectRow(row)
            
            # Add context menu options
            unignore_action = context_menu.addAction("Unignore File")
            play_action = context_menu.addAction("Play Track")
            open_folder_action = context_menu.addAction("Open Containing Folder")
            
            # Execute the menu and get the selected action
            action = context_menu.exec(self.ignored_table.viewport().mapToGlobal(position))
            
            if action == unignore_action:
                self.unignore_segment(row)
            elif action == play_action:
                self.play_ignored_segment(row)
            elif action == open_folder_action:
                self.open_ignored_segment_folder(row)
    
    def play_segment(self, row):
        """Play the selected segment from the main table"""
        item = self.segments_table.item(row, 0)
        if not item:
            return
            
        filepath = item.data(Qt.ItemDataRole.UserRole)
        if filepath and os.path.exists(filepath):
            try:
                # Use platform-specific commands to open the default media player
                if platform.system() == 'Windows':
                    os.startfile(filepath)
                elif platform.system() == 'Darwin':  # macOS
                    subprocess.call(['open', filepath])
                else:  # Linux and other Unix-like systems
                    subprocess.call(['xdg-open', filepath])
                self.parent().log(f"Playing file: {os.path.basename(filepath)}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not play the file: {str(e)}")
    
    def open_segment_folder(self, row):
        """Open the folder containing the selected segment"""
        item = self.segments_table.item(row, 0)
        if not item:
            return
            
        filepath = item.data(Qt.ItemDataRole.UserRole)
        if filepath and os.path.exists(filepath):
            folder = os.path.dirname(filepath)
            try:
                # Use platform-specific commands to open the folder
                if platform.system() == 'Windows':
                    os.startfile(folder)
                elif platform.system() == 'Darwin':  # macOS
                    subprocess.call(['open', folder])
                else:  # Linux and other Unix-like systems
                    subprocess.call(['xdg-open', folder])
                self.parent().log(f"Opened folder: {folder}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not open the folder: {str(e)}")
    
    def export_segment_info(self, row):
        """Export information about the selected segment"""
        item = self.segments_table.item(row, 0)
        if not item:
            return
            
        filepath = item.data(Qt.ItemDataRole.UserRole)
        if not filepath or not os.path.exists(filepath):
            return
            
        title_item = self.segments_table.item(row, 1)
        artist_item = self.segments_table.item(row, 2)
        album_item = self.segments_table.item(row, 3)
        year_item = self.segments_table.item(row, 4)
        
        title = title_item.text() if title_item else ""
        artist = artist_item.text() if artist_item else ""
        album = album_item.text() if album_item else ""
        year = year_item.text() if year_item else ""
        
        # Create a report
        report = []
        report.append(f"Track Information Report")
        report.append(f"======================")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"")
        report.append(f"Filename: {os.path.basename(filepath)}")
        report.append(f"Path: {filepath}")
        report.append(f"")
        report.append(f"Metadata:")
        report.append(f"---------")
        report.append(f"Title: {title}")
        report.append(f"Artist: {artist}")
        report.append(f"Album: {album}")
        report.append(f"Year: {year}")
        
        # Write to file
        try:
            # Generate filename
            report_name = f"{os.path.splitext(os.path.basename(filepath))[0]}_info_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            report_path = os.path.join(os.path.dirname(filepath), report_name)
            
            with open(report_path, 'w') as f:
                f.write('\n'.join(report))
                
            QMessageBox.information(self, "Export Successful", 
                                  f"Track information exported to:\n{report_path}")
            self.parent().log(f"Exported track information to: {report_path}")
            
            # Open the report file
            try:
                if platform.system() == 'Windows':
                    os.startfile(report_path)
                elif platform.system() == 'Darwin':  # macOS
                    subprocess.call(['open', report_path])
                else:  # Linux and other Unix-like systems
                    subprocess.call(['xdg-open', report_path])
            except Exception:
                pass  # Ignore errors when opening the file
                
        except Exception as e:
            QMessageBox.warning(self, "Export Failed", 
                              f"Failed to export track information:\n{str(e)}")
    
    def ignore_segment(self, row):
        """Move a segment from the active table to the ignored table"""
        # Disconnect signals to prevent unwanted triggers
        self.segments_table.cellChanged.disconnect(self.on_segment_cell_changed)
        
        # Get data from the row
        items_data = []
        filepath = None
        
        for col in range(self.segments_table.columnCount()):
            item = self.segments_table.item(row, col)
            if item:
                item_data = (item.text(), item.data(Qt.ItemDataRole.UserRole))
                items_data.append(item_data)
                if col == 0:  # Filepath is stored in first column
                    filepath = item.data(Qt.ItemDataRole.UserRole)
        
        if filepath:
            # Add to ignored files list
            self.ignored_files.append(filepath)
            
            # Remove from segment files list
            if filepath in self.segment_files:
                self.segment_files.remove(filepath)
            
            # Insert into ignored table
            new_row = self.ignored_table.rowCount()
            self.ignored_table.insertRow(new_row)
            
            for col, (text, data) in enumerate(items_data):
                new_item = QTableWidgetItem(text)
                if data:
                    new_item.setData(Qt.ItemDataRole.UserRole, data)
                self.ignored_table.setItem(new_row, col, new_item)
            
            # Remove from segments table
            self.segments_table.removeRow(row)
            
            # Update ignored group title and expand if needed
            count = self.ignored_table.rowCount()
            self.ignored_group.setTitle(f"Ignored Files ({count})")
            if not self.ignored_group.isChecked() and count == 1:
                self.ignored_group.setChecked(True)
        
        # Reconnect signals
        self.segments_table.cellChanged.connect(self.on_segment_cell_changed)
    
    def unignore_segment(self, row):
        """Move a segment from the ignored table back to the active table"""
        # Get data from the row
        items_data = []
        filepath = None
        
        for col in range(self.ignored_table.columnCount()):
            item = self.ignored_table.item(row, col)
            if item:
                item_data = (item.text(), item.data(Qt.ItemDataRole.UserRole))
                items_data.append(item_data)
                if col == 0:  # Filepath is stored in first column
                    filepath = item.data(Qt.ItemDataRole.UserRole)
        
        if filepath:
            # Disconnect signals to prevent unwanted triggers
            self.segments_table.cellChanged.disconnect(self.on_segment_cell_changed)
            
            # Add back to segment files list
            self.segment_files.append(filepath)
            
            # Remove from ignored files list
            if filepath in self.ignored_files:
                self.ignored_files.remove(filepath)
            
            # Insert into segments table
            new_row = self.segments_table.rowCount()
            self.segments_table.insertRow(new_row)
            
            for col, (text, data) in enumerate(items_data):
                new_item = QTableWidgetItem(text)
                if data:
                    new_item.setData(Qt.ItemDataRole.UserRole, data)
                self.segments_table.setItem(new_row, col, new_item)
            
            # Remove from ignored table
            self.ignored_table.removeRow(row)
            
            # Update ignored group title
            count = self.ignored_table.rowCount()
            self.ignored_group.setTitle(f"Ignored Files ({count})")
            
            # Reconnect signals
            self.segments_table.cellChanged.connect(self.on_segment_cell_changed)
    
    def play_ignored_segment(self, row):
        """Play an ignored segment"""
        item = self.ignored_table.item(row, 0)
        if not item:
            return
            
        filepath = item.data(Qt.ItemDataRole.UserRole)
        if filepath and os.path.exists(filepath):
            try:
                if platform.system() == 'Windows':
                    os.startfile(filepath)
                elif platform.system() == 'Darwin':  # macOS
                    subprocess.call(['open', filepath])
                else:  # Linux and other Unix-like systems
                    subprocess.call(['xdg-open', filepath])
                self.parent().log(f"Playing file: {os.path.basename(filepath)}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not play the file: {str(e)}")
    
    def open_ignored_segment_folder(self, row):
        """Open the folder containing an ignored segment"""
        item = self.ignored_table.item(row, 0)
        if not item:
            return
            
        filepath = item.data(Qt.ItemDataRole.UserRole)
        if filepath and os.path.exists(filepath):
            folder = os.path.dirname(filepath)
            try:
                if platform.system() == 'Windows':
                    os.startfile(folder)
                elif platform.system() == 'Darwin':  # macOS
                    subprocess.call(['open', folder])
                else:  # Linux and other Unix-like systems
                    subprocess.call(['xdg-open', folder])
                self.parent().log(f"Opened folder: {folder}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not open the folder: {str(e)}")
    
    def apply_segment_metadata(self):
        """Apply metadata to MP3 files"""
        try:
            import mutagen
            from mutagen.easyid3 import EasyID3
        except ImportError:
            QMessageBox.warning(self, "Missing Dependency", 
                             "The mutagen library is required for metadata editing. Please install it with 'pip install mutagen'")
            return
        
        if not self.segment_files:
            QMessageBox.information(self, "No Files", "No files available for metadata update.")
            return
        
        files_to_update = []
        
        # Collect data from active table (non-ignored files)
        for row in range(self.segments_table.rowCount()):
            filepath_item = self.segments_table.item(row, 0)
            title_item = self.segments_table.item(row, 1)
            artist_item = self.segments_table.item(row, 2)
            album_item = self.segments_table.item(row, 3)
            year_item = self.segments_table.item(row, 4)
            
            if filepath_item and filepath_item.data(Qt.ItemDataRole.UserRole):
                filepath = filepath_item.data(Qt.ItemDataRole.UserRole)
                title = title_item.text() if title_item else ""
                artist = artist_item.text() if artist_item else ""
                album = album_item.text() if album_item else ""
                year = year_item.text() if year_item else ""
                
                files_to_update.append({
                    'filepath': filepath,
                    'title': title,
                    'artist': artist,
                    'album': album,
                    'year': year
                })
        
        if not files_to_update:
            QMessageBox.information(self, "No Files Selected", "No files selected for metadata update.")
            return
            
        # Apply metadata to each file
        success_count = 0
        for file_info in files_to_update:
            try:
                # Load or create ID3 tags
                try:
                    audio = EasyID3(file_info['filepath'])
                except mutagen.id3.ID3NoHeaderError:
                    # File doesn't have an ID3 tag, add one
                    mutagen.File(file_info['filepath'], easy=True)
                    audio = EasyID3(file_info['filepath'])
                
                # Update metadata
                if file_info['title']:
                    audio['title'] = file_info['title']
                if file_info['artist']:
                    audio['artist'] = file_info['artist']
                if file_info['album']:
                    audio['album'] = file_info['album']
                if file_info['year']:
                    audio['date'] = file_info['year']
                    
                # Save changes
                audio.save()
                success_count += 1
            except Exception as e:
                self.parent().log(f"Error updating metadata for {os.path.basename(file_info['filepath'])}: {str(e)}")
        
        # Show results
        QMessageBox.information(self, "Metadata Updated", 
                             f"Successfully updated metadata for {success_count} of {len(files_to_update)} files.")
    
    def update_metadata_info(self, track_path):
        """Update the metadata editor with track information"""
        # Look for custom metadata
        custom_metadata_path = os.path.join(track_path, "track_metadata.json")
        if os.path.exists(custom_metadata_path):
            try:
                with open(custom_metadata_path, 'r') as f:
                    metadata = json.load(f)
                    # Set common fields for segments tab
                    self.common_album_edit.setText(metadata.get('album', ''))
                    self.common_year_edit.setText(metadata.get('date', ''))
            except Exception:
                pass
    
    def parent(self):
        """Get the parent KirtanProcessorApp instance"""
        # Walk up the parent tree to find the main window instance
        parent = super().parent()
        while parent and not isinstance(parent, KirtanProcessorApp):
            parent = parent.parent()
        return parent

class KirtanProcessorApp(QMainWindow):
    """Main application window for Kirtan Processor"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kirtan Audio Processor")
        self.updating_profile = False
        self.current_profile_name = None
        
        # Initialize variables
        self.processing_worker = None
        self.audio_files = []
        self.selected_files = []  # Make sure this is initialized
        self.is_processing = False
        self.settings = {}  # Ensure settings is initialized before loading defaults
        self.profiles = {}
        self.track_profile_assignments = {}
        self.initializing = True  # Prevent auto-save during startup
        
        # Create a timer for delayed profile name updates
        self.profile_name_timer = QTimer()
        self.profile_name_timer.setSingleShot(True)
        self.profile_name_timer.timeout.connect(self.apply_profile_name_change)
        self.pending_profile_name = None
        
        # Create context menus once for better performance
        self.initialize_context_menus()
        
        # Load last opened folder from settings if available
        settings = QSettings()
        last_folder = settings.value("last_opened_folder", "")
        self.working_dir = last_folder if last_folder else ""
        # Help icon (must be after QApplication is constructed)
        self.HELP_ICON = QPixmap(16, 16)
        self.HELP_ICON.fill(QColor(255, 255, 255, 0))  # Transparent background
        painter = QPainter(self.HELP_ICON)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(30, 144, 255))  # Dodger blue
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawEllipse(2, 2, 12, 12)
        font = painter.font()
        font.setBold(True)
        font.setPointSize(10)
        painter.setFont(font)
        painter.setPen(QColor(30, 144, 255))
        painter.drawText(self.HELP_ICON.rect(), Qt.AlignmentFlag.AlignCenter, '?')
        painter.end()
        # Load from default.json if available. Do not immediately load QSettings, so default.json always takes precedence on startup.
        self.load_default_json()
        # Initialize UI first so widgets exist
        self.init_ui()
        # If default folder loaded, update UI and scan
        if self.working_dir:
            # Update directory path in UI
            self.dir_path_edit.setText(self.working_dir)
            # Scan default folder to populate files
            self.scan_directory(self.working_dir)
            # Show status message for default folder
            self.statusBar().showMessage(f"Directory selected: {self.working_dir}")
        # Now load profiles into widgets (settings will be loaded after widgets are created in create_settings_tab)
        self.load_profiles_to_ui()
        
        # Do NOT call self.load_profiles() at startup; only use when restoring factory defaults or importing.
        self.initializing = False  # Allow auto-save after startup
        
    def initialize_context_menus(self):
        """Initialize context menus once during app startup"""
        # Create the files table context menu
        self.files_table_context_menu = QMenu(self)
        self.files_table_context_menu.addAction("Process Selected Files")
        self.files_table_context_menu.addAction("Reprocess All Channels")
        self.files_table_context_menu.addAction("View Track Info")
        self.files_table_context_menu.addAction("Open Folder")  # Add new "Open Folder" option
        self.files_table_context_menu.addSeparator()
        
        # Create the segment operations submenu
        segment_menu = self.files_table_context_menu.addMenu("Segment Operations")
        segment_menu.addAction("Trim Segment")
        segment_menu.addAction("Apply Fade to Segment")
        segment_menu.addAction("Export Segment")
        
        self.files_table_context_menu.addSeparator()
        self.files_table_context_menu.addAction("Select All")
        self.files_table_context_menu.addAction("Select None")
        
        # Create the segments context menu for the TrackInfoPanel
        segments_menu = QMenu(self)
        segments_menu.addAction("Play Segment")
        segments_menu.addAction("Trim Segment")
        segments_menu.addAction("Apply Fade")
        segments_menu.addAction("Export Segment")
        
        # Store for use by the TrackInfoPanel
        self.segments_context_menu = segments_menu
        
    def load_default_json(self):
        """Load default profiles and settings from default.json if available"""
        try:
            default_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "default.json")
            
            if os.path.exists(default_path):
                self.log(f"Loading default settings from {default_path}", detailed=True)
                with open(default_path, 'r') as f:
                    defaults = json.load(f)
                    
                    # Load profiles from default.json
                    if "profiles" in defaults:
                        self.profiles = defaults["profiles"]
                        
                    # Load settings from default.json
                    if "settings" in defaults:
                        self.settings = defaults["settings"]
                        
                        # Check if default_folder is present in the settings
                        if "default_folder" in self.settings and self.settings["default_folder"]:
                            default_folder = self.settings["default_folder"]
                            # Make sure path separators are normalized for the current OS
                            default_folder = os.path.normpath(default_folder)
                            if os.path.exists(default_folder):
                                self.working_dir = default_folder
                                self.log(f"Using default folder from default.json: {default_folder}", detailed=True)
                            else:
                                self.log(f"Default folder in default.json doesn't exist: {default_folder}", detailed=True)
                                
                        # Also check for last_opened_folder as an alternative
                        elif "last_opened_folder" in self.settings and self.settings["last_opened_folder"]:
                            last_folder = self.settings["last_opened_folder"]
                            # Make sure path separators are normalized for the current OS
                            last_folder = os.path.normpath(last_folder)
                            if os.path.exists(last_folder):
                                self.working_dir = last_folder
                                self.log(f"Using last opened folder from default.json: {last_folder}", detailed=True)
                            else:
                                self.log(f"Last opened folder in default.json doesn't exist: {last_folder}", detailed=True)
                        
                self.log("Default settings and profiles loaded successfully", detailed=True)
            else:
                self.log("No default.json found, using built-in defaults", detailed=True)
                # Use built-in defaults if no file is found
                self.settings = DEFAULT_SETTINGS
                
        except Exception as e:
            self.log(f"Error loading default.json: {str(e)}")
            # Fall back to built-in defaults
            self.settings = DEFAULT_SETTINGS
            
    def log(self, message, detailed=False):
        """Log a message to the appropriate log tab"""
        timestamp = time.strftime("[%H:%M:%S] ")
        msg = f"{timestamp}{message}"
        
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
        
    def init_ui(self):
        """Initialize the application UI"""
        # Initialize track-related variables
        self.detected_inputs = []
        self.track_profile_assignments = {}
        
        # Set window size and position
        self.resize(1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main vertical layout with two sections:
        # 1. Tab area (top)
        # 2. Fixed control area (bottom)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tabs
        tab_widget = QTabWidget()
        
        # Create processor tab
        processor_tab = QWidget()
        self.create_processor_tab(processor_tab)
        tab_widget.addTab(processor_tab, "Processor")
        
        # Create profiles tab
        profiles_tab = QWidget()
        self.create_profiles_tab(profiles_tab)
        tab_widget.addTab(profiles_tab, "Processing Profiles")
        
        # Create settings tab
        self.create_settings_tab(tab_widget)
        
        # Create metadata tab (renamed from Track Information) and add it last
        metadata_tab = QWidget()
        self.create_track_info_tab(metadata_tab)
        tab_widget.addTab(metadata_tab, "Metadata")
        
        # Add tabs to main layout
        main_layout.addWidget(tab_widget, 3)  # 3:1 ratio for tabs vs bottom fixed area
        
        # Create fixed bottom section that persists across tabs
        bottom_container = QWidget()
        bottom_container.setStyleSheet("QWidget { background-color: #f0f0f0; border-top: 1px solid #d0d0d0; }")
        bottom_layout = QVBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(10, 10, 10, 10)
        
        # Add a header label to the bottom section
        bottom_header = QLabel("Processing Controls")
        bottom_header.setStyleSheet("QLabel { font-weight: bold; font-size: 14px; color: #303030; }")
        bottom_layout.addWidget(bottom_header)
        
        # Create horizontal layout for log and controls
        bottom_row = QHBoxLayout()
        
        # Processing Log (left, 2/3 width)
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)
        
        # Dual log tabs
        self.log_tab_widget = QTabWidget()
        self.standard_log_text = QTextEdit()
        self.standard_log_text.setReadOnly(True)
        self.detailed_log_text = QTextEdit()
        self.detailed_log_text.setReadOnly(True)
        self.log_tab_widget.addTab(self.standard_log_text, "Standard Log")
        self.log_tab_widget.addTab(self.detailed_log_text, "Detailed Log")
        self.log_tab_widget.setCurrentIndex(0)  # Standard log as default
        log_layout.addWidget(self.log_tab_widget)
        
        # Clear buttons for each log
        clear_buttons_layout = QHBoxLayout()
        self.clear_standard_log_button = QPushButton("Clear Standard Log")
        self.clear_detailed_log_button = QPushButton("Clear Detailed Log")
        self.clear_standard_log_button.clicked.connect(lambda: self.standard_log_text.clear())
        self.clear_detailed_log_button.clicked.connect(lambda: self.detailed_log_text.clear())
        clear_buttons_layout.addWidget(self.clear_standard_log_button)
        clear_buttons_layout.addWidget(self.clear_detailed_log_button)
        log_layout.addLayout(clear_buttons_layout)
        
        log_group.setMinimumWidth(400)
        bottom_row.addWidget(log_group, 2)
        
        # Controls & Progress (right, 1/3 width)
        controls_widget = QWidget()
        controls_vlayout = QVBoxLayout(controls_widget)
        controls_vlayout.setSpacing(10)
        
        # Process button
        self.process_button = QPushButton("Process All Track Files")
        self.process_button.setMinimumHeight(36)
        self.process_button.clicked.connect(self.start_processing)
        self.process_button.setStyleSheet("QPushButton { background-color: #8DFF8D; }")  # Default green
        controls_vlayout.addWidget(self.process_button)
        
        # Stop button
        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setStyleSheet("QPushButton { background-color: #FFD8B1; }")  # Pastel orange
        controls_vlayout.addWidget(self.stop_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(18)
        controls_vlayout.addWidget(self.progress_bar)
        
        # CPU usage
        cpu_label = QLabel("CPU:")
        self.cpu_usage_bar = QProgressBar()
        self.cpu_usage_bar.setRange(0, 100)
        self.cpu_usage_bar.setFixedHeight(14)
        controls_vlayout.addWidget(cpu_label)
        controls_vlayout.addWidget(self.cpu_usage_bar)
        
        # Memory usage
        memory_label = QLabel("Memory:")
        self.memory_usage_bar = QProgressBar()
        self.memory_usage_bar.setRange(0, 100)
        self.memory_usage_bar.setFixedHeight(14)
        controls_vlayout.addWidget(memory_label)
        controls_vlayout.addWidget(self.memory_usage_bar)
        
        # Exit button
        self.exit_button = QPushButton("Exit")
        self.exit_button.setMinimumHeight(28)
        self.exit_button.clicked.connect(self.close)
        self.exit_button.setStyleSheet("QPushButton { background-color: #FFB3BA; }")  # Pastel red
        controls_vlayout.addWidget(self.exit_button)
        
        controls_vlayout.addStretch(1)
        bottom_row.addWidget(controls_widget, 1)
        
        # Add bottom row to bottom layout
        bottom_layout.addLayout(bottom_row)
        
        # Add bottom container to main layout
        main_layout.addWidget(bottom_container, 1)  # 1/4 of the space
        
        # Initialize instance variables
        self.working_dir = ""
        self.audio_files = []
        self.selected_files = []
        self.is_processing = False
        # Initialize log buffers for dual logging
        self.standard_log_buffer = []
        self.detailed_log_buffer = []
        
        # Start resource usage monitoring
        self.resource_timer = QTimer()
        self.resource_timer.timeout.connect(self.update_resource_usage)
        self.resource_timer.start(2000)  # Update every 2 seconds
        
        # Add status bar
        self.statusBar().showMessage("Ready")

    def create_processor_tab(self, tab):
        """Create the main processor tab UI"""
        # Main layout
        layout = QVBoxLayout(tab)

        # 1. Directory section (full width)
        dir_group = QGroupBox("Directory")
        dir_layout = QVBoxLayout(dir_group)
        dir_select_layout = QHBoxLayout()
        self.dir_path_edit = QLineEdit()
        self.dir_path_edit.setReadOnly(True)
        self.dir_path_edit.setPlaceholderText("No directory selected")
        self.browse_button = QPushButton("Browse...")
        self.scan_button = QPushButton("Scan Directory")
        self.browse_button.setFixedWidth(120)
        self.scan_button.setFixedWidth(120)
        self.browse_button.clicked.connect(self.browse_directory)
        self.scan_button.clicked.connect(lambda: self.scan_directory(self.working_dir))
        dir_select_layout.addWidget(self.dir_path_edit)
        dir_select_layout.addWidget(self.browse_button)
        dir_select_layout.addWidget(self.scan_button)
        self.set_default_folder_button = QPushButton("Set Default Folder")
        self.set_default_folder_button.setFixedWidth(140)
        self.set_default_folder_button.clicked.connect(self.set_default_browse_folder)
        dir_select_layout.addWidget(self.set_default_folder_button)
        dir_layout.addLayout(dir_select_layout)
        layout.addWidget(dir_group)

        # Add extra spacing after Directory section
        layout.addSpacing(8)

        # 2. Channel Profiles section (mapping input channels to processing profiles)
        profiles_group = QGroupBox("Channel Profiles")
        profiles_group.setStyleSheet("QGroupBox { margin-top: 10px; margin-bottom: 10px; border: 1px solid #cccccc; border-radius: 6px; padding: 8px 8px 8px 8px; background: #fafbfc; } ")
        profiles_group.setMinimumHeight(200)  # Fixed minimum height
        profiles_layout = QVBoxLayout(profiles_group)
        
        # Add description label
        description_label = QLabel("Assign profiles to input channels (TR1, TR2, etc.). Each track will have its channels processed with the selected profiles.")
        description_label.setWordWrap(True)
        description_label.setStyleSheet("font-style: italic; color: #666666; margin-bottom: 5px;")
        profiles_layout.addWidget(description_label)

        self.track_assignments_table = QTableWidget()
        self.track_assignments_table.setColumnCount(2)
        self.track_assignments_table.setHorizontalHeaderLabels(["Input Channel", "Profile"])
        self.track_assignments_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.track_assignments_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.track_assignments_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        profiles_layout.addWidget(self.track_assignments_table)
        layout.addWidget(profiles_group)
        layout.addSpacing(8)  # Reduced spacing to give more room to Track Files

        # 3. Track Files section - This will expand vertically with the window
        files_group = QGroupBox("Track Files")
        files_group.setStyleSheet("QGroupBox { margin-top: 10px; margin-bottom: 10px; border: 1px solid #cccccc; border-radius: 6px; padding: 8px 8px 8px 8px; background: #f7f7fa; } ")
        files_layout = QVBoxLayout(files_group)
        self.files_table = QTableWidget()
        self.files_table.setColumnCount(3)
        self.files_table.setHorizontalHeaderLabels(["Track", "Inputs", "Status"])
        
        # Update column sizing for better visual display
        self.files_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)  # Track column stretches with window
        self.files_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # Inputs column fits content
        self.files_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Status column fits content
        
        # Set minimum row height to ensure cells aren't squeezed vertically
        self.files_table.verticalHeader().setDefaultSectionSize(40)
        self.files_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        
        # Make sure the horizontal header is visible
        self.files_table.horizontalHeader().setVisible(True)
        
        # Give the table a minimum width to prevent squished columns
        self.files_table.setMinimumWidth(400)
        
        self.files_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.files_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.files_table.itemSelectionChanged.connect(self.update_selection)
        # Context menu for right-click
        self.files_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.files_table.customContextMenuRequested.connect(self.show_files_table_context_menu)
        self.files_table.itemChanged.connect(self.handle_track_name_edit)
        files_layout.addWidget(self.files_table)
        layout.addWidget(files_group, 1)  # Give this widget a stretch factor of 1

        # Add stretch at the end to push everything up
        layout.addStretch()

    def create_track_info_tab(self, tab):
        """Create the Track Information tab UI"""
        # Main layout
        layout = QVBoxLayout(tab)
        
        # Create the track info panel in the tab
        self.track_info_panel = TrackInfoPanel()
        
        # Add the track info panel to the tab
        layout.addWidget(self.track_info_panel)
        layout.addStretch(1)  # Add stretch to keep everything at the top

    def create_profiles_tab(self, tab):
        """Create the profiles tab where users can manage and edit processing profiles"""
        # Main layout
        layout = QVBoxLayout(tab)
        
        # Create split view with profiles list on left, editor on right
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Profiles list panel
        profiles_panel = QWidget()
        profiles_layout = QVBoxLayout(profiles_panel)
        
        # Add profile list widget
        profiles_label = QLabel("Processing Profiles")
        profiles_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        profiles_layout.addWidget(profiles_label)
        
        self.profiles_list = QListWidget()
        self.profiles_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.profiles_list.currentItemChanged.connect(self.on_profile_selected)
        profiles_layout.addWidget(self.profiles_list)
        
        # Profile actions panel
        actions_layout = QHBoxLayout()
        self.add_profile_btn = QPushButton("New Profile")
        self.add_profile_btn.clicked.connect(self.add_new_profile)
        self.copy_profile_btn = QPushButton("Copy Profile")
        self.copy_profile_btn.clicked.connect(self.copy_selected_profile)
        self.delete_profile_btn = QPushButton("Delete Profile")
        self.delete_profile_btn.clicked.connect(self.delete_selected_profile)
        
        actions_layout.addWidget(self.add_profile_btn)
        actions_layout.addWidget(self.copy_profile_btn)
        actions_layout.addWidget(self.delete_profile_btn)
        profiles_layout.addLayout(actions_layout)
        
        # Add import/export buttons
        io_layout = QHBoxLayout()
        self.import_profiles_btn = QPushButton("Import Profiles")
        self.import_profiles_btn.clicked.connect(self.import_profiles)
        self.export_profiles_btn = QPushButton("Export Profiles")
        self.export_profiles_btn.clicked.connect(self.export_profiles)
        io_layout.addWidget(self.import_profiles_btn)
        io_layout.addWidget(self.export_profiles_btn)
        profiles_layout.addLayout(io_layout)
        
        # Right side: Profile editor panel
        editor_panel = QScrollArea()
        editor_panel.setWidgetResizable(True)
        editor_panel.setFrameShape(QFrame.Shape.NoFrame)
        
        editor_content = QWidget()
        self.editor_layout = QVBoxLayout(editor_content)
        
        # Profile name and settings
        self.profile_name_edit = QLineEdit()
        self.profile_name_edit.setPlaceholderText("Profile Name")
        self.profile_name_edit.textChanged.connect(self.on_profile_name_changed)
        
        # Create editor form
        self.editor_form = QFormLayout()
        self.editor_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.editor_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        
        # Profile description
        self.profile_description = QLineEdit()
        self.profile_description.setPlaceholderText("Short description of this profile")
        self.profile_description.textChanged.connect(self.on_profile_changed)
        self.editor_form.addRow("Description:", self.profile_description)
        
        # Basic Processing Settings
        # Add a header
        basic_label = QLabel("Basic Settings")
        basic_label.setStyleSheet("font-weight: bold; margin-top: 15px;")
        self.editor_layout.addWidget(self.profile_name_edit)
        self.editor_layout.addWidget(basic_label)
        
        # Gain control
        gain_layout = QHBoxLayout()
        self.gain_input = QDoubleSpinBox()
        self.gain_input.setRange(-60.0, 60.0)
        self.gain_input.setValue(0.0)
        self.gain_input.setSingleStep(0.5)
        self.gain_input.setDecimals(1)
        self.gain_input.setSuffix(" dB")
        self.gain_input.valueChanged.connect(self.on_gain_changed)
        
        gain_layout.addWidget(self.gain_input)
        
        self.editor_form.addRow("Gain:", gain_layout)
        
        # Normalization Settings
        self.norm_group = QGroupBox("Normalization")
        self.norm_group.setCheckable(True)
        self.norm_group.setChecked(True)
        self.norm_group.toggled.connect(self.on_profile_changed)
        
        norm_layout = QFormLayout()
        self.norm_method = QComboBox()
        self.norm_method.addItems(["Peak", "RMS", "LUFS"])
        self.norm_method.currentIndexChanged.connect(self.on_profile_changed)
        
        self.norm_target = QDoubleSpinBox()
        self.norm_target.setRange(-30.0, 0.0)
        self.norm_target.setValue(-3.0)
        self.norm_target.setSingleStep(0.5)
        self.norm_target.setDecimals(1)
        self.norm_target.setSuffix(" dB")
        self.norm_target.valueChanged.connect(self.on_profile_changed)
        
        norm_layout.addRow("Method:", self.norm_method)
        norm_layout.addRow("Target Level:", self.norm_target)
        self.norm_group.setLayout(norm_layout)
        self.editor_form.addRow("", self.norm_group)
        
        # Dynamics Processing
        self.dynamics_group = QGroupBox("Dynamics Processing")
        self.dynamics_group.setCheckable(True)
        self.dynamics_group.setChecked(False)
        self.dynamics_group.toggled.connect(self.on_profile_changed)
        
        dynamics_layout = QVBoxLayout()
        
        # Compressor settings
        comp_form = QFormLayout()
        self.comp_threshold = QDoubleSpinBox()
        self.comp_threshold.setRange(-60.0, 0.0)
        self.comp_threshold.setValue(-18.0)
        self.comp_threshold.setSingleStep(1.0)
        self.comp_threshold.setDecimals(1)
        self.comp_threshold.setSuffix(" dB")
        self.comp_threshold.valueChanged.connect(self.on_profile_changed)
        
        self.comp_ratio = QDoubleSpinBox()
        self.comp_ratio.setRange(1.0, 20.0)
        self.comp_ratio.setValue(2.5)
        self.comp_ratio.setSingleStep(0.1)
        self.comp_ratio.setDecimals(1)
        self.comp_ratio.setSuffix(":1")
        self.comp_ratio.valueChanged.connect(self.on_profile_changed)
        
        self.comp_attack = QDoubleSpinBox()
        self.comp_attack.setRange(1.0, 200.0)
        self.comp_attack.setValue(20.0)
        self.comp_attack.setSingleStep(5.0)
        self.comp_attack.setDecimals(1)
        self.comp_attack.setSuffix(" ms")
        self.comp_attack.valueChanged.connect(self.on_profile_changed)
        
        self.comp_release = QDoubleSpinBox()
        self.comp_release.setRange(10.0, 1000.0)
        self.comp_release.setValue(250.0)
        self.comp_release.setSingleStep(10.0)
        self.comp_release.setDecimals(1)
        self.comp_release.setSuffix(" ms")
        self.comp_release.valueChanged.connect(self.on_profile_changed)
        
        comp_form.addRow("Threshold:", self.comp_threshold)
        comp_form.addRow("Ratio:", self.comp_ratio)
        comp_form.addRow("Attack:", self.comp_attack)
        comp_form.addRow("Release:", self.comp_release)
        
        # Limiter settings
        self.limiter_group = QGroupBox("Limiter")
        self.limiter_group.setCheckable(True)
        self.limiter_group.setChecked(False)
        self.limiter_group.toggled.connect(self.on_profile_changed)
        
        limiter_form = QFormLayout()
        self.limiter_threshold = QDoubleSpinBox()
        self.limiter_threshold.setRange(-12.0, 0.0)
        self.limiter_threshold.setValue(-1.0)
        self.limiter_threshold.setSingleStep(0.1)
        self.limiter_threshold.setDecimals(1)
        self.limiter_threshold.valueChanged.connect(self.on_profile_changed)
        
        self.limiter_release = QDoubleSpinBox()
        self.limiter_release.setRange(10.0, 500.0)
        self.limiter_release.setValue(50.0)
        self.limiter_release.setSingleStep(10.0)
        self.limiter_release.setDecimals(1)
        self.limiter_release.valueChanged.connect(self.on_profile_changed)
        
        limiter_form.addRow("Threshold:", self.limiter_threshold)
        limiter_form.addRow("Release:", self.limiter_release)
        self.limiter_group.setLayout(limiter_form)
        
        dynamics_layout.addLayout(comp_form)
        dynamics_layout.addWidget(self.limiter_group)
        self.dynamics_group.setLayout(dynamics_layout)
        self.editor_form.addRow("", self.dynamics_group)
        
        # EQ Settings
        self.eq_group = QGroupBox("Equalization")
        self.eq_group.setCheckable(True)
        self.eq_group.setChecked(False)
        self.eq_group.toggled.connect(self.on_profile_changed)
        
        eq_layout = QFormLayout()
        
        self.hp_filter = QSpinBox()
        self.hp_filter.setRange(20, 500)
        self.hp_filter.setValue(100)
        self.hp_filter.setSingleStep(10)
        self.hp_filter.setSuffix(" Hz")
        self.hp_filter.valueChanged.connect(self.on_profile_changed)
        
        self.lp_filter = QSpinBox()
        self.lp_filter.setRange(1000, 20000)
        self.lp_filter.setValue(10000)
        self.lp_filter.setSingleStep(500)
        self.lp_filter.setSuffix(" Hz")
        self.lp_filter.valueChanged.connect(self.on_profile_changed)
        
        eq_layout.addRow("High Pass Filter:", self.hp_filter)
        eq_layout.addRow("Low Pass Filter:", self.lp_filter)
        self.eq_group.setLayout(eq_layout)
        self.editor_form.addRow("", self.eq_group)
        
        # Add save button
        self.save_btn = QPushButton("Set Current Profile as Default")
        self.save_btn.clicked.connect(self.save_current_profile)
        self.save_btn.setStyleSheet("background-color: #8DFF8D;")
        
        # Add reset button
        self.reset_btn = QPushButton("Reset to Default")
        self.reset_btn.clicked.connect(self.reset_current_profile)
        
        # Add save all profiles button
        self.save_all_profiles_btn = QPushButton("Set All Profiles as Default")
        self.save_all_profiles_btn.clicked.connect(self.save_all_profiles)
        self.save_all_profiles_btn.setStyleSheet("background-color: #8DFF8D;")
        
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.reset_btn)
        buttons_layout.addStretch(1)
        buttons_layout.addWidget(self.save_btn)
        buttons_layout.addWidget(self.save_all_profiles_btn)
        
        self.editor_layout.addLayout(self.editor_form)
        self.editor_layout.addStretch(1)
        self.editor_layout.addLayout(buttons_layout)
        
        editor_panel.setWidget(editor_content)
        
        # Add to splitter and main layout
        splitter.addWidget(profiles_panel)
        splitter.addWidget(editor_panel)
        splitter.setSizes([300, 700])  # Initial sizes
        
        layout.addWidget(splitter)

    def create_settings_tab(self, tab_widget):
        """Create the settings tab with app configuration options"""
        settings_tab = QWidget()
        layout = QVBoxLayout(settings_tab)
        
        # Using scroll area for settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # --- Segmentation Settings ---
        segment_group = QGroupBox("Segmentation Settings")
        segment_layout = QFormLayout()
        
        # Silence threshold
        self.settings_silence_threshold = QSpinBox()
        self.settings_silence_threshold.setRange(10, 40)
        self.settings_silence_threshold.setSingleStep(1)
        self.settings_silence_threshold.setValue(self.settings.get("silence_threshold", 17))
        self.settings_silence_threshold.valueChanged.connect(lambda val: self.update_setting("silence_threshold", val))
        thresh_help = self.create_help_button("Silence detection threshold in dB below input level. Higher values (e.g. 20-30) detect quieter parts as silence.")
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(self.settings_silence_threshold)
        thresh_layout.addWidget(thresh_help)
        segment_layout.addRow("Silence Threshold (-dB):", thresh_layout)
        
        # Min silence duration
        self.settings_min_silence = QSpinBox()
        self.settings_min_silence.setRange(0, 60)
        self.settings_min_silence.setSingleStep(1)
        self.settings_min_silence.setValue(self.settings.get("min_silence", 4000) // 1000)  # Convert from ms to seconds
        self.settings_min_silence.setSuffix(" sec")
        self.settings_min_silence.valueChanged.connect(lambda val: self.update_setting("min_silence", val * 1000))  # Convert seconds to ms for storage
        silence_help = self.create_help_button("Minimum duration of silence required to consider it a segment boundary")
        silence_layout = QHBoxLayout()
        silence_layout.addWidget(self.settings_min_silence)
        silence_layout.addWidget(silence_help)
        segment_layout.addRow("Min Silence Duration:", silence_layout)
        
        # Seek step
        self.settings_seek_step = QSpinBox()
        self.settings_seek_step.setRange(0, 60)
        self.settings_seek_step.setSingleStep(1) 
        self.settings_seek_step.setValue(self.settings.get("seek_step", 2000) // 1000)  # Convert ms to seconds
        self.settings_seek_step.setSuffix(" sec")
        self.settings_seek_step.valueChanged.connect(lambda val: self.update_setting("seek_step", val * 1000))  # Convert seconds to ms for storage
        seek_help = self.create_help_button("Step size for silence detection. Lower values are more accurate but slower.")
        seek_layout = QHBoxLayout()
        seek_layout.addWidget(self.settings_seek_step)
        seek_layout.addWidget(seek_help)
        segment_layout.addRow("Seek Step:", seek_layout)
        
        # Min time between segments
        self.settings_min_time_between = QSpinBox()
        self.settings_min_time_between.setRange(0, 60)
        self.settings_min_time_between.setSingleStep(1)
        self.settings_min_time_between.setValue(self.settings.get("min_time_between_segments", 10000) // 1000)  # Convert ms to seconds
        self.settings_min_time_between.setSuffix(" sec")
        self.settings_min_time_between.valueChanged.connect(lambda val: self.update_setting("min_time_between_segments", val * 1000))  # Convert seconds to ms for storage
        between_help = self.create_help_button("Minimum time required between detected segments")
        between_layout = QHBoxLayout()
        between_layout.addWidget(self.settings_min_time_between)
        between_layout.addWidget(between_help)
        segment_layout.addRow("Min Time Between Segments:", between_layout)
        
        # Min segment length
        self.settings_min_segment = QSpinBox()
        self.settings_min_segment.setRange(5, 60)
        self.settings_min_segment.setSingleStep(5)
        self.settings_min_segment.setValue(self.settings.get("min_segment_length", 15))
        self.settings_min_segment.setSuffix(" min")
        self.settings_min_segment.valueChanged.connect(lambda val: self.update_setting("min_segment_length", val))
        segment_help = self.create_help_button("Minimum length for a detected segment")
        
        # Create the layout here before using it
        self.segment_min_layout = QHBoxLayout()
        self.segment_min_layout.addWidget(self.settings_min_segment)
        self.segment_min_layout.addWidget(segment_help)
        segment_layout.addRow("Min Segment Length:", self.segment_min_layout)
        
        # Dropout length
        self.settings_dropout = QSpinBox()
        self.settings_dropout.setRange(10, 300)
        self.settings_dropout.setSingleStep(10)
        self.settings_dropout.setValue(self.settings.get("dropout", 60))
        self.settings_dropout.setSuffix(" sec")
        self.settings_dropout.valueChanged.connect(lambda val: self.update_setting("dropout", val))
        dropout_help = self.create_help_button("Ignore detected segments shorter than this duration")
        dropout_layout = QHBoxLayout()
        dropout_layout.addWidget(self.settings_dropout)
        dropout_layout.addWidget(dropout_help)
        segment_layout.addRow("Dropout Length:", dropout_layout)
        
        # Pre segment padding
        self.settings_pre_padding = QSpinBox()
        self.settings_pre_padding.setRange(-180, 180)
        self.settings_pre_padding.setSingleStep(1)
        self.settings_pre_padding.setValue(self.settings.get("pre_segment_padding", 0) // 1000)  # Convert ms to seconds
        self.settings_pre_padding.setSuffix(" sec")
        self.settings_pre_padding.valueChanged.connect(lambda val: self.update_setting("pre_segment_padding", val * 1000))  # Convert seconds to ms for storage
        pre_pad_help = self.create_help_button("Additional time to add before each segment")
        pre_pad_layout = QHBoxLayout()
        pre_pad_layout.addWidget(self.settings_pre_padding)
        pre_pad_layout.addWidget(pre_pad_help)
        segment_layout.addRow("Pre-segment Padding:", pre_pad_layout)
        
        # Post segment padding
        self.settings_post_padding = QSpinBox()
        self.settings_post_padding.setRange(-180, 180)
        self.settings_post_padding.setSingleStep(1)
        self.settings_post_padding.setValue(self.settings.get("post_segment_padding", 0) // 1000)  # Convert ms to seconds
        self.settings_post_padding.setSuffix(" sec")
        self.settings_post_padding.valueChanged.connect(lambda val: self.update_setting("post_segment_padding", val * 1000))  # Convert seconds to ms for storage
        post_pad_help = self.create_help_button("Additional time to add after each segment")
        post_pad_layout = QHBoxLayout()
        post_pad_layout.addWidget(self.settings_post_padding)
        post_pad_layout.addWidget(post_pad_help)
        segment_layout.addRow("Post-segment Padding:", post_pad_layout)
        
        # Save unsegmented checkbox
        self.settings_save_unsegmented = QCheckBox("Save complete track in addition to segments")
        self.settings_save_unsegmented.setChecked(self.settings.get("save_unsegmented", False))
        self.settings_save_unsegmented.stateChanged.connect(lambda state: self.update_setting("save_unsegmented", bool(state)))
        unseg_help = self.create_help_button("Save a complete processed audio file in addition to segments")
        unseg_layout = QHBoxLayout()
        unseg_layout.addWidget(self.settings_save_unsegmented)
        unseg_layout.addWidget(unseg_help)
        segment_layout.addRow("", unseg_layout)
        
        # Trim only checkbox
        self.settings_trim_only = QCheckBox("Trim only (no additional processing)")
        self.settings_trim_only.setChecked(self.settings.get("trim_only", False))
        self.settings_trim_only.stateChanged.connect(lambda state: self.update_setting("trim_only", bool(state)))
        trim_help = self.create_help_button("Only trim audio into segments without applying processing profiles")
        trim_layout = QHBoxLayout()
        trim_layout.addWidget(self.settings_trim_only)
        trim_layout.addWidget(trim_help)
        segment_layout.addRow("", trim_layout)
        
        # Batch normalize checkbox
        self.settings_batch_normalize = QCheckBox("Apply batch normalization across segments and chunks")
        self.settings_batch_normalize.setChecked(self.settings.get("batch_normalize", False))
        self.settings_batch_normalize.stateChanged.connect(lambda state: self.update_setting("batch_normalize", bool(state)))
        batch_help = self.create_help_button("Apply consistent normalization across all segments and chunks for uniform loudness without jumps")
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(self.settings_batch_normalize)
        batch_layout.addWidget(batch_help)
        segment_layout.addRow("", batch_layout)
        
        segment_group.setLayout(segment_layout)
        scroll_layout.addWidget(segment_group)
        
        # --- Processing Settings ---
        processing_group = QGroupBox("Processing Settings")
        processing_layout = QFormLayout()
        
        # Process speed
        self.settings_process_speed = QComboBox()
        self.settings_process_speed.addItems(["Conservative (Stable)", "Balanced", "Full Speed"])
        current_speed = self.settings.get("process_speed", "Full Speed")
        self.settings_process_speed.setCurrentText(current_speed)
        self.settings_process_speed.currentTextChanged.connect(lambda val: self.update_setting("process_speed", val))
        speed_help = self.create_help_button("Processing speed. Conservative mode is more stable but slower.")
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(self.settings_process_speed)
        speed_layout.addWidget(speed_help)
        processing_layout.addRow("Process Speed:", speed_layout)
        
        # Cache size
        self.settings_cache_size = QComboBox()
        self.settings_cache_size.addItems(["512 MB", "1024 MB", "2048 MB", "4096 MB", "8192 MB"])
        current_cache = self.settings.get("cache_size", 2048)
        self.settings_cache_size.setCurrentText(f"{current_cache} MB")
        self.settings_cache_size.currentTextChanged.connect(lambda val: self.update_setting("cache_size", int(val.split()[0])))
        cache_help = self.create_help_button("Audio processing cache size. Larger values use more RAM but improve performance.")
        cache_layout = QHBoxLayout()
        cache_layout.addWidget(self.settings_cache_size)
        cache_layout.addWidget(cache_help)
        processing_layout.addRow("Cache Size:", cache_layout)
        
        # Multi-core processing
        self.settings_multi_core = QCheckBox("Use multi-core processing")
        self.settings_multi_core.setChecked(self.settings.get("multi_core", True))
        self.settings_multi_core.stateChanged.connect(lambda state: self.update_setting("multi_core", bool(state)))
        multi_help = self.create_help_button("Enable multi-core processing for better performance")
        multi_layout = QHBoxLayout()
        multi_layout.addWidget(self.settings_multi_core)
        multi_layout.addWidget(multi_help)
        processing_layout.addRow("", multi_layout)
        
        # Processor limit
        self.settings_processor_limit = QSpinBox()
        self.settings_processor_limit.setRange(1, 32)
        self.settings_processor_limit.setValue(self.settings.get("processor_limit", 4))
        self.settings_processor_limit.valueChanged.connect(lambda val: self.update_setting("processor_limit", val))
        processor_help = self.create_help_button("Maximum number of processor cores to use")
        processor_layout = QHBoxLayout()
        processor_layout.addWidget(self.settings_processor_limit)
        processor_layout.addWidget(processor_help)
        processing_layout.addRow("Processor Limit:", processor_layout)
        
        processing_group.setLayout(processing_layout)
        scroll_layout.addWidget(processing_group)
        
        # --- Output Settings ---
        output_group = QGroupBox("Output Settings")
        output_layout = QFormLayout()
        
        # Output format
        self.settings_output_format = QComboBox()
        self.settings_output_format.addItems(["mp3", "wav", "flac", "ogg"])
        current_format = self.settings.get("output_format", "mp3")
        self.settings_output_format.setCurrentText(current_format)
        self.settings_output_format.currentTextChanged.connect(lambda val: self.update_setting("output_format", val))
        format_help = self.create_help_button("Audio output format")
        format_layout = QHBoxLayout()
        format_layout.addWidget(self.settings_output_format)
        format_layout.addWidget(format_help)
        output_layout.addRow("Output Format:", format_layout)
        
        # MP3 bitrate
        self.settings_mp3_bitrate = QComboBox()
        self.settings_mp3_bitrate.addItems(["64", "96", "128", "160", "192", "256", "320"])
        current_bitrate = str(self.settings.get("mp3_bitrate", 128))
        self.settings_mp3_bitrate.setCurrentText(current_bitrate)
        self.settings_mp3_bitrate.currentTextChanged.connect(lambda val: self.update_setting("mp3_bitrate", int(val)))
        bitrate_help = self.create_help_button("MP3 bitrate in kbps. Higher values have better quality but larger file size.")
        bitrate_layout = QHBoxLayout()
        bitrate_layout.addWidget(self.settings_mp3_bitrate)
        bitrate_layout.addWidget(bitrate_help)
        output_layout.addRow("MP3 Bitrate:", bitrate_layout)
        
        # Show all operations
        self.settings_show_operations = QCheckBox("Show all processing operations in log")
        self.settings_show_operations.setChecked(self.settings.get("show_all_operations", False))
        self.settings_show_operations.stateChanged.connect(lambda state: self.update_setting("show_all_operations", bool(state)))
        ops_help = self.create_help_button("Show detailed operations in log (may be verbose)")
        ops_layout = QHBoxLayout()
        ops_layout.addWidget(self.settings_show_operations)
        ops_layout.addWidget(ops_help)
        output_layout.addRow("", ops_layout)
        
        # Confirm overwrite
        self.settings_confirm_overwrite = QCheckBox("Confirm before overwriting files")
        self.settings_confirm_overwrite.setChecked(self.settings.get("confirm_overwrite", True))
        self.settings_confirm_overwrite.stateChanged.connect(lambda state: self.update_setting("confirm_overwrite", bool(state)))
        overwrite_help = self.create_help_button("Prompt for confirmation before overwriting existing files")
        overwrite_layout = QHBoxLayout()
        overwrite_layout.addWidget(self.settings_confirm_overwrite)
        overwrite_layout.addWidget(overwrite_help)
        output_layout.addRow("", overwrite_layout)
        
        output_group.setLayout(output_layout)
        scroll_layout.addWidget(output_group)
        
        # Add Reset to Defaults button
        reset_btn = QPushButton("Reset to Default Settings")
        reset_btn.clicked.connect(self.reset_settings)
        scroll_layout.addWidget(reset_btn)
        
        # Add Set as Default button
        set_default_btn = QPushButton("Set as Default")
        set_default_btn.setStyleSheet("background-color: #8DFF8D;")
        set_default_btn.clicked.connect(self.save_settings_as_default)
        scroll_layout.addWidget(set_default_btn)
        
        # Add stretcher to push everything to top
        scroll_layout.addStretch(1)
        
        # Set the scroll area content
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        # Add the tab to widget
        tab_widget.addTab(settings_tab, "Settings")

    def create_help_button(self, help_text):
        """Create help button with tooltip"""
        help_btn = QPushButton()
        help_btn.setIcon(QIcon(self.HELP_ICON))
        help_btn.setToolTip(help_text)
        help_btn.setMaximumWidth(24)
        help_btn.setMaximumHeight(24)
        help_btn.setStyleSheet("QPushButton { border: none; }")
        return help_btn
        
    def update_setting(self, key, value):
        """Update a setting value"""
        if self.initializing:
            return
            
        self.settings[key] = value
        
        # Save settings to QSettings
        settings = QSettings()
        settings.beginGroup("Settings")
        settings.setValue(key, value)
        settings.endGroup()
        
        # Log the change
        self.log(f"Updated setting: {key} = {value}", detailed=True)
        
    def reset_settings(self):
        """Reset settings to default values"""
        # Confirm with user
        reply = QMessageBox.question(
            self, 
            "Reset Settings", 
            "Are you sure you want to reset all settings to default values?", 
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Reset to built-in defaults
            self.settings = DEFAULT_SETTINGS.copy()
            
            # Update UI controls
            self.settings_silence_threshold.setValue(self.settings.get("silence_threshold", 17))
            self.settings_min_silence.setValue(self.settings.get("min_silence", 4000) // 1000)
            self.settings_seek_step.setValue(self.settings.get("seek_step", 2000) // 1000)
            self.settings_min_time_between.setValue(self.settings.get("min_time_between_segments", 10000) // 1000)
            self.settings_min_segment.setValue(self.settings.get("min_segment_length", 15))
            self.settings_dropout.setValue(self.settings.get("dropout", 60))
            self.settings_pre_padding.setValue(self.settings.get("pre_segment_padding", 0) // 1000)
            self.settings_post_padding.setValue(self.settings.get("post_segment_padding", 0) // 1000)
            self.settings_save_unsegmented.setChecked(self.settings.get("save_unsegmented", False))
            self.settings_trim_only.setChecked(self.settings.get("trim_only", False))
            self.settings_batch_normalize.setChecked(self.settings.get("batch_normalize", False))
            self.settings_process_speed.setCurrentText(self.settings.get("process_speed", "Full Speed"))
            self.settings_cache_size.setCurrentText(f"{self.settings.get('cache_size', 2048)} MB")
            self.settings_multi_core.setChecked(self.settings.get("multi_core", True))
            self.settings_processor_limit.setValue(self.settings.get("processor_limit", 4))
            self.settings_output_format.setCurrentText(self.settings.get("output_format", "mp3"))
            self.settings_mp3_bitrate.setCurrentText(str(self.settings.get("mp3_bitrate", 128)))
            self.settings_show_operations.setChecked(self.settings.get("show_all_operations", False))
            self.settings_confirm_overwrite.setChecked(self.settings.get("confirm_overwrite", True))
            
            # Save to QSettings
            settings = QSettings()
            settings.beginGroup("Settings")
            for key, value in self.settings.items():
                settings.setValue(key, value)
            settings.endGroup()
            
            self.log("All settings reset to default values")
            
            # Show confirmation
            QMessageBox.information(
                self,
                "Settings Reset",
                "All settings have been reset to default values."
            )

    def save_settings_as_default(self):
        """Save current settings as default in default.json"""
        try:
            default_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "default.json")
            self.log(f"Saving current settings to {default_path}", detailed=True)
            
            # Load existing defaults or create new structure
            defaults = {}
            if os.path.exists(default_path):
                try:
                    with open(default_path, 'r') as f:
                        defaults = json.load(f)
                except Exception as e:
                    self.log(f"Could not read existing default.json, creating new file: {str(e)}", detailed=True)
            
            # Update settings section
            defaults["settings"] = self.settings
            
            # Make sure we preserve profiles if they exist
            if "profiles" not in defaults:
                defaults["profiles"] = self.profiles
            
            # Save back to file
            with open(default_path, 'w') as f:
                json.dump(defaults, f, indent=2)
            
            self.log("Current settings saved as default")
            QMessageBox.information(self, "Settings Saved", "Current settings have been saved as default.")
        except Exception as e:
            self.log(f"Error saving settings as default: {str(e)}")
            QMessageBox.critical(self, "Save Error", f"Failed to save settings as default: {str(e)}")

    def log_from_worker(self, message):
        """Slot to handle logs from the ProcessingWorker and route to the correct log tabs.
        Filters and groups logs for the Standard Log to improve readability and reduce redundancy.
        Technical details and chunk-by-chunk logs are only shown in the Detailed Log.
        """
        timestamp = time.strftime("[%H:%M:%S] ")
        msg = f"{timestamp}{message}"
        upper_msg = message.upper()
        # Always send everything to Detailed Log
        self.detailed_log_buffer.append(msg)
        self.detailed_log_text.append(msg)

        # Channel processing summary cache
        self.channel_summary_cache = getattr(self, 'channel_summary_cache', {})

        # Filtering logic for Standard Log
        show_in_standard = False
        formatted = None

        # Section headers and high-level actions
        if "SELECTED DIRECTORY:" in upper_msg:
            formatted = f"\n=== Directory Selected ===\n{msg}\n"
            show_in_standard = True
        elif "FOUND" in upper_msg and "CHANNEL" in upper_msg:
            show_in_standard = True
        elif "FOUND" in upper_msg and "TRACK" in upper_msg:
            show_in_standard = True
        elif "PROCESSING WITH THE FOLLOWING SEGMENTATION" in upper_msg:
            formatted = f"\n=== Processing Started ===\n{msg}\n"
            show_in_standard = True
        elif "PROCESSING TRACK" in upper_msg and "/" in message:
            show_in_standard = True
        elif "LOADING CHANNEL:" in upper_msg:
            show_in_standard = True
        elif "PERFORMANCE SUMMARY" in upper_msg:
            formatted = f"\n=== Performance Summary ===\n"
            show_in_standard = True
        elif "PROCESSING COMPLETE" in upper_msg or "✅" in message:
            formatted = f"\n✅ Processing complete!\n"
            show_in_standard = True
        elif "ERROR" in upper_msg or "WARNING" in upper_msg:
            show_in_standard = True
        # Suppress technical/verbose lines
        elif any(x in upper_msg for x in ["CHUNK", "PLAN", "CHECKLIST", "NORMALIZATION", "ASSIGNED", "SEGMENT", "CONCATENAT", "BATCH NORM", "PEAK NORMALIZATION", "LUFS NORMALIZATION", "RMS NORMALIZATION", "POST-SEGMENT", "PRE-SEGMENT", "PROFILE:"]):
            show_in_standard = False
        else:
            show_in_standard = False

        # Channel processing summary
        if "CHANNEL" in upper_msg and "COMPLETE" in upper_msg:
            channel_name = message.split(":")[1].strip()
            channel_summary = self.channel_summary_cache.get(channel_name, {})
            channel_summary['chunks'] = channel_summary.get('chunks', 0) + 1
            if "NORMALIZATION" in upper_msg:
                channel_summary['normalization'] = True
            if "COMPRESSOR" in upper_msg:
                channel_summary['compressor'] = True
            self.channel_summary_cache[channel_name] = channel_summary
            # Flush channel summary to Standard Log
            summary_msg = f"Channel '{channel_name}' processed: {channel_summary['chunks']} chunks, "
            if channel_summary.get('normalization'):
                summary_msg += "normalized, "
            if channel_summary.get('compressor'):
                summary_msg += "compressed"
            summary_msg = summary_msg.strip() + "."
            self.standard_log_buffer.append(summary_msg)
            self.standard_log_text.append(summary_msg)
            # Clear channel summary cache
            del self.channel_summary_cache[channel_name]

        # Segment detection summary
        if "SEGMENT DETECTION COMPLETE" in upper_msg:
            segment_summary = message.split(":")[1].strip()
            self.standard_log_buffer.append(segment_summary)
            self.standard_log_text.append(segment_summary)

        if show_in_standard:
            entry = formatted if formatted else msg
            # Prevent redundant consecutive entries
            if not self.standard_log_buffer or self.standard_log_buffer[-1] != entry:
                self.standard_log_buffer.append(entry)
                self.standard_log_text.append(entry)

    def update_selection(self):
        """Update selected files when selection changes"""
        # Get all selected rows
        selected_rows = set(index.row() for index in self.files_table.selectedIndexes())
        
        # Clear previous selection
        self.selected_files = []
        
        # Add selected files
        for row in selected_rows:
            # Get the custom widget in column 0
            track_widget = self.files_table.cellWidget(row, 0)
            if track_widget and isinstance(track_widget, ExpandableTrackWidget):
                self.selected_files.append(track_widget.track_path)
        
        # Process button should always be enabled regardless of selection
        # Update track info panel with the first selected track
        if self.selected_files:
            self.track_info_panel.update_panel(self.selected_files[0])
        else:
            self.track_info_panel.clear_panel()

    def browse_directory(self):
        """Open a file dialog to select a directory containing audio files"""
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        
        # Try to get the default folder from settings
        default_folder = ""
        
        # First check if we have a working directory that exists
        if self.working_dir and os.path.exists(self.working_dir):
            default_folder = self.working_dir
            self.log(f"Using current working directory: {default_folder}", detailed=True)
        else:
            # Try to load from default.json directly
            try:
                default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "default.json")
                if os.path.exists(default_path):
                    with open(default_path, 'r') as f:
                        defaults = json.load(f)
                        if "settings" in defaults and "default_folder" in defaults["settings"]:
                            folder = defaults["settings"]["default_folder"]
                            if os.path.exists(folder):
                                default_folder = folder
                                self.log(f"Loaded default folder from default.json: {default_folder}", detailed=True)
            except Exception as e:
                self.log(f"Error reading default folder from default.json: {str(e)}", detailed=True)
        
        # Log the directory we're starting with
        self.log(f"Opening file dialog with initial directory: {default_folder or 'default location'}")
        
        # Set the directory explicitly using an absolute path
        if default_folder:
            dialog.setDirectory(os.path.abspath(default_folder))
        
        # Show dialog and wait for selection
        if dialog.exec():
            selected_dirs = dialog.selectedFiles()
            if selected_dirs:
                selected_dir = selected_dirs[0]
                self.working_dir = selected_dir
                
                # Save to QSettings for persistence
                settings = QSettings()
                settings.setValue("last_opened_folder", selected_dir)
                
                # Update UI
                self.dir_path_edit.setText(selected_dir)
                self.log(f"Selected directory: {selected_dir}")
                
                # Scan the selected directory
                self.scan_directory(selected_dir)
                
                # Update status bar
                self.statusBar().showMessage(f"Directory selected: {selected_dir}")
    
    def set_default_browse_folder(self):
        """Set the current folder as the default folder for future sessions"""
        if not self.working_dir or not os.path.exists(self.working_dir):
            QMessageBox.warning(self, "No Directory", "Please select a directory first.")
            return
        
        # Save to QSettings for persistence - using last_opened_folder key for startup loading
        settings = QSettings()
        settings.setValue("last_opened_folder", self.working_dir)
        settings.setValue("default_folder", self.working_dir)  # Keep for backward compatibility
        
        # Also save to default.json for persistence across installations
        try:
            # Get the absolute path to default.json in the application directory
            app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_path = os.path.join(app_dir, "default.json")
            
            self.log(f"Saving default folder to {default_path}", detailed=True)
            
            # Create default.json if it doesn't exist
            if not os.path.exists(default_path):
                with open(default_path, 'w') as f:
                    json.dump({"settings": {"default_folder": self.working_dir}}, f, indent=2)
                self.log(f"Created default.json with default folder: {self.working_dir}", detailed=True)
            else:
                # Load existing file
                try:
                    with open(default_path, 'r') as f:
                        defaults = json.load(f)
                
                    # Add default_folder to settings section
                    if "settings" not in defaults:
                        defaults["settings"] = {}
                        
                    # Update both keys for consistency
                    defaults["settings"]["default_folder"] = self.working_dir
                    defaults["settings"]["last_opened_folder"] = self.working_dir
                    
                    # Write updated defaults back to file
                    with open(default_path, 'w') as f:
                        json.dump(defaults, f, indent=2)
                    
                    self.log(f"Default folder saved to default.json: {self.working_dir}", detailed=True)
                except Exception as e:
                    self.log(f"Error reading/writing default.json: {str(e)}", detailed=True)
        except Exception as e:
            self.log(f"Error saving default folder to default.json: {str(e)}", detailed=True)
        
        # Inform the user
        QMessageBox.information(self, "Default Folder Set", 
                             f"Successfully set {self.working_dir} as the default folder.")
        self.log(f"Default folder set: {self.working_dir}")
        
        # Update status bar
        self.statusBar().showMessage(f"Default folder set: {self.working_dir}")

    def show_files_table_context_menu(self, position):
        """Show context menu for track files table"""
        clicked_index = self.files_table.indexAt(position)
        if not clicked_index.isValid():
            return
        
        # Only determine clicked row before showing context menu
        row = clicked_index.row()
        track_widget = self.files_table.cellWidget(row, 0)
        clicked_track_path = None
        if track_widget and isinstance(track_widget, ExpandableTrackWidget):
            clicked_track_path = track_widget.track_path
            
        # Show the pre-created context menu immediately
        action = self.files_table_context_menu.exec(self.files_table.viewport().mapToGlobal(position))
        if not action:
            return
            
        if action.text() == "Process Selected Files":
            if clicked_track_path:
                self.log(f"Processing single track: {os.path.basename(clicked_track_path)}")
                # Create a temporary list with only the clicked track
                original_selected_files = self.selected_files.copy()
                self.selected_files = [clicked_track_path]
                self.start_processing()
                # Restore original selection after processing is initiated
                self.selected_files = original_selected_files
            else:
                self.log("Process Selected Files selected")
                self.start_processing()
        elif action.text() == "Reprocess All Channels":
            self.log("Reprocess All Channels selected")
            self.reprocess_all_channels()
        elif action.text() == "View Track Info":
            # Use the item data if available for faster lookup
            item = self.files_table.item(row, 0)
            track_path = None
            if item is not None:
                track_path = item.data(Qt.ItemDataRole.UserRole)
            if not track_path:
                # Fallback to widget
                track_widget = self.files_table.cellWidget(row, 0)
                if isinstance(track_widget, ExpandableTrackWidget):
                    track_path = track_widget.track_path
            if track_path:
                self.log(f"View Track Info selected for track: {os.path.basename(track_path)}")
                self.track_info_panel.update_panel(track_path)
        elif action.text() == "Open Folder":
            # Get track path from the widget
            track_widget = self.files_table.cellWidget(row, 0)
            if isinstance(track_widget, ExpandableTrackWidget):
                track_path = track_widget.track_path
                if track_path and os.path.exists(track_path):
                    self.log(f"Opening folder for track: {os.path.basename(track_path)}")
                    try:
                        # Use platform-specific commands to open file explorer at the track location
                        if platform.system() == 'Windows':
                            os.startfile(track_path)
                        elif platform.system() == 'Darwin':  # macOS
                            subprocess.call(['open', track_path])
                        else:  # Linux and other Unix-like systems
                            subprocess.call(['xdg-open', track_path])
                    except Exception as e:
                        self.log(f"Error opening folder: {str(e)}")
                        QMessageBox.warning(self, "Error", f"Could not open the folder: {str(e)}")
        elif action.text() == "Trim Segment":
            widget = self.files_table.cellWidget(row, 0)
            if isinstance(widget, ExpandableTrackWidget):
                self.log(f"Trim Segment selected for track: {os.path.basename(widget.track_path)}")
                self.show_segment_selection_dialog("trim", widget.track_path)
        elif action.text() == "Apply Fade to Segment":
            widget = self.files_table.cellWidget(row, 0)
            if isinstance(widget, ExpandableTrackWidget):
                self.log(f"Apply Fade to Segment selected for track: {os.path.basename(widget.track_path)}")
                self.show_segment_selection_dialog("fade", widget.track_path)
        elif action.text() == "Export Segment":
            widget = self.files_table.cellWidget(row, 0)
            if isinstance(widget, ExpandableTrackWidget):
                self.log(f"Export Segment selected for track: {os.path.basename(widget.track_path)}")
                self.show_segment_selection_dialog("export", widget.track_path)
        elif action.text() == "Select All":
            self.log("Select All selected")
            self.files_table.selectAll()
        elif action.text() == "Select None":
            self.log("Select None selected")
            self.files_table.clearSelection()
            
    def on_profile_selected(self, current, previous):
        """Handler for profile selection change"""
        if self.updating_profile:
            return
        
        if current:
            profile_name = current.text()
            self.load_profile_to_editor(profile_name)
        else:
            self.clear_profile_editor()
    
    def load_profile_to_editor(self, profile_name):
        """Load the selected profile into the editor"""
        if profile_name not in self.profiles:
            self.log(f"Profile {profile_name} not found")
            return
        
        self.updating_profile = True
        self.current_profile_name = profile_name
        profile = self.profiles[profile_name]
        
        # Set basic values
        self.profile_name_edit.setText(profile_name)
        self.profile_description.setText(profile.get('description', ''))
        
        # Set gain
        gain = profile.get('gain', 0)
        self.gain_input.setValue(gain)
        
        # Set normalization
        normalize = profile.get('normalize', {})
        if isinstance(normalize, dict):
            self.norm_group.setChecked(normalize.get('enabled', True))
            
            method = normalize.get('method', 'peak').lower()
            method_index = 0  # Default to peak
            if method == 'rms':
                method_index = 1
            elif method == 'lufs':
                method_index = 2
            self.norm_method.setCurrentIndex(method_index)
            
            self.norm_target.setValue(normalize.get('target_level', -3.0))
        else:
            # Handle legacy format where normalize was a boolean
            self.norm_group.setChecked(bool(normalize))
            
        # Set dynamics
        dynamic_config = profile.get('dynamic_processing', {})
        self.dynamics_group.setChecked(dynamic_config.get('enabled', False))
        
        # Set compressor values
        comp_config = dynamic_config.get('compressor', {}) if isinstance(dynamic_config, dict) else {}
        self.comp_threshold.setValue(comp_config.get('threshold', -18.0))
        self.comp_ratio.setValue(comp_config.get('ratio', 2.5))
        self.comp_attack.setValue(comp_config.get('attack', 20.0))
        self.comp_release.setValue(comp_config.get('release', 250.0))
        
        # Set limiter values - first check for the new standalone limiter structure
        if 'limiter' in profile:
            limiter_config = profile.get('limiter', {})
            self.limiter_group.setChecked(limiter_config.get('enabled', False))
            self.limiter_threshold.setValue(limiter_config.get('threshold', -1.0))
            self.limiter_release.setValue(limiter_config.get('release', 50.0))
        else:
            # Legacy format - check if limiter was inside dynamic_processing
            limiter_config = dynamic_config.get('limiter', {}) if isinstance(dynamic_config, dict) else {}
            self.limiter_group.setChecked(limiter_config.get('enabled', False))
            self.limiter_threshold.setValue(limiter_config.get('threshold', -1.0))
            self.limiter_release.setValue(limiter_config.get('release', 50.0))
        
        # Set EQ values - retrieve eq_config before using it
        self.eq_group.setChecked(profile.get('use_eq', False))
        eq_config = profile.get('eq', {})
        self.hp_filter.setValue(eq_config.get('high_pass', 100))
        self.lp_filter.setValue(eq_config.get('low_pass', 10000))
        
        self.updating_profile = False
    
    def clear_profile_editor(self):
        """Clear the profile editor"""
        self.updating_profile = True
        self.current_profile_name = None
        
        # Clear fields
        self.profile_name_edit.clear()
        self.profile_description.clear()
        self.gain_input.setValue(0.0)
        self.norm_group.setChecked(True)
        self.norm_method.setCurrentIndex(0)
        self.norm_target.setValue(-3.0)
        self.dynamics_group.setChecked(False)
        self.comp_threshold.setValue(-18.0)
        self.comp_ratio.setValue(2.5)
        self.comp_attack.setValue(20.0)
        self.comp_release.setValue(250.0)
        self.limiter_group.setChecked(False)
        self.limiter_threshold.setValue(-1.0)
        self.limiter_release.setValue(50.0)
        self.eq_group.setChecked(False)
        self.hp_filter.setValue(100)
        self.lp_filter.setValue(10000)
        
        self.updating_profile = False
    
    def on_gain_changed(self, value):
        """Update gain label when input changes"""
        if not self.updating_profile:
            self.on_profile_changed()
    
    def on_profile_name_changed(self, text):
        """Handle profile name changes"""
        if self.updating_profile:
            return
        
        # Don't allow empty names
        if not text.strip():
            return
            
        # Don't do anything if name hasn't changed
        if text == self.current_profile_name:
            return
        
        # Start the timer for delayed processing
        self.pending_profile_name = text
        self.profile_name_timer.start(500)  # Delay of 500ms
    
    def apply_profile_name_change(self):
        """Apply the profile name change after the delay"""
        text = self.pending_profile_name
        if not text:
            return
            
        self.pending_profile_name = None
        
        # Check if the name already exists in another profile
        if text in self.profiles and text != self.current_profile_name:
            # Skip if we're trying to create a duplicate
            self.log(f"Profile name '{text}' already exists", detailed=True)
            # Revert to original name
            self.updating_profile = True
            self.profile_name_edit.setText(self.current_profile_name)
            self.updating_profile = False
            return
        
        # Remember the old profile name
        old_name = self.current_profile_name
        
        if old_name and old_name in self.profiles:
            # Create a copy of the profile data
            profile_data = self.profiles[old_name].copy()
            
            # Remove the old profile
            del self.profiles[old_name]
            
            # Add the profile with the new name
            self.profiles[text] = profile_data
            
            # Update the current profile name
            self.current_profile_name = text
            
            # Update the selected item in the profiles list
            for i in range(self.profiles_list.count()):
                if self.profiles_list.item(i).text() == old_name:
                    self.profiles_list.item(i).setText(text)
                    break
            
            # Update track profile assignments that use this profile
            for channel, profile in self.track_profile_assignments.items():
                if profile == old_name:
                    self.track_profile_assignments[channel] = text
            
            # Also update the channel assignment dropdown selections in the UI
            for row in range(self.track_assignments_table.rowCount()):
                combo = self.track_assignments_table.cellWidget(row, 1)
                if combo:
                    current_text = combo.currentText()
                    if current_text == old_name:
                        idx = combo.findText(text)
                        if idx >= 0:
                            combo.setCurrentIndex(idx)
                    # Update the item in the dropdown list
                    for i in range(combo.count()):
                        if combo.itemText(i) == old_name:
                            combo.setItemText(i, text)
                            break
            
            # Log the change
            self.log(f"Renamed profile: {old_name} -> {text}")
            
            # Mark that the profile is changed
            self.on_profile_changed()
        else:
            # Handle case where old_name is None or not in profiles
            self.current_profile_name = text
    
    def on_profile_changed(self):
        """Apply changes immediately and mark profile as changed"""
        if self.updating_profile or not self.current_profile_name:
            return
        
        # Get current profile data from UI
        profile = {}
        profile['description'] = self.profile_description.text()
        profile['gain'] = self.gain_input.value()
        
        # Normalization - ensure the enabled state is properly saved
        profile['normalize'] = {
            'enabled': self.norm_group.isChecked(),
            'method': ['peak', 'rms', 'lufs'][self.norm_method.currentIndex()],
            'target_level': self.norm_target.value()
        }
        
        # Dynamics
        profile['dynamic_processing'] = {
            'enabled': self.dynamics_group.isChecked(),  # Now correctly saves the enabled state
            'compressor': {
                'threshold': self.comp_threshold.value(),
                'ratio': self.comp_ratio.value(),
                'attack': self.comp_attack.value(),
                'release': self.comp_release.value()
            },
            'limiter': {
                'enabled': self.limiter_group.isChecked(),
                'threshold': self.limiter_threshold.value(),
                'release': self.limiter_release.value()
            }
        }
        
        # EQ
        profile['use_eq'] = self.eq_group.isChecked()
        profile['eq'] = {
            'high_pass': self.hp_filter.value(),
            'low_pass': self.lp_filter.value()
        }
        
        # Apply changes to profile immediately
        self.profiles[self.current_profile_name] = profile
        
        # Visual indicator that profiles have been automatically applied but can be saved to default
        self.save_btn.setStyleSheet("background-color: #FFD8B1;")
        self.save_btn.setText("Set Current Profile as Default")
        
        self.log(f"Profile '{self.current_profile_name}' updated", detailed=True)

    def add_new_profile(self):
        """Add a new profile to the list"""
        # Ask for profile name
        name, ok = QInputDialog.getText(
            self,
            "New Profile",
            "Enter profile name:",
        )
        
        if ok and name:
            # Check if name exists
            if name in self.profiles:
                QMessageBox.warning(
                    self,
                    "Profile Exists",
                    f"A profile with the name '{name}' already exists."
                )
                return
                
            # Create new profile
            self.profiles[name] = self.create_default_profile()
            
            # Add to list and select it
            self.updating_profile = True
            item = QListWidgetItem(name)
            self.profiles_list.addItem(item)
            self.profiles_list.setCurrentItem(item)
            self.updating_profile = False
            
            # Load into editor
            self.load_profile_to_editor(name)
            self.log(f"Created new profile: {name}")
            
            # Save profiles
            self.save_profiles()
    
    def copy_selected_profile(self):
        """Copy the selected profile"""
        if not self.current_profile_name:
            QMessageBox.warning(
                self,
                "No Profile Selected",
                "Please select a profile to copy."
            )
            return
            
        # Ask for new profile name
        name, ok = QInputDialog.getText(
            self,
            "Copy Profile",
            f"Enter name for copy of '{self.current_profile_name}':",
            text=f"{self.current_profile_name} (Copy)"
        )
        
        if ok and name:
            # Check if name exists
            if name in self.profiles:
                QMessageBox.warning(
                    self,
                    "Profile Exists",
                    f"A profile with the name '{name}' already exists."
                )
                return
                
            # Create copy of profile
            self.profiles[name] = self.profiles[self.current_profile_name].copy()
            
            # Add to list and select it
            self.updating_profile = True
            item = QListWidgetItem(name)
            self.profiles_list.addItem(item)
            self.profiles_list.setCurrentItem(item)
            self.updating_profile = False
            
            # Load into editor
            self.load_profile_to_editor(name)
            self.log(f"Copied profile '{self.current_profile_name}' to '{name}'")
            
            # Save profiles
            self.save_profiles()
    
    def delete_selected_profile(self):
        """Delete the selected profile"""
        if not self.current_profile_name:
            QMessageBox.warning(
                self,
                "No Profile Selected",
                "Please select a profile to delete."
            )
            return
            
        # Don't allow deletion of special profiles
        if self.current_profile_name in ["Kirtan (Vocals)", "Tabla", "Sangat (Harmonium)", "Sangat (Tamboura)", "Do Not Process"]:
            QMessageBox.warning(
                self,
                "Cannot Delete",
                f"The profile '{self.current_profile_name}' is a system profile and cannot be deleted."
            )
            return
            
        # Confirm deletion
        reply = QMessageBox.question(
            self, 
            "Delete Profile", 
            f"Are you sure you want to delete profile '{self.current_profile_name}'?", 
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Delete from profiles
            del self.profiles[self.current_profile_name]
            
            # Remove from list
            for i in range(self.profiles_list.count()):
                if self.profiles_list.item(i).text() == self.current_profile_name:
                    self.profiles_list.takeItem(i)
                    break
            
            # Clear editor
            self.clear_profile_editor()
            self.log(f"Deleted profile: {self.current_profile_name}")
            
            # Save profiles
            self.save_profiles()
    
    def import_profiles(self):
        """Import profiles from a JSON file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Profiles",
            "",
            "JSON Files (*.json)"
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            if not isinstance(data, dict):
                raise ValueError("Invalid file format")
                
            # If the file contains a 'profiles' key, use that
            import_profiles = data.get('profiles', data)
                
            if not import_profiles or not isinstance(import_profiles, dict):
                raise ValueError("No valid profiles found in file")
                
            # Confirm import
            reply = QMessageBox.question(
                self, 
                "Import Profiles", 
                f"Import {len(import_profiles)} profiles? Existing profiles with the same names will be overwritten.", 
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                QMessageBox.StandardButton.Yes
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Update profiles
                for name, profile in import_profiles.items():
                    self.profiles[name] = profile
                    
                # Refresh list
                self.load_profiles_to_ui()
                self.log(f"Imported {len(import_profiles)} profiles from {os.path.basename(file_path)}")
                
                # Save to settings file
                self.save_profiles()
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to import profiles: {str(e)}"
            )
    
    def export_profiles(self):
        """Export profiles to a JSON file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Profiles",
            "kirtan_processor_profiles.json",
            "JSON Files (*.json)"
        )
        
        if not file_path:
            return
            
        try:
            # Ensure file has .json extension
            if not file_path.lower().endswith('.json'):
                file_path += '.json'
                
            # Export data structure
            export_data = {
                'profiles': self.profiles,
                'export_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            self.log(f"Exported {len(self.profiles)} profiles to {os.path.basename(file_path)}")
            
            # Show success
            QMessageBox.information(
                self,
                "Export Successful",
                f"Successfully exported {len(self.profiles)} profiles to {os.path.basename(file_path)}"
            )
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export profiles: {str(e)}"
            )
    
    def save_current_profile(self):
        """Save the current profile settings"""
        if not self.current_profile_name:
            self.log("No profile selected to save")
            return
        
        # Get profile data from UI
        profile = {}
        profile['description'] = self.profile_description.text()
        profile['gain'] = self.gain_input.value()
        
        # Normalization - ensure the enabled state is properly saved
        profile['normalize'] = {
            'enabled': self.norm_group.isChecked(),
            'method': ['peak', 'rms', 'lufs'][self.norm_method.currentIndex()],
            'target_level': self.norm_target.value()
        }
        
        # Dynamics
        profile['dynamic_processing'] = {
            'enabled': False,
            'compressor': {
                'threshold': self.comp_threshold.value(),
                'ratio': self.comp_ratio.value(),
                'attack': self.comp_attack.value(),
                'release': self.comp_release.value()
            },
            'limiter': {
                'enabled': self.limiter_group.isChecked(),
                'threshold': self.limiter_threshold.value(),
                'release': self.limiter_release.value()
            }
        }
        
        # EQ
        profile['use_eq'] = self.eq_group.isChecked()
        profile['eq'] = {
            'high_pass': self.hp_filter.value(),
            'low_pass': self.lp_filter.value()
        }
        
        # Save to profiles
        self.profiles[self.current_profile_name] = profile
        
        # Reset the save button
        self.save_btn.setStyleSheet("background-color: #8DFF8D;")
        self.save_btn.setText("Set Current Profile as Default")
        
        # Save to settings file
        self.save_profiles()
        self.log(f"Saved profile: {self.current_profile_name}")
    
    def save_all_profiles(self):
        """Save all profiles as default"""
        try:
            default_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "default.json")
            self.log(f"Saving all profiles to {default_path}", detailed=True)

            # Load existing defaults or create new structure
            defaults = {}
            if os.path.exists(default_path):
                try:
                    with open(default_path, 'r') as f:
                        defaults = json.load(f)
                except Exception as e:
                    self.log(f"Could not read existing default.json, creating new file: {str(e)}", detailed=True)

            # Update profiles section
            defaults["profiles"] = self.profiles

            # Save back to file
            with open(default_path, 'w') as f:
                json.dump(defaults, f, indent=2)

            self.log("All profiles saved as default")
            QMessageBox.information(self, "Profiles Saved", "All profiles have been saved as default.")
        except Exception as e:
            self.log(f"Error saving all profiles as default: {str(e)}")
            QMessageBox.critical(self, "Save Error", f"Failed to save all profiles as default: {str(e)}")

    def reset_current_profile(self):
        """Reset the current profile to default values"""
        if not self.current_profile_name:
            return
            
        # Confirm with user
        reply = QMessageBox.question(
            self, 
            "Reset Profile", 
            f"Reset profile '{self.current_profile_name}' to default values?", 
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if self.current_profile_name in self.profiles:
                # Load the default profile if available or create a new one
                try:
                    default_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "default.json")
                    if os.path.exists(default_path):
                        with open(default_path, 'r') as f:
                            defaults = json.load(f)
                            if "profiles" in defaults and self.current_profile_name in defaults["profiles"]:
                                self.profiles[self.current_profile_name] = defaults["profiles"][self.current_profile_name].copy()
                            else:
                                self.profiles[self.current_profile_name] = self.create_default_profile()
                    else:
                        self.profiles[self.current_profile_name] = self.create_default_profile()
                except:
                    self.profiles[self.current_profile_name] = self.create_default_profile()
                    
                # Load the profile into the editor
                self.load_profile_to_editor(self.current_profile_name)
                self.log(f"Reset profile '{self.current_profile_name}' to defaults")
    
    def create_default_profile(self):
        """Create a default profile template"""
        return {
            'description': "User created profile",
            'gain': 0.0,
            'normalize': {
                'enabled': True,
                'method': 'peak',
                'target_level': -3.0
            },
            'dynamic_processing': {
                'enabled': False,
                'compressor': {
                    'threshold': -18.0,
                    'ratio': 2.5,
                    'attack': 20.0,
                    'release': 250.0
                },
                'limiter': {
                    'enabled': False,
                    'threshold': -1.0,
                    'release': 50.0
                }
            },
            'use_eq': False,
            'eq': {
                'high_pass': 100,
                'low_pass': 10000
            }
        }
        
    def load_profiles_to_ui(self):
        """Load profiles into the UI"""
        self.updating_profile = True
        
        # Clear the list
        if hasattr(self, 'profiles_list'):
            self.profiles_list.clear()
            
            # Sort profiles in the same order as the pull-down menu:
            # First Kirtan, then Tabla, then Sangat, then other profiles
            profiles_order = []
            
            # First add "Kirtan" profiles
            if "Kirtan (Vocals)" in self.profiles:
                profiles_order.append("Kirtan (Vocals)")
            kirtan_profiles = [name for name in self.profiles.keys() if name.startswith("Kirtan") and name != "Kirtan (Vocals)"]
            profiles_order.extend(sorted(kirtan_profiles))
            
            # Then add "Tabla" profiles
            if "Tabla" in self.profiles:
                profiles_order.append("Tabla")
            tabla_profiles = [name for name in self.profiles.keys() if name.startswith("Tabla") and name != "Tabla"]
            profiles_order.extend(sorted(tabla_profiles))
            
            # Then add "Sangat" profiles
            sangat_profiles = [name for name in self.profiles.keys() if name.startswith("Sangat")]
            profiles_order.extend(sorted(sangat_profiles))
            
            # Finally add remaining profiles (excluding already added ones and Do Not Process)
            added_profiles = set(profiles_order)
            added_profiles.add("Do Not Process")  # Don't add it twice if it's already there
            remaining_profiles = [name for name in self.profiles.keys() 
                                if name not in added_profiles and name != "Do Not Process"]
            profiles_order.extend(sorted(remaining_profiles))
            
            # Add all profiles except "Do Not Process"
            for name in profiles_order:
                if name != "Do Not Process":  # Skip the internal Do Not Process profile
                    self.profiles_list.addItem(name)
                
            self.updating_profile = False
            
            # Select first profile if available
            if self.profiles_list.count() > 0:
                self.profiles_list.setCurrentRow(0)
    
    def save_profiles(self):
        """Save profiles to settings"""
        """Process only the selected files (triggered from context menu)"""
        self.start_processing()
        
    def start_processing(self):
        """Start processing the selected audio files"""
        # Add debug information to identify the caller
        import traceback
        stack = traceback.extract_stack()
        caller_info = stack[-2]  # Get the caller's information
        self.log(f"DEBUG: start_processing called from {caller_info.name} at {caller_info.filename}:{caller_info.lineno}", detailed=True)
        self.log(f"DEBUG: Caller object: {self.sender()}", detailed=True)

        if self.is_processing:
            self.log("DEBUG: Already processing, ignoring call", detailed=True)
            return
            
        # Always use all audio files regardless of which button triggered processing
        files_to_process = self.audio_files
        self.log("Processing all available tracks")
        
        if not files_to_process:
            self.log("No files available for processing")
            QMessageBox.information(self, "No Files", "No audio files found to process. Please scan a directory with audio files first.")
            return
            
        # Update UI state
        self.is_processing = True
        self.process_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        
        # Change button color to pastel green to indicate processing is occurring
        self.process_button.setStyleSheet("QPushButton { background-color: #C1FFC1; }")  # Pastel green
        
        # Start blinking effect for the button
        self.start_button_blinking()
        
        # Record start time for completion popup
        self.processing_start_time = time.time()
        
        # Log processing start
        if len(files_to_process) == 1:
            self.log(f"Processing 1 track")
        else:
            self.log(f"Processing {len(files_to_process)} tracks")
        
        # Create processing worker - send 'self' as the app parameter
        self.processing_worker = ProcessingWorker(self)
        self.processing_worker.selected_files = files_to_process  # Add the files to process
        
        # Connect signals
        self.processing_worker.progress_update.connect(self.log_from_worker)
        self.processing_worker.progress_bar.connect(self.update_progress)
        self.processing_worker.processing_finished.connect(self.on_processing_finished)
        
        # Start processing
        self.processing_worker.start()

    def start_button_blinking(self):
        """Start blinking effect for the Process All Track Files button."""
        self.blinking = True
        self.blink_state = True

        def toggle_blink():
            if not self.blinking:
                self.process_button.setStyleSheet("QPushButton { background-color: #C1FFC1; }")  # Reset to pastel green
                return

            if self.blink_state:
                self.process_button.setStyleSheet("QPushButton { background-color: #FFD700; }")  # Yellow
            else:
                self.process_button.setStyleSheet("QPushButton { background-color: #C1FFC1; }")  # Pastel green

            self.blink_state = not self.blink_state
            QTimer.singleShot(500, toggle_blink)  # Toggle every 500ms

        toggle_blink()

    def stop_button_blinking(self):
        """Stop blinking effect for the Process All Track Files button."""
        self.blinking = False
        self.process_button.setStyleSheet("QPushButton { background-color: #8DFF8D; }")  # Reset to default green

    def stop_processing(self):
        """Stop the current processing operation"""
        if not self.is_processing or not self.processing_worker:
            return
            
        self.log("Stopping processing...")
        
        # Stop blinking effect
        self.stop_button_blinking()
        
        # Set the stop flag on the worker thread
        if hasattr(self.processing_worker, 'stop_requested'):
            self.processing_worker.stop_requested = True
            
        # Disable stop button to prevent multiple clicks
        self.stop_button.setEnabled(False)
        
        # Change button color to indicate stopping
        self.process_button.setStyleSheet("QPushButton { background-color: #FFB3BA; }")  # Light red

    def on_processing_finished(self, success):
        """Handle completion of processing"""
        self.is_processing = False
        self.process_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        # Stop blinking effect
        self.stop_button_blinking()
        
        # Change button color back to regular green when processing is complete
        self.process_button.setStyleSheet("QPushButton { background-color: #8DFF8D; }")
        
        # Calculate processing time
        processing_time = 0
        if hasattr(self, 'processing_start_time'):
            processing_time = time.time() - self.processing_start_time
        
        if success:
            QMessageBox.information(self, "Processing Complete", f"Processing completed successfully in {processing_time:.2f} seconds.")
        else:
            QMessageBox.warning(self, "Processing Failed", "Processing did not complete successfully.")
        
        # Clear the processing worker reference
        self.processing_worker = None
    
    def update_resource_usage(self):
        """Update resource usage indicators"""
        try:
            if not hasattr(self, 'perf_tracker'):
                self.perf_tracker = PerformanceTracker(self.log)
            
            # Get CPU and memory usage - using a non-blocking approach
            try:
                cpu_usage = self.perf_tracker.get_cpu_percent()
                memory_usage = self.perf_tracker.get_memory_percent()
                
                # Update bars, but only if widgets still exist
                if hasattr(self, 'cpu_usage_bar') and self.cpu_usage_bar is not None:
                    self.cpu_usage_bar.setValue(int(cpu_usage))
                if hasattr(self, 'memory_usage_bar') and self.memory_usage_bar is not None:
                    self.memory_usage_bar.setValue(int(memory_usage))
            except KeyboardInterrupt:
                # Silently handle keyboard interrupts
                pass
            except Exception as e:
                # Handle other exceptions without crashing
                self.log(f"Resource monitor: {str(e)}", detailed=True)
        except Exception as e:
            # Catch-all to prevent any resource monitoring issue from crashing the app
            pass

    def scan_directory(self, directory):
        """Scan a directory for audio tracks and files"""
        if not directory or not os.path.isdir(directory):
            self.log("Invalid directory")
            return
        
        self.log(f"Scanning directory: {directory}")
        
        # Update UI immediately to show we're processing
        self.statusBar().showMessage("Scanning directory...")
        QApplication.processEvents()  # Process UI events to update the status bar
        
        # Clear previous files
        self.audio_files = []
        self.files_table.clearContents()
        self.files_table.setRowCount(0)
        
        # Use a more efficient approach for directory scanning
        try:
            # Disable table updates during population to avoid lag
            self.files_table.setUpdatesEnabled(False)
            
            # Create a progress counter to allow UI updates during long scans
            progress_counter = 0
            
            # Use a single pass to collect all directories with .wav files
            track_dirs = []

            # Check if current directory directly contains WAV files
            has_wav_in_current = False
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.lower().endswith('.wav'):
                        has_wav_in_current = True
                        break
            
            # If the selected directory itself contains WAV files, treat it as a track directory
            if has_wav_in_current:
                self.log(f"Detected WAV files directly in selected folder. Treating as a track directory.")
                track_dirs.append(directory)
            
            # Use os.scandir which is more efficient than listdir + os.path calls
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_dir():
                        # Check if directory contains wav files using scandir
                        has_wav = False
                        with os.scandir(entry.path) as subentries:
                            for subentry in subentries:
                                if subentry.is_file() and subentry.name.lower().endswith('.wav'):
                                    has_wav = True
                                    break
                        
                        if has_wav:
                            track_dirs.append(entry.path)
                    
                    # Update UI periodically during scan for responsiveness
                    progress_counter += 1
                    if progress_counter % 10 == 0:
                        QApplication.processEvents()
            
            # Create all table rows at once
            self.files_table.setRowCount(len(track_dirs))
            
            # Process in batches to ensure UI responsiveness
            batch_size = 10
            for i in range(0, len(track_dirs), batch_size):
                batch = track_dirs[i:i + batch_size]
                
                for j, track_dir in enumerate(batch):
                    row_index = i + j
                    
                    # Get track name from directory name (more efficient)
                    track_name = os.path.basename(track_dir)
                    
                    # Count files rather than loading them all into memory
                    wav_count = 0
                    mp3_count = 0
                    with os.scandir(track_dir) as entries:
                        for entry in entries:
                            if entry.is_file():
                                if entry.name.lower().endswith('.wav'):
                                    wav_count += 1
                                elif entry.name.lower().endswith('.mp3'):
                                    mp3_count += 1
                    
                    inputs_text = f"{wav_count} channels"
                    status_text = f"Processed ({mp3_count} files)" if mp3_count > 0 else "Not processed"
                    
                    # Create expandable widget for this track
                    track_widget = ExpandableTrackWidget(track_dir, track_name)
                    
                    # Add regular cells for columns 1 and 2
                    inputs_item = QTableWidgetItem(inputs_text)
                    inputs_item.setFlags(inputs_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # Make read-only
                    
                    status_item = QTableWidgetItem(status_text)
                    status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # Make read-only
                    
                    # Set custom widget in column 0
                    self.files_table.setCellWidget(row_index, 0, track_widget)
                    
                    # Set regular items in columns 1 and 2
                    self.files_table.setItem(row_index, 1, inputs_item)
                    self.files_table.setItem(row_index, 2, status_item)
                    
                    # Store in audio_files list
                    self.audio_files.append(track_dir)
                
                # Update UI after each batch
                QApplication.processEvents()
            
            # Re-enable table updates and ensure proper column sizing
            self.files_table.setUpdatesEnabled(True)
            
            # Ensure columns are properly sized
            self.files_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            self.files_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            self.files_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            
            # Detect input channels for profile assignment in a separate thread or with low priority timer
            # to avoid blocking the UI
            QTimer.singleShot(200, lambda: self.detect_input_channels(directory, track_dirs))
            
            # Update status bar
            self.statusBar().showMessage(f"Found {len(track_dirs)} tracks")
            self.log(f"Found {len(track_dirs)} tracks")
            
        except Exception as e:
            self.log(f"Error scanning directory: {str(e)}")
            import traceback
            traceback.print_exc()
            
        # Force garbage collection to free memory after scan
        import gc
        gc.collect()

    def detect_input_channels(self, directory, track_dirs):
        """Detect input channels across all tracks"""
        self.log("Detecting input channels...", detailed=True)
        channels = set()
        
        # First pass - collect all unique channel names across all tracks
        for track_dir in track_dirs:
            try:
                if os.path.exists(track_dir):
                    # Find all WAV files (case insensitive)
                    wav_files = [f for f in os.listdir(track_dir) if f.lower().endswith('.wav')]
                    
                    for wav in wav_files:
                        # Extract channel name from filename (e.g., Arshdeep Singh_Tr1.WAV -> Tr1)
                        basename = os.path.splitext(wav)[0]  # Remove extension
                        if '_' in basename:
                            # Extract the part after the last underscore (e.g., "Tr1")
                            channel = basename.split('_')[-1]
                        else:
                            # If no underscore, use the whole basename
                            channel = basename
                        
                        channels.add(channel)
                        self.log(f"Found channel: {channel} in {os.path.basename(track_dir)}", detailed=True)
            except Exception as e:
                self.log(f"Error scanning track {track_dir}: {str(e)}", detailed=True)
        
        # Sort channels alphabetically
        self.detected_inputs = sorted(list(channels))
        
        # Clear and rebuild track assignments table
        self.track_assignments_table.clearContents()
        self.track_assignments_table.setRowCount(len(self.detected_inputs))
        
        # Create a sorted profile list with specific order: Kirtan first, then Tabla, then Sangat, then others
        # Also ensure "Do Not Process" is available in the dropdown (for assignment) but not in profile tab
        profiles_order = []
        
        # First add "Kirtan" profiles
        if "Kirtan (Vocals)" in self.profiles:
            profiles_order.append("Kirtan (Vocals)")
        kirtan_profiles = [name for name in self.profiles.keys() if name.startswith("Kirtan") and name != "Kirtan (Vocals)"]
        profiles_order.extend(sorted(kirtan_profiles))
        
        # Then add "Tabla" profiles
        if "Tabla" in self.profiles:
            profiles_order.append("Tabla")
        tabla_profiles = [name for name in self.profiles.keys() if name.startswith("Tabla") and name != "Tabla"]
        profiles_order.extend(sorted(tabla_profiles))
        
        # Then add "Sangat" profiles
        sangat_profiles = [name for name in self.profiles.keys() if name.startswith("Sangat")]
        profiles_order.extend(sorted(sangat_profiles))
        
        # Finally add remaining profiles (excluding already added ones)
        added_profiles = set(profiles_order)
        added_profiles.add("Do Not Process")  # Don't add it twice if it's already there
        remaining_profiles = [name for name in self.profiles.keys() 
                             if name not in added_profiles and not name.startswith(("Kirtan", "Tabla", "Sangat"))]
        profiles_order.extend(sorted(remaining_profiles))
        
        # Add "Do Not Process" at the end if it exists
        if "Do Not Process" in self.profiles:
            profiles_order.append("Do Not Process")
        
        # Default profile is "Kirtan (Vocals)" if available
        default_profile = "Kirtan (Vocals)" if "Kirtan (Vocals)" in profiles_order else (profiles_order[0] if profiles_order else "")
        
        for i, channel in enumerate(self.detected_inputs):
            # Add channel name to first column
            channel_item = QTableWidgetItem(channel)
            channel_item.setFlags(channel_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # Make read-only
            self.track_assignments_table.setItem(i, 0, channel_item)
            
            # Create profile selection combobox
            profile_combo = QComboBox()
            profile_combo.addItems(profiles_order)
            
            # Set default profile based on recognized input channel names
            if channel.lower() in ["tr1", "vocal", "vocals", "lead", "kirtan", "vox", "voice"]:
                selected_profile = "Kirtan (Vocals)" if "Kirtan (Vocals)" in profiles_order else default_profile
            elif channel.lower() in ["tr2", "tabla", "tablas", "drums", "percussion", "drum"]:
                selected_profile = "Tabla" if "Tabla" in profiles_order else default_profile
            elif channel.lower() in ["tr3", "harmonium", "harm", "keys", "keyboard", "piano"]:
                selected_profile = "Sangat (Harmonium)" if "Sangat (Harmonium)" in profiles_order else default_profile
            elif channel.lower() in ["tr4", "trlr", "tamboura", "tambura", "tanpura", "drone"]:
                selected_profile = "Sangat (Tamboura)" if "Sangat (Tamboura)" in profiles_order else default_profile
            elif channel.lower() in ["trmic"]:
                selected_profile = "Room Mic" if "Room Mic" in profiles_order else default_profile
            else:
                selected_profile = default_profile
                
            # Try to find the profile in the list
            index = profile_combo.findText(selected_profile)
            if index >= 0:
                profile_combo.setCurrentIndex(index)
                
            # Set channel-profile assignment
            channel_key = channel.strip()
            self.track_profile_assignments[channel_key] = selected_profile
                
            # Connect signal to update track profile assignments
            profile_combo.currentTextChanged.connect(lambda val, ch=channel_key: self.update_track_profile(ch, val))
                
            # Add to table
            self.track_assignments_table.setCellWidget(i, 1, profile_combo)
            
        # Resize columns
        self.track_assignments_table.resizeColumnsToContents()
        self.log(f"Detected {len(self.detected_inputs)} input channels", detailed=True)

    def update_track_profile(self, channel, profile_name):
        """Update the profile assignment for a channel"""
        self.track_profile_assignments[channel] = profile_name
        self.log(f"Assigned channel '{channel}' to profile '{profile_name}'", detailed=True)

    def handle_track_name_edit(self, item):
        """Handle renaming of a track in the files table"""
        if item.column() != 0:
            return
            
        new_name = item.text().strip()
        original_path = item.data(Qt.ItemDataRole.UserRole)
        
        if not original_path or not os.path.exists(original_path):
            self.log(f"Error: Cannot rename track - path not found: {original_path}")
            return
            
        # Get parent directory and original folder name
        parent_dir = os.path.dirname(original_path)
        orig_basename = os.path.basename(original_path)
        
        # Build the new path
        new_path = os.path.join(parent_dir, new_name)
        
        # Don't do anything if the name hasn't changed
        if orig_basename == new_name:
            return
            
        # Check if target already exists
        if os.path.exists(new_path):
            QMessageBox.warning(self, "Rename Error", 
                             f"Cannot rename to '{new_name}' - a track with this name already exists.")
            # Reset the item text to the original name
            item.setText(orig_basename)
            return
        
        try:
            # Rename the directory
            os.rename(original_path, new_path)
            
            # Update the path stored in the item's user data
            item.setData(Qt.ItemDataRole.UserRole, new_path)
            
            # Update audio_files list
            if original_path in self.audio_files:
                index = self.audio_files.index(original_path)
                self.audio_files[index] = new_path
                
            # Update selected_files list
            if original_path in self.selected_files:
                index = self.selected_files.index(original_path)
                self.selected_files[index] = new_path
                
            self.log(f"Renamed track '{orig_basename}' to '{new_name}'")
            
        except Exception as e:
            self.log(f"Error renaming track: {str(e)}")
            # Reset the item text to the original name
            item.setText(orig_basename)
            QMessageBox.warning(self, "Rename Error", f"Failed to rename track: {str(e)}")

    def reprocess_all_channels(self):
        """Reprocess all channels of the selected tracks"""
        if not self.selected_files:
            QMessageBox.warning(self, "No Tracks Selected", "Please select tracks to reprocess.")
            return
        
        # Confirm with the user
            QMessageBox.warning(self, "No Segments", f"No processed segments found for track '{track_name}'.")
            return
        
        # Create segment selection dialog
        dialog = QDialog(self)
        if operation_type == "trim":
            dialog.setWindowTitle("Trim Segment")
        elif operation_type == "fade":
            dialog.setWindowTitle("Apply Fade to Segment")
        else:  # export
            dialog.setWindowTitle("Export Segment")
        
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"Select segments from track '{track_name}':"))
        
        # Add segment list
        segment_list = QListWidget()
        segment_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        
        # Sort segments by number
        segment_pattern = re.compile(r'Segment\s+(\d+)', re.IGNORECASE)
        sorted_segments = []
        
        for file in segment_files:
            match = segment_pattern.search(file)
            if match:
                segment_num = int(match.group(1))
                sorted_segments.append((segment_num, file))
        
        sorted_segments.sort()
        
        for num, file in sorted_segments:
            item = QListWidgetItem(f"Segment {num}")
            item.setData(Qt.ItemDataRole.UserRole, os.path.join(track_path, file))
            segment_list.addItem(item)
        
        layout.addWidget(segment_list)
        
        # Add operation-specific controls
        if operation_type == "trim":
            # Add trim parameters
            trim_layout = QFormLayout()
            trim_start = QDoubleSpinBox(); trim_start.setRange(-180, 180); trim_start.setDecimals(2); trim_start.setSuffix(" s")
            trim_end = QDoubleSpinBox(); trim_end.setRange(-180, 180); trim_end.setDecimals(2); trim_end.setSuffix(" s")
            trim_layout.addRow("Trim from start:", trim_start)
            trim_layout.addRow("Trim from end:", trim_end)
            
            trim_widget = QWidget()
            trim_widget.setLayout(trim_layout)
            layout.addWidget(trim_widget)
        
        elif operation_type == "fade":
            # Add fade parameters
            fade_layout = QFormLayout()
            
            fade_in = QSpinBox(); fade_in.setRange(0, 5000); fade_in.setSuffix(" ms")
            fade_out = QSpinBox(); fade_out.setRange(0, 5000); fade_out.setSuffix(" ms")
            fade_type_combo = QComboBox(); fade_type_combo.addItems(["Linear", "Logarithmic", "Exponential"])
            fade_layout.addRow("Fade in:", fade_in)
            fade_layout.addRow("Fade out:", fade_out)
            fade_layout.addRow("Fade type:", fade_type_combo)
            
            fade_widget = QWidget()
            fade_widget.setLayout(fade_layout)
            layout.addWidget(fade_widget)
        
        elif operation_type == "export":
            # Add export parameters
            export_layout = QFormLayout()
            
            fmt_combo = QComboBox(); fmt_combo.addItems(["mp3","wav","flac","ogg"]); fmt_combo.setCurrentText(self.settings.get("output_format","mp3"))
            qual_combo = QComboBox(); qual_combo.addItems(["128","192","256","320"]); qual_combo.setCurrentText(str(self.settings.get("mp3_bitrate",128)))
            export_dest = QComboBox(); export_dest.addItems(["Original track folder","Custom location"])
            
            export_layout.addRow("Format:", fmt_combo)
            export_layout.addRow("Quality:", qual_combo)
            export_layout.addRow("Destination:", export_dest)
            
            export_widget = QWidget()
            export_widget.setLayout(export_layout)
            layout.addWidget(export_widget)
        
        # Add buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        # Show dialog
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_items = segment_list.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "No Selection", "No segments were selected.")
                return
            
            selected_files = [item.data(Qt.ItemDataRole.UserRole) for item in selected_items]
            
            # Process based on operation type
            if operation_type == "trim":
                self.log(f"Trimming segments: start={trim_start.value()}s, end={trim_end.value()}s")
                self.trim_segments(selected_files, int(trim_start.value()*1000), int(trim_end.value()*1000))
            elif operation_type == "fade":
                self.apply_fade_to_segments(selected_files, fade_in.value(), fade_out.value(), fade_type_combo.currentText())
            elif operation_type == "export":
                dest = None
                if export_dest.currentText() == "Custom location":
                    dest = QFileDialog.getExistingDirectory(self, "Select Export Directory")
                    if not dest:  # User canceled
                        return
                
                self.export_segments(
                    selected_files, 
                    export_format.currentText(), 
                    int(export_quality.currentText()), 
                    dest
                )
    
    def trim_segments(self, segment_files, trim_start_ms, trim_end_ms):
        """Trim the selected segments"""
        try:
            from pydub import AudioSegment
            
            # Create and show progress dialog
            progress = ProgressDialog(
                self, 
                "Trimming Segments",
                f"Trimming {len(segment_files)} segment(s)"
            )
            
            self.log(f"Trimming segments: start={trim_start_ms/1000}s, end={trim_end_ms/1000}s")
            successful_trims = 0
            
            for index, file_path in enumerate(segment_files):
                # Update progress dialog
                progress.update_progress(index, len(segment_files))
                progress.update_file(f"Processing: {os.path.basename(file_path)}")
                progress.update_status(f"File {index + 1} of {len(segment_files)}")
                
                try:
                    # Load audio
                    progress.update_status("Loading audio file...")
                    audio = AudioSegment.from_file(file_path)
                    
                    # Check if trim values are valid
                    duration_ms = len(audio)
                    if trim_start_ms + trim_end_ms >= duration_ms:
                        self.log(f"Error: Cannot trim more than the segment duration for {os.path.basename(file_path)}")
                        progress.update_status(f"Error: Cannot trim {os.path.basename(file_path)} (too short)")
                        continue
                    
                    # Apply trimming
                    progress.update_status("Trimming audio...")
                    trimmed_audio = audio[trim_start_ms:duration_ms - trim_end_ms]
                    
                    # Create a new filename for the trimmed version
                    dir_path = os.path.dirname(file_path)
                    filename = os.path.basename(file_path)
                    name, ext = os.path.splitext(filename)
                    
                    # Create descriptive filename indicating the trim operation
                    trim_desc = ""
                    if trim_start_ms > 0:
                        trim_desc += f"_trim{int(trim_start_ms/1000)}s_start"
                    if trim_end_ms > 0:
                        trim_desc += f"_trim{int(trim_end_ms/1000)}s_end"
                    
                    new_filename = f"{name}{trim_desc}{ext}"
                    new_path = os.path.join(dir_path, new_filename)
                    
                    # If file with same name already exists, add timestamp
                    if os.path.exists(new_path):
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        new_filename = f"{name}{trim_desc}_{timestamp}{ext}"
                        new_path = os.path.join(dir_path, new_filename)
                    
                    # Export to new file
                    progress.update_status("Exporting trimmed audio...")
                    format = ext.lower().replace('.', '')
                    if format == "mp3":
                        trimmed_audio.export(
                            new_path, 
                            format=format, 
                            bitrate=f"{self.settings.get('mp3_bitrate', 128)}k"
                        )
                    else:
                        trimmed_audio.export(new_path, format=format)
                    
                    # Create new metadata json file if original one exists
                    json_file = file_path.replace('.mp3', '.json')
                    new_json_file = new_path.replace('.mp3', '.json')
                    
                    if os.path.exists(json_file):
                        try:
                            progress.update_status("Updating metadata...")
                            with open(json_file, 'r') as f:
                                metadata = json.load(f)
                            
                            # Update duration
                            metadata['duration_ms'] = len(trimmed_audio)
                            
                            # Update segment range if present
                            if 'segment_range' in metadata:
                                start_ms, end_ms = metadata['segment_range']
                                if trim_start_ms > 0:
                                    start_ms += trim_start_ms
                                if trim_end_ms > 0:
                                    end_ms -= trim_end_ms
                                metadata['segment_range'] = [start_ms, end_ms]
                            
                            # Add trim operation to history
                            if 'operations' not in metadata:
                                metadata['operations'] = []
                            
                            metadata['operations'].append({
                                'operation': 'trim',
                                'trim_start_ms': trim_start_ms,
                                'trim_end_ms': trim_end_ms,
                                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                                'original_file': filename
                            })
                            
                            # Save new metadata file
                            with open(new_json_file, 'w') as f:
                                json.dump(metadata, f, indent=2)
                        except Exception as e:
                            self.log(f"Error updating metadata for {os.path.basename(file_path)}: {str(e)}")
                    
                    self.log(f"Trimmed {os.path.basename(file_path)} ({trim_start_ms}ms from start, {trim_end_ms}ms from end)")
                    self.log(f"Exported new file: {new_filename}")
                    successful_trims += 1
                    progress.update_status("Completed successfully")
                    
                except Exception as e:
                    self.log(f"Error trimming {os.path.basename(file_path)}: {str(e)}")
                    progress.update_status(f"Error: {str(e)}")
            
            # Set progress to 100% when done
            progress.update_progress(100)
            progress.update_message(f"Completed: {successful_trims} of {len(segment_files)} files trimmed")
            progress.update_status("Operation complete")
            
            # Close progress dialog
            progress.accept()
            
            if successful_trims > 0:
                QMessageBox.information(self, "Trim Complete", f"Successfully trimmed {successful_trims} segments and saved as new files.")
            
                # Refresh the track info panel if a track is selected
                if self.selected_files:
                    self.track_info_panel.update_panel(self.selected_files[0])
                
                # Save profile assignments before scanning directory
                saved_assignments = self.track_profile_assignments.copy()
                
                # Rescan directory to update UI with new files
                if self.working_dir:
                    self.scan_directory(self.working_dir)
                    
                # Restore the profile assignments
                self.track_profile_assignments = saved_assignments
            else:
                QMessageBox.warning(self, "Trim Failed", "No segments were successfully trimmed.")
                
        except ImportError:
            QMessageBox.critical(self, "Missing Dependency", 
                               "This operation requires the pydub library. Please install it with 'pip install pydub'.")

    def apply_fade_to_segments(self, segment_files, fade_in_ms, fade_out_ms, fade_type):
        """Apply fade in/out to the selected segments"""
        try:
            from pydub import AudioSegment
            
            # Create and show progress dialog
            progress = ProgressDialog(
                self,
                "Applying Fade Effects",
                f"Applying {fade_type} fade to {len(segment_files)} segment(s)"
            )
            
            self.log(f"Applying fade: in={fade_in_ms}ms, out={fade_out_ms}ms, type={fade_type}")
            successful_fades = 0
            
            for index, file_path in enumerate(segment_files):
                # Update progress dialog
                progress.update_progress(index, len(segment_files))
                progress.update_file(f"Processing: {os.path.basename(file_path)}")
                progress.update_status(f"File {index + 1} of {len(segment_files)}")
                
                try:
                    # Load audio
                    progress.update_status("Loading audio file...")
                    audio = AudioSegment.from_file(file_path)
                    
                    # Apply fade in/out based on fade type
                    if fade_in_ms > 0:
                        progress.update_status(f"Applying {fade_type} fade in...")
                        if fade_type == "Linear":
                            audio = audio.fade_in(fade_in_ms)
                        elif fade_type == "Logarithmic":
                            audio = audio.fade_in(fade_in_ms, from_gain=-120)
                        else:  # Exponential
                            audio = audio.fade_in(fade_in_ms, from_gain=-50)
                    
                    if fade_out_ms > 0:
                        progress.update_status(f"Applying {fade_type} fade out...")
                        if fade_type == "Linear":
                            audio = audio.fade_out(fade_out_ms)
                        elif fade_type == "Logarithmic":
                            audio = audio.fade_out(fade_out_ms, to_gain=-120)
                        else:  # Exponential
                            audio = audio.fade_out(fade_out_ms, to_gain=-50)
                    
                    # Create a new filename for the faded version
                    dir_path = os.path.dirname(file_path)
                    filename = os.path.basename(file_path)
                    name, ext = os.path.splitext(filename)
                    
                    # Create descriptive filename indicating the fade operation
                    fade_desc = ""
                    if fade_in_ms > 0:
                        fade_desc += f"_fade{int(fade_in_ms)}ms_in"
                    if fade_out_ms > 0:
                        fade_desc += f"_fade{int(fade_out_ms)}ms_out"
                    fade_desc += f"_{fade_type.lower()}"
                    
                    new_filename = f"{name}{fade_desc}{ext}"
                    new_path = os.path.join(dir_path, new_filename)
                    
                    # If file with same name already exists, add timestamp
                    if os.path.exists(new_path):
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        new_filename = f"{name}{fade_desc}_{timestamp}{ext}"
                        new_path = os.path.join(dir_path, new_filename)
                    
                    # Export to new file
                    progress.update_status("Exporting audio with fade effects...")
                    format = ext.lower().replace('.', '')
                    if format == "mp3":
                        audio.export(
                            new_path, 
                            format=format, 
                            bitrate=f"{self.settings.get('mp3_bitrate', 128)}k"
                        )
                    else:
                        audio.export(new_path, format=format)
                    
                    # Create new metadata json file if original one exists
                    json_file = file_path.replace('.mp3', '.json')
                    new_json_file = new_path.replace('.mp3', '.json')
                    
                    if os.path.exists(json_file):
                        try:
                            progress.update_status("Updating metadata...")
                            with open(json_file, 'r') as f:
                                metadata = json.load(f)
                            
                            # Add fade operation to history
                            if 'operations' not in metadata:
                                metadata['operations'] = []
                            
                            metadata['operations'].append({
                                'operation': 'fade',
                                'fade_in_ms': fade_in_ms,
                                'fade_out_ms': fade_out_ms,
                                'fade_type': fade_type,
                                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                                'original_file': filename
                            })
                            
                            # Save new metadata file
                            with open(new_json_file, 'w') as f:
                                json.dump(metadata, f, indent=2)
                                
                        except Exception as e:
                            self.log(f"Error updating metadata for {os.path.basename(file_path)}: {str(e)}")
                    
                    self.log(f"Applied {fade_type} fade to {os.path.basename(file_path)} ({fade_in_ms}ms in, {fade_out_ms}ms out)")
                    self.log(f"Exported new file: {new_filename}")
                    successful_fades += 1
                    progress.update_status("Completed successfully")
                    
                except Exception as e:
                    self.log(f"Error applying fade to {os.path.basename(file_path)}: {str(e)}")
                    progress.update_status(f"Error: {str(e)}")
            
            # Set progress to 100% when done
            progress.update_progress(100)
            progress.update_message(f"Completed: {successful_fades} of {len(segment_files)} files processed")
            progress.update_status("Operation complete")
            
            # Close progress dialog
            progress.accept()
            
            if successful_fades > 0:
                QMessageBox.information(self, "Fade Complete", f"Successfully applied fade to {successful_fades} segments and saved as new files.")
                
                # Refresh the track info panel if a track is selected
                if self.selected_files:
                    self.track_info_panel.update_panel(self.selected_files[0])
                
                # Save profile assignments before scanning directory
                saved_assignments = self.track_profile_assignments.copy()
                
                # Rescan directory to update UI with new files
                if self.working_dir:
                    self.scan_directory(self.working_dir)
                    
                # Restore the profile assignments
                self.track_profile_assignments = saved_assignments
            else:
                QMessageBox.warning(self, "Fade Failed", "No segments were successfully faded.")
            
        except ImportError:
            QMessageBox.critical(self, "Missing Dependency", 
                              "This operation requires the pydub library. Please install it with 'pip install pydub'.")

    def update_progress(self, value):
        """Update the progress bar value"""
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(value)
            QApplication.processEvents()  # Process UI events to show updates immediately

    def show_segment_selection_dialog(self, operation_type, track_path):
        """Show dialog to select segments for operations like trim, fade, or export"""
        if not track_path or not os.path.exists(track_path):
            QMessageBox.warning(self, "Invalid Track", "Selected track folder does not exist.")
            return

        # Find all MP3 segments in the folder
        segment_files = []
        try:
            for file in os.listdir(track_path):
                if file.lower().endswith('.mp3') and 'segment' in file.lower():
                    segment_files.append(os.path.join(track_path, file))
            
            if not segment_files:
                QMessageBox.warning(self, "No Segments", "No segments found in this track folder.")
                return
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error listing segment files: {str(e)}")
            return
        
        track_name = os.path.basename(track_path)
        
        # Create segment selection dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{operation_type.title()} Segments")
        dialog.setMinimumWidth(500)
        
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"Select segments from track '{track_name}':"))
        
        # Add segment list
        segment_list = QListWidget()
        segment_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        
        # Sort segments by number
        segment_pattern = re.compile(r'Segment\s+(\d+)', re.IGNORECASE)
        sorted_segments = []
        
        for file in segment_files:
            filename = os.path.basename(file)
            match = segment_pattern.search(filename)
            if match:
                segment_num = int(match.group(1))
                sorted_segments.append((segment_num, file))
            else:
                sorted_segments.append((999, file))  # Put unsorted at end
        
        sorted_segments.sort()
        
        for num, file in sorted_segments:
            item = QListWidgetItem(os.path.basename(file))
            item.setData(Qt.ItemDataRole.UserRole, file)
            segment_list.addItem(item)
        
        layout.addWidget(segment_list)
        
        # Add operation-specific controls
        if operation_type == "trim":
            trim_group = QGroupBox("Trim Settings")
            trim_layout = QFormLayout(trim_group)
            
            trim_start = QSpinBox()
            trim_start.setRange(0, 60000)
            trim_start.setSingleStep(100)
            trim_start.setSuffix(" ms")
            trim_start.setValue(500)
            
            trim_end = QSpinBox()
            trim_end.setRange(0, 60000)
            trim_end.setSingleStep(100)
            trim_end.setSuffix(" ms")
            trim_end.setValue(500)
            
            trim_layout.addRow("Trim from start:", trim_start)
            trim_layout.addRow("Trim from end:", trim_end)
            
            layout.addWidget(trim_group)
            
        elif operation_type == "fade":
            fade_group = QGroupBox("Fade Settings")
            fade_layout = QFormLayout(fade_group)
            
            fade_in = QSpinBox()
            fade_in.setRange(0, 30000)
            fade_in.setSingleStep(100)
            fade_in.setSuffix(" ms")
            fade_in.setValue(1000)
            
            fade_out = QSpinBox()
            fade_out.setRange(0, 30000)
            fade_out.setSingleStep(100)
            fade_out.setSuffix(" ms")
            fade_out.setValue(1500)
            
            fade_type = QComboBox()
            fade_type.addItems(["Linear", "Exponential", "Logarithmic"])
            
            fade_layout.addRow("Fade in:", fade_in)
            fade_layout.addRow("Fade out:", fade_out)
            fade_layout.addRow("Fade type:", fade_type)
            
            layout.addWidget(fade_group)
            
        elif operation_type == "export":
            export_group = QGroupBox("Export Settings")
            export_layout = QFormLayout(export_group)
            
            format_combo = QComboBox()
            format_combo.addItems(["mp3", "wav", "flac", "ogg"])
            
            quality_combo = QComboBox()
            quality_combo.addItems(["128", "192", "256", "320"])
            
            export_layout.addRow("Format:", format_combo)
            export_layout.addRow("Quality:", quality_combo)
            
            # Add destination folder selection
            dest_layout = QHBoxLayout()
            dest_edit = QLineEdit()
            dest_edit.setReadOnly(True)
            dest_edit.setPlaceholderText("Use same folder as source")
            
            browse_btn = QPushButton("Browse...")
            browse_btn.clicked.connect(lambda: self._browse_export_destination(dest_edit))
            
            dest_layout.addWidget(dest_edit)
            dest_layout.addWidget(browse_btn)
            
            export_layout.addRow("Destination:", dest_layout)
            
            layout.addWidget(export_group)
        
        # Add buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        # Show dialog
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Get selected segments
            selected_segments = []
            for item in segment_list.selectedItems():
                selected_segments.append(item.data(Qt.ItemDataRole.UserRole))
            
            if not selected_segments:
                QMessageBox.warning(self, "No Selection", "No segments selected.")
                return
                
            # Perform the operation
            if operation_type == "trim":
                self.trim_segments(selected_segments, trim_start.value(), trim_end.value())
            elif operation_type == "fade":
                self.apply_fade_to_segments(selected_segments, fade_in.value(), fade_out.value(), fade_type.currentText())
            elif operation_type == "export":
                self.export_segments(selected_segments, format_combo.currentText(), quality_combo.currentText(), 
                                     dest_edit.text() if dest_edit.text() else None)
    
    def _browse_export_destination(self, edit_widget):
        """Browse for export destination folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Export Destination")
        if folder:
            edit_widget.setText(folder)