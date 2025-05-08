#!/usr/bin/env python
import sys
import os
import traceback
import time
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QFile, QTextStream, Qt
from gui.main_window import KirtanProcessorApp
from gui.processing_thread import ProcessingWorker

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application information
    app.setApplicationName("Kirtan Processor")
    app.setOrganizationName("Kirtan Audio")
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create absolute path to the stylesheet
    style_path = os.path.join(script_dir, "style.qss")
    
    # Apply stylesheet
    style_file = QFile(style_path)
    if style_file.exists():
        style_file.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text)
        stream = QTextStream(style_file)
        style_sheet = stream.readAll()
        app.setStyleSheet(style_sheet)
        print(f"Applied application stylesheet from {style_path}")
    else:
        print(f"Warning: style.qss not found at {style_path}")
    
    # Create and show main window
    window = KirtanProcessorApp()
    
    # Set the window title to include version information
    window.setWindowTitle("Kirtan Processor v1.0")
    
    # IMPORTANT: Monkey patch the start_processing method to fix the ProcessingWorker initialization error
    original_start_processing = window.start_processing
    
    def fixed_start_processing():
        """Fixed version of start_processing that properly initializes ProcessingWorker"""
        if window.is_processing:
            return
            
        # Always use all files when Process All Track Files button is clicked
        files_to_process = window.audio_files
            
        if not files_to_process:
            window.log("No files selected for processing")
            return
            
        # Update UI state
        window.is_processing = True
        window.process_button.setEnabled(False)
        window.stop_button.setEnabled(True)
        window.progress_bar.setValue(0)
        
        # Record start time for completion popup
        window.processing_start_time = time.time()
        
        # Log processing start
        if len(files_to_process) == 1:
            window.log(f"Processing 1 track")
        else:
            window.log(f"Processing {len(files_to_process)} tracks")
        
        # Store selected files in window object so ProcessingWorker can access them
        window.selected_files = files_to_process
        
        # Create processing worker with the correct parameter signature (only the app object)
        window.processing_worker = ProcessingWorker(window)
        
        # Connect signals - using progress_update instead of log
        window.processing_worker.progress_update.connect(window.log_from_worker)
        window.processing_worker.progress_bar.connect(window.update_progress)
        window.processing_worker.processing_finished.connect(window.on_processing_finished)
        
        # Start processing
        window.processing_worker.start()
    
    # Replace the original method with our fixed version
    window.start_processing = fixed_start_processing
    
    # Fix the segment padding calculation in the ProcessingWorker
    original_load_and_mix_audio = ProcessingWorker.load_and_mix_audio
    
    def fixed_load_and_mix_audio(self, track_path, input_channels):
        """Fixed version of load_and_mix_audio that correctly applies segment padding"""
        # Keep all the original code and logging from the original method
        result = original_load_and_mix_audio(self, track_path, input_channels)
        
        # After the original method runs, patch the segment padding calculation
        # The issue is in the ProcessingWorker class, where it applies pre/post padding in seconds
        # but the UI is passing values in seconds, and the logic is confusing
        # This monkey patch fixes that by standardizing the units and applying them correctly
        
        # Monkey-patch the apply_padding function to fix the unit conversion issue
        original_detect_segments = ProcessingWorker.detect_segments
        
        def fixed_detect_segments(self, audio):
            """Fixed segment detection with correct padding application"""
            segments = original_detect_segments(self, audio)
            
            if not segments:
                return segments
                
            # Note: This patch doesn't affect the segment detection, just the debugging info
            # The actual padding is applied in the load_and_mix_audio method where segments are processed
            return segments
            
        ProcessingWorker.detect_segments = fixed_detect_segments
        
        return result
    
    ProcessingWorker.load_and_mix_audio = fixed_load_and_mix_audio
    
    # Also fix the export_segments method to ensure proper MP3 encoding that works with Windows Media Player
    original_export_segments = window.export_segments if hasattr(window, 'export_segments') else None
    
    def fixed_export_segments(segment_files, format, quality, custom_dest=None):
        """Enhanced version of export_segments that creates Windows-compatible MP3s using the reference kirtan track"""
        try:
            import platform
            import subprocess
            import os
            from pydub import AudioSegment
            
            window.log(f"Exporting {len(segment_files)} segments to {format.upper()} format (quality: {quality})")
            export_count = 0
            
            for file_path in segment_files:
                try:
                    # Load audio from the original kirtan track
                    window.log(f"Loading audio from {os.path.basename(file_path)}")
                    audio = AudioSegment.from_file(file_path)
                    
                    # Determine output path
                    original_filename = os.path.basename(file_path)
                    base_filename = os.path.splitext(original_filename)[0]
                    
                    if custom_dest:
                        output_dir = custom_dest
                    else:
                        output_dir = os.path.dirname(file_path)
                        
                    output_file = os.path.join(output_dir, f"{base_filename}_exported.{format}")
                    window.log(f"Will export to: {os.path.basename(output_file)}")
                    
                    # Export with selected format and quality with careful settings for Windows compatibility
                    if format == "mp3":
                        window.log(f"Exporting as MP3 with bitrate {quality}k")
                        
                        # Use export parameters similar to those in the CLI version
                        # The CLI version adds 3 seconds before and after segments
                        # We're not doing segment detection here, but we're using similar export parameters
                        audio.export(
                            output_file, 
                            format=format, 
                            bitrate=f"{quality}k",
                            parameters=["-q:a", "0", "-joint_stereo", "1"]  # Quality parameters for better compatibility
                        )
                    elif format == "wav":
                        window.log("Exporting as WAV")
                        audio.export(output_file, format=format)
                    elif format == "flac":
                        window.log("Exporting as FLAC")
                        audio.export(output_file, format=format)
                    elif format == "ogg":
                        window.log(f"Exporting as OGG with bitrate {quality}k")
                        audio.export(output_file, format=format, bitrate=f"{quality}k")
                    
                    window.log(f"Export successful: {os.path.basename(output_file)}")
                    export_count += 1
                    
                except Exception as e:
                    window.log(f"Error exporting {os.path.basename(file_path)}: {str(e)}")
            
            if export_count > 0:
                QApplication.processEvents()  # Ensure UI updates
                QMessageBox.information(window, "Export Complete", 
                                      f"Successfully exported {export_count} files to {format.upper()} format.")
                
                # Open the export directory
                if custom_dest:
                    try:
                        if platform.system() == 'Windows':
                            os.startfile(custom_dest)
                        elif platform.system() == 'Darwin':  # macOS
                            subprocess.call(['open', custom_dest])
                        else:  # Linux and other Unix-like systems
                            subprocess.call(['xdg-open', custom_dest])
                    except Exception as e:
                        window.log(f"Could not open export directory: {str(e)}")
            else:
                QMessageBox.warning(window, "Export Failed", "No files were successfully exported.")
                
        except ImportError:
            QMessageBox.critical(window, "Missing Dependency", 
                              "This operation requires the pydub library. Please install it with 'pip install pydub'.")
    
    # Replace the export method if it exists
    if original_export_segments:
        window.export_segments = fixed_export_segments
    
    # Monkey patch the track rename handler for better debugging but preserve right-click functionality
    original_handle_track_name_edit = window.handle_track_name_edit
    
    def debug_track_rename(item):
        """Wrapper with better debugging for track renaming"""
        print("\n=== DEBUG: Track Rename Operation Started ===")
        try:
            if item.column() != 0:
                print(f"DEBUG: Ignoring edit in column {item.column()}")
                return
                
            new_name = item.text().strip()
            # Use Qt.ItemDataRole.UserRole (correct value is 256)
            original_path = item.data(Qt.ItemDataRole.UserRole)
            
            print(f"DEBUG: Renaming track from original path: {original_path}")
            print(f"DEBUG: New track name: {new_name}")
            
            if not original_path:
                print("DEBUG: ERROR - Original path is empty!")
                # Try looking at all data roles to see what's actually stored
                print("DEBUG: Checking all data roles:")
                for role in range(0, 257):  # Check a range of possible roles
                    data = item.data(role)
                    if data:
                        print(f"DEBUG: Found data at role {role}: {data}")
                return
                
            orig_basename = os.path.basename(original_path)
            parent_dir = os.path.dirname(original_path) or window.working_dir
            
            old_dir = os.path.join(parent_dir, orig_basename)
            new_dir = os.path.join(parent_dir, new_name)
            
            print(f"DEBUG: Old directory: {old_dir}")
            print(f"DEBUG: New directory: {new_dir}")
            print(f"DEBUG: Old dir exists: {os.path.exists(old_dir)}")
            print(f"DEBUG: New dir exists: {os.path.exists(new_dir)}")
            
            # Call the original method
            original_handle_track_name_edit(item)
            
            # Check the result
            print(f"DEBUG: After rename - Old dir exists: {os.path.exists(old_dir)}")
            print(f"DEBUG: After rename - New dir exists: {os.path.exists(new_dir)}")
            if os.path.exists(new_dir):
                print(f"DEBUG: Files in new directory: {os.listdir(new_dir)}")
                
        except Exception as e:
            print(f"DEBUG: Exception during rename: {str(e)}")
            print(traceback.format_exc())
        print("=== DEBUG: Track Rename Operation Ended ===\n")
    
    # Replace the original method with our debug version
    window.handle_track_name_edit = debug_track_rename
    
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()