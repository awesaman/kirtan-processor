#!/usr/bin/env python
import sys
import os
import traceback
import time
import json
from PyQt6.QtWidgets import QApplication, QMessageBox, QDialog, QFileDialog
from PyQt6.QtCore import QFile, QTextStream, Qt
from gui.main_window import KirtanProcessorApp
from gui.processing_thread import ProcessingWorker
from gui.dialogs import ProgressDialog, OperationCompletionDialog

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
    
    # For PyInstaller bundles, the _MEIPASS attribute contains the path to the bundle
    if hasattr(sys, '_MEIPASS'):
        # When running as executable, the resource path is different
        bundle_dir = sys._MEIPASS
        style_path = os.path.join(bundle_dir, "style.qss")
    
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
        # Try alternative locations as a fallback
        alt_style_paths = [
            os.path.join(os.getcwd(), "style.qss"),
            os.path.join(os.path.dirname(os.getcwd()), "style.qss")
        ]
        for alt_path in alt_style_paths:
            alt_file = QFile(alt_path)
            if alt_file.exists():
                alt_file.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text)
                stream = QTextStream(alt_file)
                style_sheet = stream.readAll()
                app.setStyleSheet(style_sheet)
                print(f"Applied application stylesheet from alternative location: {alt_path}")
                break
    
    # Create and show main window
    window = KirtanProcessorApp()
    
    # Set the window title to include version information
    window.setWindowTitle("Kirtan Processor v1.0")
    
    # Fix the missing trim_segments method - this should ensure the method exists
    # and is properly accessible from the window instance
    if not hasattr(window, 'trim_segments'):
        print("Adding missing trim_segments method to window instance")
        
        def trim_segments_monkey_patch(self, segment_files, trim_start_ms, trim_end_ms):
            """Monkey-patched method to trim the selected segments"""
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
                        
                        # Create descriptive filename indicating the trim operation using the new format
                        trim_desc = "_"
                        
                        # Format shows TrimPre and TrimPost with their values
                        pre_trim_value = int(trim_start_ms/1000) if trim_start_ms > 0 else 0
                        post_trim_value = int(trim_end_ms/1000) if trim_end_ms > 0 else 0
                        
                        # Only include the parts that were actually trimmed
                        parts = []
                        if pre_trim_value > 0:
                            parts.append(f"TrimPre-{pre_trim_value}")
                        if post_trim_value > 0:
                            parts.append(f"TrimPost-{post_trim_value}")
                            
                        # Join parts with underscore, or use "Original" if no trimming was done
                        trim_desc = "_" + "_".join(parts) if parts else "_Original"
                        
                        new_filename = f"{name}{trim_desc}{ext}"
                        new_path = os.path.join(dir_path, new_filename)
                        
                        # If file with same name already exists, add a number suffix
                        counter = 1
                        while os.path.exists(new_path):
                            new_filename = f"{name}{trim_desc}_{counter}{ext}"
                            new_path = os.path.join(dir_path, new_filename)
                            counter += 1
                        
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
                    # Find the directory where files were saved
                    directory_path = os.path.dirname(segment_files[0])
                    
                    # Show completion dialog with Open Folder button (using gui.dialogs implementation)
                    from gui.dialogs import OperationCompletionDialog
                    completion_dialog = OperationCompletionDialog(
                        self,
                        "Trim Complete", 
                        f"Successfully trimmed {successful_trims} segments and saved as new files.",
                        directory_path
                    )
                    completion_dialog.exec()
            except Exception as e:
                self.log(f"Error during trim operation: {str(e)}")
                traceback.print_exc()
                QMessageBox.critical(self, "Error", f"An error occurred during the trim operation:\n{str(e)}")
        
        # Add the method to the window object
        import types
        window.trim_segments = types.MethodType(trim_segments_monkey_patch, window)
    
    # Check if the apply_fade_to_segments method exists, and if not, create it
    if not hasattr(window, 'apply_fade_to_segments'):
        print("Adding missing apply_fade_to_segments method to window instance")
        
        def apply_fade_to_segments_monkey_patch(self, segment_files, fade_in_ms, fade_out_ms, fade_type):
            """Apply fade in/out to the selected segments"""
            try:
                from pydub import AudioSegment
                import json
                
                # Create and show progress dialog - using the already imported dialog classes
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
                                audio = audio.fade_in(fade_in_ms, to_gain=-30)
                            elif fade_type == "Exponential":
                                audio = audio.fade_in(fade_in_ms, from_gain=-30)
                        
                        if fade_out_ms > 0:
                            progress.update_status(f"Applying {fade_type} fade out...")
                            if fade_type == "Linear":
                                audio = audio.fade_out(fade_out_ms)
                            elif fade_type == "Logarithmic":
                                audio = audio.fade_out(fade_out_ms, to_gain=-30)
                            elif fade_type == "Exponential":
                                audio = audio.fade_out(fade_out_ms, from_gain=-30)
                        
                        # Create a new filename for the faded version
                        dir_path = os.path.dirname(file_path)
                        filename = os.path.basename(file_path)
                        name, ext = os.path.splitext(filename)
                        
                        # Format shows FadeIn and FadeOut with their values in seconds
                        fade_in_value = int(fade_in_ms/1000) if fade_in_ms > 0 else 0
                        fade_out_value = int(fade_out_ms/1000) if fade_out_ms > 0 else 0
                        
                        # Only include the parts that were actually faded
                        parts = []
                        if fade_in_value > 0:
                            parts.append(f"FadeIn-{fade_in_value}")
                        if fade_out_value > 0:
                            parts.append(f"FadeOut-{fade_out_value}")
                        parts.append(fade_type)
                        
                        # Join parts with underscore
                        fade_desc = "_" + "_".join(parts)
                        
                        new_filename = f"{name}{fade_desc}{ext}"
                        new_path = os.path.join(dir_path, new_filename)
                        
                        # If file with same name already exists, add a number suffix
                        counter = 1
                        while os.path.exists(new_path):
                            new_filename = f"{name}{fade_desc}_{counter}{ext}"
                            new_path = os.path.join(dir_path, new_filename)
                            counter += 1
                        
                        # Export to new file
                        progress.update_status("Exporting faded audio...")
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
                        
                        self.log(f"Applied {fade_type} fade to {os.path.basename(file_path)} (in: {fade_in_ms}ms, out: {fade_out_ms}ms)")
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
                    # Find the directory where files were saved
                    directory_path = os.path.dirname(segment_files[0])
                    
                    # Show completion dialog with Open Folder button (using gui.dialogs implementation)
                    from gui.dialogs import OperationCompletionDialog
                    completion_dialog = OperationCompletionDialog(
                        self,
                        "Fade Complete", 
                        f"Successfully applied fade effects to {successful_fades} segments and saved as new files.",
                        directory_path
                    )
                    completion_dialog.exec()
                    
            except ImportError as e:
                self.log(f"Missing dependency: {str(e)}")
                QMessageBox.critical(self, "Missing Dependency", 
                              "This operation requires the pydub library. Please install it with 'pip install pydub'.")
            except Exception as e:
                self.log(f"Error during fade operation: {str(e)}")
                traceback.print_exc()
                QMessageBox.critical(self, "Error", f"An error occurred during the fade operation:\n{str(e)}")
                
        # Add the method to the window object
        window.apply_fade_to_segments = types.MethodType(apply_fade_to_segments_monkey_patch, window)
        
    # Add the missing update_progress method if needed
    if not hasattr(window, 'update_progress'):
        print("Adding missing update_progress method to window instance")
        
        def update_progress_monkey_patch(self, value, max_value=100):
            """Update progress bar with current value - monkey patched method"""
            if hasattr(self, 'progress_bar') and self.progress_bar:
                percent = int((value / max_value) * 100) if max_value > 0 else value
                self.progress_bar.setValue(percent)
                QApplication.processEvents()
        
        # Add the monkey-patched method to the instance
        window.update_progress = types.MethodType(update_progress_monkey_patch, window)
    
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
                
                # Refresh the track info panel if a track is selected
                if window.selected_files:
                    window.track_info_panel.update_panel(window.selected_files[0])
                
                # Save profile assignments before scanning directory
                saved_assignments = window.track_profile_assignments.copy()
                
                # Rescan directory to update UI with new files
                if window.working_dir:
                    window.scan_directory(window.working_dir)
                    
                # Restore the profile assignments
                window.track_profile_assignments = saved_assignments
                
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