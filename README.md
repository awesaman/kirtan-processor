# Kirtan Processor

A specialized audio processing application designed for processing and cleaning kirtan recordings. The application provides both a GUI interface for interactive use and a CLI version for batch processing.

## Features

- Audio silence detection and trimming
- Metadata Tagging
- Volume normalization
- Audio dynamics processing
- File batch processing
- Customizable processing settings

## Getting Started

### Prerequisites

To run the application from source, you need Python 3.9+ and the required dependencies:

```
pip install -r requirements.txt
```

### Running the Application

#### Graphical User Interface

```
python Kirtan-Processor-gui.py
```

#### Command Line Interface

```
python Kirtan-Processor-cli.py
```

### Creating an Executable

Use the scripts in the `build-tools` folder to create standalone executables:

```
cd build-tools
python create_exe.py
```

For more options:
```
create_exe_advanced.bat
```

See the [build-tools README](build-tools/README.md) for more details.

### Cross-Platform Builds

#### Windows (.exe)
The instructions above will create a Windows executable (.exe) file.

#### macOS (.app/.dmg)
For macOS, use the dedicated build script:

```
cd build-tools
python create_mac_app.py
```

This will:
1. Create an .icns icon file suitable for macOS
2. Build a .app bundle with PyInstaller
3. Create a .dmg installer if the create-dmg utility is installed

**Requirements on macOS:**
- PyInstaller: `pip install pyinstaller pillow`
- create-dmg: `brew install create-dmg`

Note: While the script can be run on Windows (via `create_mac_app.bat`), it will only simulate the process. Actual .app and .dmg creation requires running on a macOS system.

#### iOS
Note: .exe files are Windows-specific and cannot run on iOS. Creating an iOS app would require:

1. Rewriting the application using Swift or Objective-C
2. Using Xcode for development
3. Potentially rewriting the audio processing in iOS-compatible libraries
4. Distribution through the App Store

This is beyond the scope of this project's current implementation.

## Project Structure

### Main Scripts

- **Kirtan-Processor-gui.py** - Graphical user interface for the Kirtan Processor application
- **Kirtan-Processor-cli.py** - Command-line interface for batch processing audio files
- **style.qss** - Qt stylesheet for the GUI application
- **default.json** - Default settings for the application

### Folders

- **audio/** - Core audio processing functionality
  - **detection.py** - Silence detection algorithms
  - **dynamics.py** - Audio dynamics processing
  - **export.py** - Functions for exporting processed audio
  - **filters.py** - Audio filtering algorithms
  - **normalization.py** - Volume normalization functionality

- **build-tools/** - Scripts for creating executable distributions
  - **create_exe.py** - Python script for creating Windows executables
  - **KirtanProcessor.spec** - PyInstaller specification file

- **config/** - Application configuration
  - **constants.py** - Application-wide constants
  - **defaults.py** - Default configuration settings

- **core/** - Core application logic
  - **processor.py** - Main processing engine

- **gui/** - Graphical user interface components
  - **main_window.py** - Main application window
  - **dialogs.py** - Dialog boxes for user interaction
  - **processing_thread.py** - Background processing thread
  - **html_log_viewer.py** - Log viewer component

- **images/** - Application icons and images

- **utils/** - Utility functions
  - **app_logging.py** - Logging utilities
  - **cache.py** - Caching functions
  - **file_utils.py** - File handling utilities
  - **performance.py** - Performance monitoring

- **vad_feature/** - Voice activity detection functionality (Experimental - In Progress)

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on the code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* PyQt6 for the GUI framework
* pydub for audio processing capabilities