"""Constants used throughout the application"""

# Input/output formats
INPUT_FORMAT = '.wav'
OUTPUT_FORMAT = '.mp3'

# Default silence detection settings
DEFAULT_SILENCE_THRESHOLD = -40  # dB
DEFAULT_MIN_SILENCE_LEN = 2000   # ms
DEFAULT_MIN_SEGMENT_LEN = 30000  # ms (30 seconds)

# Default settings
DEFAULT_SETTINGS = {
    # Segmentation settings
    "silence_threshold": 21,  # dB below average level
    "min_silence": 4000,      # ms
    "seek_step": 100,         # ms
    "min_segment": 60000,     # ms (1 minute)
    "pre_segment_padding": 0, # ms
    "post_segment_padding": 0, # ms
    
    # Normalization settings
    "target_level": -14.0,    # LUFS for master output
    "peak_limit": -1.0,       # dB for peak limiting
    
    # Processing settings
    "process_speed": "Fast",  # Fast, Normal, Full Speed
    "mp3_bitrate": 192,       # kbps
    "output_format": "mp3",   # mp3, wav
    
    # Display settings
    "show_all_operations": False,  # Show detailed progress
    "confirm_overwrite": True,     # Ask before overwriting
    "cache_limit_mb": 1024,        # Audio cache size
}

# Normalization types
NORMALIZATION_TYPES = {
    "RMS": "RMS (Average Power)",
    "LUFS": "LUFS (Perceived Loudness)",
    "Peak": "Peak (Maximum Level)"
}

# Process speed settings
PROCESS_SPEEDS = {
    "Fast": {
        "description": "Optimized for speed, may use more memory",
        "chunk_size": 60,  # seconds
    },
    "Normal": {
        "description": "Balanced speed and memory usage",
        "chunk_size": 30,  # seconds
    },
    "Memory-Efficient": {
        "description": "Optimized for lower memory usage",
        "chunk_size": 10,  # seconds
    },
    "Full Speed": {
        "description": "No chunking, uses maximum memory",
        "chunk_size": 0,  # seconds (no chunking)
    }
}

# Default processing profiles - basic version
# Full profiles are defined in defaults.py
DEFAULT_PROFILE_NAMES = [
    "Kirtan (Vocals)",
    "Tabla", 
    "Sangat (Harmonium)",
    "Sangat (Tamboura)",
    "Sangat (Ambient)",
    "Sangat (Other)",
    "Create New Profile..."
]