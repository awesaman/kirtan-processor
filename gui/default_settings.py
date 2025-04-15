# This file is auto-generated. Manual changes may be overwritten.

SETTINGS_VERSION = "1.0"

DEFAULT_PROFILES = {"Kirtan (Vocals)": {"gain": 0, "normalize": {"enabled": False, "target_level": 0.0, "headroom": 1.0, "method": "peak"}, "dynamic_processing": {"enabled": False, "compressor": {"threshold": -18.0, "ratio": 3.0, "attack": 20.0, "release": 150.0}, "limiter": {"threshold": -1.0, "release": 50.0}}, "low_pass": False, "low_pass_freq": 10000, "low_pass_db": -12, "high_pass": False, "high_pass_freq": 150, "high_pass_db": -12, "pan": 0}, "Tabla": {"gain": 0, "normalize": {"enabled": False, "target_level": 0.0, "headroom": 1.5, "method": "peak"}, "dynamic_processing": {"enabled": False, "compressor": {"threshold": -20.0, "ratio": 3.0, "attack": 10.0, "release": 120.0}, "limiter": {"threshold": -1.0, "release": 40.0}}, "low_pass": False, "low_pass_freq": 8000, "low_pass_db": -12, "high_pass": False, "high_pass_freq": 80, "high_pass_db": -12, "pan": 0}, "Sangat (Ambient)": {"gain": 0, "normalize": {"enabled": False, "target_level": 0.0, "headroom": 2.0, "method": "peak"}, "dynamic_processing": {"enabled": False, "compressor": {"threshold": -24.0, "ratio": 2.0, "attack": 40.0, "release": 400.0}, "limiter": {"threshold": -1.5, "release": 100.0}}, "low_pass": False, "low_pass_freq": 8000, "low_pass_db": -12, "high_pass": False, "high_pass_freq": 60, "high_pass_db": -12, "pan": 0}}

DEFAULT_SEGMENTATION = {"silence_threshold": 21, "min_silence": 4000, "seek_step": 2000, "min_time_between_segments": 10000, "min_segment_length": 15, "dropout": 1, "pre_segment_padding": 0, "post_segment_padding": 0}

DEFAULT_EXPORT = {"bitrate": 128, "album": "", "fade_in": 0, "fade_out": 0, "save_unsegmented": False, "trim_only": False, "batch_normalize": False}

DEFAULT_VISUALIZATION = {"show_waveform": False}

DEFAULT_PROCESSING = {"speed": "Full Speed", "resource_usage": "auto"}


# Function to get a deep copy of all default settings
def get_default_settings():
    """Return a deep copy of all default settings."""
    import copy
    
    settings = {
        "profiles": copy.deepcopy(DEFAULT_PROFILES),
        "segmentation": copy.deepcopy(DEFAULT_SEGMENTATION),
        "export": copy.deepcopy(DEFAULT_EXPORT),
        "visualization": copy.deepcopy(DEFAULT_VISUALIZATION),
        "processing": copy.deepcopy(DEFAULT_PROCESSING),
        "version": SETTINGS_VERSION
    }
    
    return settings
