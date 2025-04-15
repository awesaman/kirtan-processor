"""
This file contains the factory default settings for the Kirtan Processor.
These settings should NEVER be modified as they serve as the ultimate fallback.
"""

SETTINGS_VERSION = "1.0"

FACTORY_SEGMENTATION = {
    "silence_threshold": -40,
    "min_silence": 1000,
    "seek_step": 100,
    "min_time": 60,
    "min_segment": 30,
    "dropout": 0.5,
    "pre_padding": 500,
    "post_padding": 500
}

FACTORY_EXPORT = {
    "bitrate": 128,
    "album": "",
    "fade_in": 0,
    "fade_out": 0,
    "save_unsegmented": False,
    "trim_only": False,
    "batch_normalize": True
}

FACTORY_VISUALIZATION = {
    "show_waveform": False
}

FACTORY_PROCESSING = {
    "speed": "balanced",
    "resource_usage": "auto"
}

FACTORY_PROFILES = {
    "Kirtan (Vocals)": {
        "gain": 0,
        "normalize": {
            "enabled": True,
            "target_level": -3.0,
            "headroom": 2.0,
            "method": "peak"
        },
        "dynamic_processing": {
            "enabled": True,
            "compressor": {
                "threshold": -18.0,
                "ratio": 2.5,
                "attack": 20.0,
                "release": 250.0
            },
            "limiter": {
                "threshold": -1.0,
                "release": 50.0
            }
        },
        "low_pass": False,
        "low_pass_freq": 8000,
        "low_pass_db": -12,
        "high_pass": True,
        "high_pass_freq": 120,
        "high_pass_db": -12,
        "pan": 0
    },
    "Tabla": {
        "gain": 0,
        "normalize": {
            "enabled": True,
            "target_level": -2.0,
            "headroom": 1.5,
            "method": "peak"
        },
        "dynamic_processing": {
            "enabled": True,
            "compressor": {
                "threshold": -20.0,
                "ratio": 3.0,
                "attack": 10.0,
                "release": 120.0
            },
            "limiter": {
                "threshold": -1.0,
                "release": 40.0
            }
        },
        "low_pass": False,
        "low_pass_freq": 8000,
        "low_pass_db": -12,
        "high_pass": True,
        "high_pass_freq": 80,
        "high_pass_db": -12,
        "pan": 0
    },
    "Sangat (Ambient)": {
        "gain": 0,
        "normalize": {
            "enabled": True,
            "target_level": -2.0,
            "headroom": 2.0,
            "method": "peak"
        },
        "dynamic_processing": {
            "enabled": True,
            "compressor": {
                "threshold": -24.0,
                "ratio": 2.0,
                "attack": 40.0,
                "release": 400.0
            },
            "limiter": {
                "threshold": -1.5,
                "release": 100.0
            }
        },
        "low_pass": True,
        "low_pass_freq": 8000,
        "low_pass_db": -12,
        "high_pass": False,
        "high_pass_freq": 60,
        "high_pass_db": -12,
        "pan": 0
    }
}

def get_factory_defaults():
    """Return a deep copy of all factory default settings."""
    import copy
    
    settings = {
        "profiles": copy.deepcopy(FACTORY_PROFILES),
        "segmentation": copy.deepcopy(FACTORY_SEGMENTATION),
        "export": copy.deepcopy(FACTORY_EXPORT),
        "visualization": copy.deepcopy(FACTORY_VISUALIZATION),
        "processing": copy.deepcopy(FACTORY_PROCESSING),
        "version": SETTINGS_VERSION
    }
    
    return settings 