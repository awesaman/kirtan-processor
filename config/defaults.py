# Default processing profiles 
DEFAULT_PROFILES = {
    "Kirtan (Vocals)": {
        "gain": 0.0,
        "normalize_type": "RMS", 
        "target_level": -6.0,
        "use_compressor": True,
        "compression": {
            "threshold": -20.0,
            "ratio": 3.0,
            "attack": 5.0,
            "release": 50.0
        },
        "use_limiter": True,
        "limiter": {
            "threshold": -1.0,
            "release": 50.0
        },
        "use_eq": True,
        "eq": {
            "high_pass": 100,  # Hz
            "low_pass": 10000  # Hz
        },
        "description": "Lead vocals / kirtan - clear and present in the mix"
    },
    
    "Tabla": {
        "gain": 0.0,
        "normalize_type": "RMS", 
        "target_level": -12.0,
        "use_compressor": True,
        "compression": {
            "threshold": -24.0,
            "ratio": 4.0,
            "attack": 10.0,
            "release": 100.0
        },
        "use_limiter": True,
        "limiter": {
            "threshold": -1.0,
            "release": 50.0
        },
        "use_eq": True,
        "eq": {
            "high_pass": 80,  # Hz
            "low_pass": 8000  # Hz
        },
        "description": "Tabla - balanced percussion with controlled dynamics"
    },
    
    "Sangat (Harmonium)": {
        "gain": 0.0,
        "normalize_type": "RMS", 
        "target_level": -15.0,
        "use_compressor": True,
        "compression": {
            "threshold": -24.0,
            "ratio": 2.0,
            "attack": 20.0,
            "release": 200.0
        },
        "use_limiter": True,
        "limiter": {
            "threshold": -1.0,
            "release": 50.0
        },
        "use_eq": True,
        "eq": {
            "high_pass": 150,  # Hz
            "low_pass": 7000  # Hz
        },
        "description": "Harmonium - supportive with clear presence"
    },
    
    "Sangat (Tamboura)": {
        "gain": 0.0,
        "normalize_type": "RMS", 
        "target_level": -18.0,
        "use_compressor": False,
        "compression": {
            "threshold": -24.0,
            "ratio": 2.0,
            "attack": 50.0,
            "release": 500.0
        },
        "use_limiter": True,
        "limiter": {
            "threshold": -1.0,
            "release": 50.0
        },
        "use_eq": True,
        "eq": {
            "high_pass": 80,  # Hz
            "low_pass": 10000  # Hz
        },
        "description": "Tamboura - ambient drone with consistent level"
    },
    
    "Sangat (Ambient)": {
        "gain": 0.0,
        "normalize_type": "RMS", 
        "target_level": -20.0,
        "use_compressor": False,
        "use_limiter": True,
        "limiter": {
            "threshold": -1.0,
            "release": 50.0
        },
        "use_eq": True,
        "eq": {
            "high_pass": 100,  # Hz
            "low_pass": 10000  # Hz
        },
        "description": "Ambient tracks - room sound and audience at balanced level"
    },
    
    "Sangat (Other)": {
        "gain": 0.0,
        "normalize_type": "RMS", 
        "target_level": -15.0,
        "use_compressor": True,
        "compression": {
            "threshold": -24.0,
            "ratio": 2.5,
            "attack": 20.0,
            "release": 200.0
        },
        "use_limiter": True,
        "limiter": {
            "threshold": -1.0,
            "release": 50.0
        },
        "use_eq": True,
        "eq": {
            "high_pass": 100,  # Hz
            "low_pass": 9000  # Hz
        },
        "description": "Other instruments - balanced in the mix"
    }
}

# Default mix settings
DEFAULT_MIX_SETTINGS = {
    "pre_mix_scaling": True,  # Apply automatic scaling to prevent summing issues
    "final_normalize": True,  # Apply final normalization after mixing
    "final_limiter": True,    # Apply a final limiter to prevent clipping
    "final_target": -16.0,    # LUFS target for the final mix
    "final_peak": -1.0        # dB peak limit for the final mix
}