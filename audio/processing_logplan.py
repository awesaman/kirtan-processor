# processing_logplan.py
"""
Defines a plan of processing steps for logging in standard mode.
"""
PROCESSING_STEPS = [
    ("Normalization", "normalize"),
    ("Gain", "gain"),
    ("Compressor", "compressor"),
    ("Limiter", "limiter"),
    ("High-pass Filter", "high_pass"),
    ("Low-pass Filter", "low_pass"),
    ("EQ Attenuation", "eq_attenuate"),
]
