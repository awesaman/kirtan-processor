# Help texts for Settings and Profiles in Kirtan Processor
# Edit these strings to update the tooltips/help popups in the UI.

HELP_TEXTS = {
    # === MAIN FUNCTION SECTIONS ===
    'gain_section': "Adjusts the overall volume of the track. Increasing gain makes the audio louder, while decreasing it makes it quieter.",
    'normalization_section': "Automatically adjusts the loudness of the track to a target level, ensuring consistent volume across all tracks.",
    'dynamics_section': "Controls the dynamic range of the audio using compression and limiting. Helps reduce loudness differences, making quiet parts louder and loud parts softer for a more balanced sound.",
    'eq_section': "Adjusts the balance of bass, mid, and treble frequencies. Use EQ to enhance clarity, reduce unwanted noise, or shape the overall tone of the audio.",

    # === PROCESS SETTINGS ===
    'process_settings': "Settings that control how audio files are processed, including speed, memory, and CPU usage.",
    # === PROCESSING PROFILE ===
    'processing_profile': "A collection of audio processing parameters (like gain, normalization, compression, EQ) that can be saved and reused."
,
    # === SETTINGS ===
    'silence_threshold': "How much quieter a section must be (in dB) than the average to be considered silence. Lower values = more sensitive.",
    'min_silence': "Minimum duration (ms) a section must be quiet to be considered silence.",
    'seek_step': "How often (ms) the algorithm checks for silence. Smaller steps = more precise, but slower.",
    'min_time_between_segments': "Minimum time (ms) required between detected segments. Prevents too many splits.",
    'min_segment_length': "Minimum segment length (minutes) allowed after splitting.",
    'dropout': "Dropout time (minutes): Segments shorter than this are ignored (treated as silence).",
    'pre_segment_padding': "Amount of audio (seconds) to add before each segment. Positive values add extra audio before the segment, negative values make the segment start earlier.",
    'post_segment_padding': "Extra audio (seconds) added after each segment.",
    'save_unsegmented': "Export the entire processed track as one file, in addition to segments.",
    'trim_only': "Trim silence only at the start and end. No splitting in the middle.",
    'batch_normalize': "Make all tracks in a batch the same loudness (volume).",
    'process_speed': "Choose between faster processing or lower memory usage.",
    'cache_size': "How much memory (MB) to use for caching audio data.",
    'multi_core': "Enable to use multiple CPU cores for faster processing.",
    'processor_limit': "Limit the number of CPU cores used.",
    'output_format': "Choose MP3 or WAV for exported files.",
    'mp3_bitrate': "Quality of exported MP3s. Higher = better, but larger files.",
    'show_all_operations': "Show detailed logs of all operations.",
    'confirm_overwrite': "Ask before overwriting existing files.",

    # === PROFILE FIELDS ===
    'gain': "Adjusts the overall volume of the track (in dB).",
    'normalize_enabled': "Enable normalization to make the track's loudness match a target level.",
    'normalize_method': "Choose how loudness is measured: RMS (average), LUFS (perceived), or Peak (max).",
    'normalize_target_level': "The target loudness level for normalization (in dB).",
    'normalize_headroom': "Extra space (in dB) below 0 dB to avoid clipping.",
    'use_compressor': "Enable compression to reduce loudness differences within the track.",
    'compression_threshold': "Level (dB) above which compression starts.",
    'compression_ratio': "How strongly the compressor reduces loud sounds.",
    'compression_attack': "How quickly the compressor responds to loud sounds (ms).",
    'compression_release': "How quickly the compressor stops after loud sounds end (ms).",
    'use_limiter': "Enable limiting to prevent peaks from exceeding a set level.",
    'limiter_threshold': "Maximum allowed peak level (dB).",
    'limiter_release': "How quickly the limiter stops after a peak (ms).",
    'use_eq': "Enable equalization to adjust bass/treble.",
    'eq_high_pass': "Removes frequencies below this (Hz) to reduce rumble.",
    'eq_low_pass': "Removes frequencies above this (Hz) to reduce hiss.",
    'description': "A short note about what this profile is for.",
}
