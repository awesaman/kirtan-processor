1. Audio File Selection & Normalization
User can select an audio file via a file dialog.
The selected file is displayed in the GUI.
Audio is loaded using pydub and normalized to a target dBFS (e.g., -3.0 dBFS) before analysis.
2. Configurable VAD Parameters (All via GUI)
Frame size (ms): Adjustable (10–30 ms).
Aggressiveness: Adjustable (0–3, float).
Min voice segment (ms): Adjustable (50–5000 ms).
Scan window (seconds): Adjustable (10–600 s), used for both start and end analysis.
Vocal dB threshold (dBFS): Adjustable (-60 to 0 dBFS).
Consecutive frames above threshold for onset: Adjustable (1–10).
dB jump for vocal onset (dB): Adjustable (1–20 dB).
Start buffer (sec): Adjustable (-60 to 60 s, negative shifts earlier).
End buffer (sec): Adjustable (-60 to 60 s, positive shifts later).
3. VAD Analysis & Segment Detection
Runs VAD on the first and last scan windows of the audio.
Detects and logs all voice segments in each window.
Calculates and logs average dBFS for each segment.
Provides debug info for each segment (frame-by-frame dBFS).
4. Vocal Onset Detection (User-Selectable Method)
Largest dBFS jump: Finds the segment with the largest increase in average dBFS compared to previous segments.
dBFS threshold: Finds the first segment above a user-defined dBFS threshold.
Both methods are available; user can select via GUI (checkboxes, mutually exclusive).
Context logging: Shows 5 segments before/after the selected onset for transparency.
5. Vocal Offset Detection
Analyzes the last scan window for the end of vocals.
Three methods:
Significant dB drop: Detects a large drop in dBFS between segments.
Consecutive frames below threshold: Fallback if no dB drop is found.
Last segment above threshold: Last resort fallback.
Logs which method was used and the relevant segment/frame.
6. Buffer Handling
Applies user-configurable start and end buffers to detected segment boundaries.
If the start is at or before 00:00, the start buffer is ignored and a note is logged.
7. Segment Filtering
Segments shorter than the minimum voice segment length are filtered out from candidates.
Only segments longer than this threshold are considered for onset/offset.
8. Detailed Logging
All detected segments, their times (mm:ss), and average dBFS are logged.
Context blocks for onset/offset selection.
Warnings for edge cases (e.g., if the first segment starts at 00:00).
All relevant settings are logged for reproducibility/debugging.
9. GUI Features
Tabbed log output (Start/End).
Expand/collapse for segment debug info.
Clear log button.
Plotting of segments on waveform (if matplotlib/numpy available).
10. Segment Export (if implemented)
Segments can be exported with the same buffer logic as display.
Exported segments include metadata (start/end, duration, etc.).
Summary:
The tool is a robust, user-configurable VAD segment analyzer with transparent logging, multiple onset/offset detection strategies, and a modern GUI for parameter tuning and debugging. All parameters are settable via the GUI, and the log is designed for both transparency and reproducibility.