# filepath: c:\git\kirtan-processor\utils\app_logging.py

"""Logging utilities for Kirtan Processor"""
import time

def log_section_header(log_func, title):
    """Add a clear visual section header to the log"""
    # Format with dashes for visibility in log
    header = "--------------------------------------------------"
    log_func(header)
    log_func(title)
    log_func(header)

def log_chunk_processing_summary(log_func, track_name, chunk_count, before_level, after_level):
    """Log a summary of chunk processing results"""
    summary = f"⚙️ {track_name} processed in {chunk_count} chunks"
    summary += f"\nAverage level: {before_level:.1f}dB → {after_level:.1f}dB"
    log_func(summary)

def format_time(ms):
    """Format milliseconds to readable time as mm:ss or hh:mm:ss"""
    seconds = int(ms / 1000)
    minutes = seconds // 60
    seconds = seconds % 60
    
    if minutes > 59:
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"