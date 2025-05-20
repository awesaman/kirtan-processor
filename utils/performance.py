import time
import psutil
import os

class PerformanceTracker:
    """Track processing performance metrics"""
    def __init__(self, log_callback=print):
        self.start_time = 0
        self.start_memory = 0
        self.log = log_callback
        
    def start_tracking(self):
        """Start tracking performance"""
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()
        
    def log_summary(self):
        """Log performance summary"""
        elapsed = time.time() - self.start_time
        current_memory = self._get_memory_usage()
        memory_increase = current_memory - self.start_memory
        
        self.log("===== Performance Summary =====")
        self.log(f"Total processing time: {elapsed:.2f}s")
        self.log(f"Current memory usage: {self._format_bytes(current_memory)}")
        self.log(f"Memory increase: {self._format_bytes(memory_increase)}")
        self.log(f"Peak memory usage: {self._format_bytes(self._get_peak_memory())}")
        self.log("================================")
        
    def get_cpu_percent(self):
        """Get current CPU usage as a percentage"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0
    
    def get_memory_percent(self):
        """Get current memory usage as a percentage"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_percent()
        except:
            return 0.0
        
    def _get_memory_usage(self):
        """Get current memory usage"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except:
            return 0
            
    def _get_peak_memory(self):
        """Get peak memory usage"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().peak_wset
        except:
            return self._get_memory_usage()
            
    def _format_bytes(self, bytes):
        """Format bytes as human-readable string"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes < 1024:
                return f"{bytes:.1f}{unit}"
            bytes /= 1024
        return f"{bytes:.1f}TB"