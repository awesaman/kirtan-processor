import os
import gc

class AudioCache:
    """Cache for audio files to reduce memory usage and disk reads"""
    def __init__(self, max_size=3):
        self.cache = {}
        self.access_count = {}
        self.max_size = max_size
        
    def get(self, key):
        """Get an item from the cache"""
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        return None
        
    def put(self, key, value):
        """Add an item to the cache, evicting least used if needed"""
        # If cache is full, remove least accessed item
        if len(self.cache) >= self.max_size:
            least_used = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[least_used]
            del self.access_count[least_used]
            gc.collect()
            
        # Add new item
        self.cache[key] = value
        self.access_count[key] = 1
        
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.access_count.clear()
        gc.collect()