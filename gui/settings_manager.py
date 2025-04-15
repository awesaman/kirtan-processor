"""
Kirtan Processor - Settings Manager

This module provides a central manager for application settings. It handles:
1. Loading default settings
2. Loading saved settings from disk
3. Saving settings to disk
4. Providing access to settings throughout the application
"""

import os
import json
import copy
import logging
from PyQt5.QtCore import QSettings

# Import default settings
from .default_settings import get_default_settings, SETTINGS_VERSION

class SettingsManager:
    """Manager class for application settings."""
    
    def __init__(self):
        """Initialize the settings manager with default settings."""
        # Load default settings
        self.settings = get_default_settings()
        self.qt_settings = QSettings("KirtanProcessor", "App")
        self.settings_file_path = ""
        
    def load_settings(self, settings_path=None):
        """
        Load settings from a file. If no file is specified, attempt to load
        from the last used settings file or use defaults.
        
        Args:
            settings_path (str, optional): Path to settings JSON file.
                
        Returns:
            bool: True if settings were loaded successfully, False otherwise.
        """
        # If no path provided, try to get the last used settings file
        if settings_path is None:
            settings_path = self.qt_settings.value("last_settings_file", "")
        
        # If we still don't have a path, use defaults
        if not settings_path or not os.path.exists(settings_path):
            logging.info("No settings file found. Using defaults.")
            return False
            
        try:
            with open(settings_path, 'r') as f:
                loaded_settings = json.load(f)
                
            # Check settings version
            if 'version' not in loaded_settings:
                logging.warning("Settings file has no version. Using defaults.")
                return False
                
            # Merge loaded settings with defaults to ensure all required settings exist
            self._merge_settings(loaded_settings)
            self.settings_file_path = settings_path
            self.qt_settings.setValue("last_settings_file", settings_path)
            logging.info(f"Settings loaded from {settings_path}")
            return True
        except Exception as e:
            logging.error(f"Error loading settings: {e}")
            return False
    
    def save_settings(self, settings_path=None):
        """
        Save current settings to a file.
        
        Args:
            settings_path (str, optional): Path to save settings JSON file.
                If not specified, use the last path or a default.
                
        Returns:
            bool: True if settings were saved successfully, False otherwise.
        """
        if not settings_path:
            settings_path = self.settings_file_path
            
        if not settings_path:
            # No path specified, and no previous path. Use default location
            app_data_dir = os.path.join(os.path.expanduser("~"), ".kirtan_processor")
            os.makedirs(app_data_dir, exist_ok=True)
            settings_path = os.path.join(app_data_dir, "settings.json")
        
        try:
            # Ensure settings directory exists
            os.makedirs(os.path.dirname(os.path.abspath(settings_path)), exist_ok=True)
            
            with open(settings_path, 'w') as f:
                json.dump(self.settings, f, indent=2)
                
            self.settings_file_path = settings_path
            self.qt_settings.setValue("last_settings_file", settings_path)
            logging.info(f"Settings saved to {settings_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving settings: {e}")
            return False
    
    def get_settings(self):
        """Get a deep copy of the current settings."""
        return copy.deepcopy(self.settings)
    
    def update_settings(self, settings_dict):
        """
        Update settings with the provided dictionary.
        
        Args:
            settings_dict (dict): Dictionary with settings to update.
        """
        self._merge_settings(settings_dict)
        
    def reset_to_defaults(self):
        """Reset all settings to default values."""
        self.settings = get_default_settings()
    
    def get_profile(self, profile_name):
        """
        Get a specific profile by name.
        
        Args:
            profile_name (str): Name of the profile to retrieve.
            
        Returns:
            dict: The profile settings or None if not found.
        """
        profiles = self.settings.get("profiles", {})
        return copy.deepcopy(profiles.get(profile_name))
    
    def update_profile(self, profile_name, profile_settings):
        """
        Update a specific profile.
        
        Args:
            profile_name (str): Name of the profile to update.
            profile_settings (dict): New profile settings.
        """
        if "profiles" not in self.settings:
            self.settings["profiles"] = {}
            
        self.settings["profiles"][profile_name] = profile_settings
    
    def create_profile(self, profile_name, base_profile=None):
        """
        Create a new profile, optionally based on an existing one.
        
        Args:
            profile_name (str): Name for the new profile.
            base_profile (str, optional): Name of profile to use as a base.
            
        Returns:
            bool: True if profile was created, False if name already exists.
        """
        if profile_name in self.settings.get("profiles", {}):
            return False
            
        if "profiles" not in self.settings:
            self.settings["profiles"] = {}
            
        if base_profile and base_profile in self.settings["profiles"]:
            # Copy the base profile
            self.settings["profiles"][profile_name] = copy.deepcopy(
                self.settings["profiles"][base_profile]
            )
        else:
            # Create a profile with default settings
            from .default_settings import DEFAULT_PROFILES
            self.settings["profiles"][profile_name] = copy.deepcopy(
                next(iter(DEFAULT_PROFILES.values()))
            )
            
        return True
    
    def delete_profile(self, profile_name):
        """
        Delete a profile.
        
        Args:
            profile_name (str): Name of the profile to delete.
            
        Returns:
            bool: True if profile was deleted, False if not found.
        """
        if profile_name in self.settings.get("profiles", {}):
            del self.settings["profiles"][profile_name]
            return True
        return False
        
    def _merge_settings(self, new_settings):
        """
        Merge new settings with existing settings, preserving defaults for missing values.
        
        Args:
            new_settings (dict): New settings to merge.
        """
        # Helper function for recursive merging
        def _recursive_merge(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    _recursive_merge(target[key], value)
                else:
                    target[key] = value
        
        # Make a copy to avoid modifying the input
        new_settings_copy = copy.deepcopy(new_settings)
        
        # Merge at top level
        for section in self.settings:
            if section in new_settings_copy and isinstance(new_settings_copy[section], dict):
                _recursive_merge(self.settings[section], new_settings_copy[section])
            elif section in new_settings_copy:
                self.settings[section] = new_settings_copy[section]
                
        # Set the version to current
        self.settings["version"] = SETTINGS_VERSION 