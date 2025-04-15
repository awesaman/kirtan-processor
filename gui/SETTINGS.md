# Centralized Settings System for Kirtan Processor

This document explains the centralized settings system implemented for the Kirtan Processor application.

## Overview

The centralized settings system provides a single source of truth for all application defaults and user settings. This approach offers several benefits:

1. **Single Source of Truth**: All default settings are defined in one place, making it easier to maintain and update.
2. **Consistency**: By using a centralized approach, you ensure that defaults are consistently applied throughout the application.
3. **Easier Updates**: When you need to modify a default setting, you only need to change it in one place.
4. **Better Code Organization**: The settings structure is clearly defined and easier to understand.
5. **Simplified Loading/Saving**: Loading and saving settings becomes more straightforward since everything follows the same structure.

## Structure

The centralized settings system consists of three main components:

1. **default_settings.py**: Contains all default values for application settings.
2. **settings_manager.py**: Provides a manager class to handle loading, saving, and accessing settings.
3. **integration_example.py**: Demonstrates how to integrate the system with the main application.

## Usage

### Loading Settings

```python
# Initialize settings manager with defaults
self.settings_manager = SettingsManager()

# Load settings from file (or use defaults if no file exists)
self.settings_manager.load_settings()

# Get all settings as a dictionary
app_settings = self.settings_manager.get_settings()

# Access specific settings
self.profiles = app_settings["profiles"]
self.silence_threshold = app_settings["segmentation"]["silence_threshold"]
self.export_bitrate = app_settings["export"]["bitrate"]
```

### Saving Settings

```python
# Build settings dictionary from current application state
settings = {
    "profiles": self.profiles,
    "segmentation": {
        "silence_threshold": self.silence_threshold,
        # Other segmentation settings...
    },
    # Other setting categories...
}

# Update settings manager
self.settings_manager.update_settings(settings)

# Save to disk
self.settings_manager.save_settings()
```

### Working with Profiles

```python
# Get a specific profile
vocal_profile = self.settings_manager.get_profile("Kirtan (Vocals)")

# Update a profile
vocal_profile["normalize"]["method"] = "lufs"
self.settings_manager.update_profile("Kirtan (Vocals)", vocal_profile)

# Create a new profile
self.settings_manager.create_profile("New Profile", "Kirtan (Vocals)")  # Second parameter is the base profile

# Delete a profile
self.settings_manager.delete_profile("Old Profile")
```

### Resetting to Defaults

```python
# Reset all settings to defaults
self.settings_manager.reset_to_defaults()
```

## Settings Structure

The settings are organized into the following categories:

1. **profiles**: Audio processing profiles for different track types
2. **segmentation**: Settings related to track segmentation and silence detection
3. **export**: Settings for audio export (format, quality, etc.)
4. **visualization**: Settings for waveform display and other visual elements
5. **processing**: Settings related to audio processing behavior
6. **ui**: User interface preferences and history
7. **version**: Settings file version for compatibility

## Settings Storage

Settings are stored in JSON format at a user-specific location:

- Windows: `C:\Users\<username>\.kirtan_processor\settings.json`
- macOS: `/Users/<username>/.kirtan_processor/settings.json`
- Linux: `/home/<username>/.kirtan_processor/settings.json`

The location can be overridden by specifying a custom path when loading or saving settings.

## Implementation Details

### Using Deep Copies

The settings manager always provides deep copies of settings to prevent accidental modifications. When you get settings from the manager, you receive a copy that you can modify freely without affecting the original.

### Merging Settings

When loading settings from a file, the manager merges them with the defaults to ensure all required settings exist. This approach allows backward compatibility when new settings are added to the application.

### Error Handling

The settings manager includes robust error handling to prevent crashes when loading corrupt or incompatible settings files. If a settings file cannot be loaded, the application falls back to default settings. 