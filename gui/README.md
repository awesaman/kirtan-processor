# Kirtan Processor GUI Styling

This directory contains the styling files for the Kirtan Processor GUI application.

## Files

- `style.qss` - Qt Style Sheet with the application's visual theme
- `images/` - Contains images used by the style sheet:
  - `check.png` - Checkmark icon for checkboxes
  - `down-arrow.png` - Down arrow icon for comboboxes

## Styling Notes

The application uses a dark theme with blue accents, designed for audio processing workflows. The styling provides:

- Consistent dark theme throughout the application
- Clear visual hierarchy with color-coded UI elements
- Special styling for interactive elements (buttons, sliders, etc.)
- Custom styling for the process button with different states

## Using Custom Styling

The application automatically loads the `style.qss` file at startup. To customize the styling:

1. Edit the `style.qss` file to change colors, sizes, or other properties
2. Replace the images in the `images/` directory with your own if needed
3. Restart the application to see your changes

## Button States

The Process button has several states controlled by Qt properties:

- `primary="true"` - For the main action button
- `processing="true"` - When processing is active
- `highlighted="true"` - For the blinking effect during processing
- `success="true"` - When processing completes successfully

You can apply these properties to other buttons as needed.

## Dependencies

The styling requires PyQt5 or PySide2 to be installed. No additional dependencies are needed for the styling itself.

## Custom Widgets

The application includes a `StyledButton` class that inherits from `QPushButton` and applies the `primary` property when initialized with `primary=True`. 