#!/usr/bin/env python
"""
Script to create a macOS .app bundle and .dmg installer for Kirtan Processor.
This script should be run on a macOS system.

Requirements:
- PyInstaller (`pip install pyinstaller`)
- create-dmg (`brew install create-dmg`)
"""

import os
import sys
import subprocess
import platform
from PIL import Image
import shutil

# Get the root directory (one level up from the script location)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def check_platform():
    """Check if the script is running on macOS."""
    if platform.system() != 'Darwin':
        print("Warning: This script is designed to be run on macOS.")
        print("You are currently running on:", platform.system())
        print("The script will continue, but it may not work as expected.")
        return False
    return True

def create_icns_icon():
    """Create an .icns icon file for macOS applications."""
    print("Creating .icns icon file...")
    
    # Source icon path
    icon_path = os.path.join(ROOT_DIR, 'app_icon.ico')
    icns_path = os.path.join(ROOT_DIR, 'app_icon.icns')
    
    if os.path.exists(icns_path):
        print(f"Using existing icon: {icns_path}")
        return icns_path
    
    # Check if the .ico file exists, if not generate a simple icon
    if not os.path.exists(icon_path):
        print("Could not find an icon. Will use default PyInstaller icon.")
        return None
    
    try:
        # Convert .ico to .icns using PIL
        img = Image.open(icon_path)
        
        # Create iconset directory
        iconset_dir = os.path.join(os.path.dirname(icon_path), 'AppIcon.iconset')
        if not os.path.exists(iconset_dir):
            os.makedirs(iconset_dir)
            
        # Generate different sizes required for .icns
        for size in [16, 32, 64, 128, 256, 512, 1024]:
            resized = img.resize((size, size))
            resized.save(os.path.join(iconset_dir, f'icon_{size}x{size}.png'))
            # Also create 2x versions for retina displays
            if size <= 512:
                double_size = size * 2
                resized = img.resize((double_size, double_size))
                resized.save(os.path.join(iconset_dir, f'icon_{size}x{size}@2x.png'))
        
        # Use iconutil to create .icns file (macOS only)
        if platform.system() == 'Darwin':
            subprocess.run(['iconutil', '-c', 'icns', iconset_dir, '-o', icns_path], check=True)
            print(f"Created icon at: {os.path.abspath(icns_path)}")
            return icns_path
        else:
            print("Warning: iconutil is only available on macOS. Cannot create .icns file.")
            return None
    except Exception as e:
        print(f"Error creating .icns icon: {e}")
        return None

def create_app_bundle():
    """Create a .app bundle using PyInstaller."""
    print("Creating .app bundle for Kirtan-Processor-gui.py...")
    
    # Get the icon path
    icon_path = create_icns_icon()
    
    # Base PyInstaller command
    cmd = [
        'pyinstaller',
        '--onefile',
        '--windowed',
        '--name=KirtanProcessor'
    ]
    
    # Add icon if available
    if icon_path and os.path.exists(icon_path):
        cmd.append(f'--icon={icon_path}')
    
    # Add data files
    images_dir = os.path.join(ROOT_DIR, 'images')
    style_file = os.path.join(ROOT_DIR, 'style.qss')
    
    if os.path.exists(images_dir):
        if platform.system() == 'Darwin':
            cmd.append(f'--add-data={images_dir}:images')
        else:
            cmd.append(f'--add-data={images_dir};images')
            
    if os.path.exists(style_file):
        if platform.system() == 'Darwin':
            cmd.append(f'--add-data={style_file}:.')
        else:
            cmd.append(f'--add-data={style_file};.')
        
    # Add main script
    cmd.append(os.path.join(ROOT_DIR, 'Kirtan-Processor-gui.py'))
    
    # Run PyInstaller
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    app_name = 'KirtanProcessor.app'
    dist_app_path = os.path.join('dist', app_name)
    root_app_path = os.path.join(ROOT_DIR, app_name)
    
    if result.returncode == 0:
        print("Success! App bundle created in dist folder.")
        if os.path.exists(dist_app_path):
            print(f"Moving {dist_app_path} to {root_app_path}")
            if os.path.exists(root_app_path):
                shutil.rmtree(root_app_path)
            shutil.move(dist_app_path, root_app_path)
            print(f"App bundle location: {root_app_path}")
            # Optionally clean up dist folder
            if os.path.exists('dist'):
                try:
                    shutil.rmtree('dist')
                except Exception as e:
                    print(f"Warning: Could not remove dist folder: {e}")
            return root_app_path
        else:
            print(f"ERROR: Expected .app not found at {dist_app_path}")
    else:
        print("Error creating app bundle:")
        print(result.stderr)
    return None

def create_dmg_installer(app_path):
    """Create a .dmg installer using create-dmg."""
    if not app_path:
        print("No .app bundle found. Cannot create .dmg installer.")
        return
    
    print("Creating .dmg installer...")
    
    # Check if create-dmg is installed
    if platform.system() == 'Darwin':
        try:
            subprocess.run(['which', 'create-dmg'], check=True, capture_output=True)
        except:
            print("create-dmg not found. Please install it using 'brew install create-dmg'")
            return
    else:
        print("Warning: create-dmg is only available on macOS. Cannot create .dmg file.")
        print("When running on macOS, install create-dmg with: brew install create-dmg")
        return
    
    # Create dmg
    dmg_name = 'KirtanProcessor.dmg'
    dist_dmg_path = os.path.join('dist', dmg_name)
    root_dmg_path = os.path.join(ROOT_DIR, dmg_name)
    
    cmd = [
        'create-dmg',
        '--volname', 'Kirtan Processor',
        '--window-pos', '200', '120',
        '--window-size', '600', '300',
        '--icon', 'KirtanProcessor.app', '150', '150',
        '--app-drop-link', '450', '150',
        dist_dmg_path,
        app_path
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Success! DMG installer created at: {os.path.abspath(dist_dmg_path)}")
        # Move DMG to root
        if os.path.exists(dist_dmg_path):
            print(f"Moving {dist_dmg_path} to {root_dmg_path}")
            if os.path.exists(root_dmg_path):
                os.remove(root_dmg_path)
            shutil.move(dist_dmg_path, root_dmg_path)
            print(f"DMG location: {root_dmg_path}")
        # Optionally clean up dist folder
        if os.path.exists('dist'):
            try:
                shutil.rmtree('dist')
            except Exception as e:
                print(f"Warning: Could not remove dist folder: {e}")
    else:
        print("Error creating DMG installer:")
        print(result.stderr)

def main():
    """Main function."""
    print("==================================")
    print(" Kirtan Processor macOS App Creator")
    print("==================================")
    print()
    
    # Check if running on macOS
    is_macos = check_platform()
    
    # Change to the script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create .app bundle
    app_path = create_app_bundle()
    
    # Create .dmg installer if on macOS
    if is_macos and app_path:
        create_dmg_installer(app_path)
    elif not is_macos:
        print("\nNote: To create a proper .app bundle and .dmg installer, run this script on macOS.")
        print("Current build is for demonstration purposes only and may not work on macOS.")
    
    print("\nDone!")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
