#!/usr/bin/env python
"""
Unified Windows build script for Kirtan Processor
- Creates icon if needed
- Performs clean or normal build
- Includes all required resources
- Usage:
    python build_windows.py [--clean] [--icon-only]
"""
import os
import sys
import subprocess
from PIL import Image, ImageDraw
import shutil

def create_icon(icon_path):
    """Create a simple icon file if one doesn't exist."""
    if os.path.exists(icon_path):
        print(f"Using existing icon: {icon_path}")
        return icon_path
    try:
        img = Image.new('RGB', (256, 256), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.ellipse((50, 50, 206, 206), fill=(65, 105, 225))
        img.save(icon_path, format='ICO')
        print(f"Created icon at: {os.path.abspath(icon_path)}")
        return icon_path
    except Exception as e:
        print(f"Error creating icon: {e}")
        return None

def build_executable(clean=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, '..'))
    spec_file = os.path.join(script_dir, 'KirtanProcessor.spec')
    icon_path = os.path.join(root_dir, 'app_icon.ico')
    style_path = os.path.join(root_dir, 'style.qss')
    images_dir = os.path.join(root_dir, 'images')
    default_json = os.path.join(root_dir, 'default.json')
    main_script = os.path.join(root_dir, 'Kirtan-Processor-gui.py')

    print(f"Script directory: {script_dir}")
    print(f"Root directory: {root_dir}")
    print(f"Icon path: {icon_path}")
    print(f"Style file: {style_path}")
    print(f"Default JSON: {default_json}")
    print(f"Images directory: {images_dir}")
    print(f"Main script: {main_script}")

    # Clean build: remove spec file and dist/build folders
    if clean:
        if os.path.exists(spec_file):
            print(f"Removing existing spec file: {spec_file}")
            os.remove(spec_file)
        for folder in [os.path.join(root_dir, 'dist'), os.path.join(root_dir, 'build')]:
            if os.path.exists(folder):
                print(f"Removing folder: {folder}")
                shutil.rmtree(folder)
        # Also clean up build/dist in build-tools
        for folder in [os.path.join(script_dir, 'dist'), os.path.join(script_dir, 'build')]:
            if os.path.exists(folder):
                print(f"Removing folder from build-tools: {folder}")
                shutil.rmtree(folder)

    # Check required files
    if not os.path.exists(main_script):
        print(f"ERROR: Main script not found at: {main_script}")
        return False
    if not os.path.exists(style_path):
        print(f"ERROR: Style file not found at: {style_path}")
        return False
    if not os.path.exists(icon_path):
        print(f"WARNING: Icon file not found at: {icon_path}")
        icon_path = None
    if not os.path.exists(default_json):
        print(f"WARNING: Default settings file not found at: {default_json}")
    if not os.path.exists(images_dir):
        print(f"WARNING: Images directory not found at: {images_dir}")

    # Build PyInstaller command
    cmd = [
        'pyinstaller',
        '--onefile',
        '--noconsole',
        '--name=KirtanProcessor',
        f'--add-data={style_path};.',
    ]
    if os.path.exists(default_json):
        cmd.append(f'--add-data={default_json};.')
    if os.path.exists(images_dir):
        cmd.append(f'--add-data={images_dir};images')
    if icon_path and os.path.exists(icon_path):
        cmd.append(f'--icon={icon_path}')
    cmd.append(main_script)

    print("Running PyInstaller with command:")
    print(' '.join(cmd))
    # Change working directory to root before running PyInstaller
    old_cwd = os.getcwd()
    os.chdir(root_dir)
    try:
        result = subprocess.run(cmd)
    finally:
        os.chdir(old_cwd)
    exe_name = 'KirtanProcessor.exe'
    dist_exe_path = os.path.join(root_dir, 'dist', exe_name)
    root_exe_path = os.path.join(root_dir, exe_name)
    if result.returncode == 0:
        print("\nSuccess! Executable created in dist folder.")
        if os.path.exists(dist_exe_path):
            # Move the .exe to the root folder
            print(f"Moving {dist_exe_path} to {root_exe_path}")
            if os.path.exists(root_exe_path):
                os.remove(root_exe_path)
            shutil.move(dist_exe_path, root_exe_path)
            print(f"Executable location: {root_exe_path}")
        else:
            print(f"ERROR: Expected .exe not found at {dist_exe_path}")
        # Optionally clean up dist folder
        if os.path.exists(os.path.join(root_dir, 'dist')):
            try:
                shutil.rmtree(os.path.join(root_dir, 'dist'))
            except Exception as e:
                print(f"Warning: Could not remove dist folder: {e}")
        # Clean up build/dist in build-tools after build
        for folder in [os.path.join(script_dir, 'dist'), os.path.join(script_dir, 'build')]:
            if os.path.exists(folder):
                print(f"Removing leftover folder from build-tools: {folder}")
                shutil.rmtree(folder)
        return True
    else:
        print("\nError creating executable. Please check the output above.")
        return False

def check_exe_contents(exe_path):
    """Check if required files are included in the executable"""
    import tempfile, shutil
    if not os.path.exists(exe_path):
        print(f"Error: Executable not found at {exe_path}")
        return False
    print(f"Checking executable: {exe_path}")
    temp_dir = tempfile.mkdtemp()
    try:
        try:
            import PyInstaller
            pyinstaller_dir = os.path.dirname(PyInstaller.__file__)
            archive_viewer = os.path.join(pyinstaller_dir, 'archive_viewer.py')
            if os.path.exists(archive_viewer):
                print(f"Using archive_viewer at: {archive_viewer}")
                cmd = [sys.executable, archive_viewer, exe_path]
                print(f"Running command: {' '.join(cmd)}")
                commands_file = os.path.join(temp_dir, 'commands.txt')
                with open(commands_file, 'w') as f:
                    f.write(f"X {temp_dir}\nq\n")
                with open(commands_file, 'r') as f:
                    result = subprocess.run(cmd, stdin=f, text=True, capture_output=True)
                if result.returncode != 0:
                    print("Error extracting files:")
                    print(result.stderr)
                else:
                    style_found = False
                    default_found = False
                    print("Searching for style.qss and default.json in extracted files...")
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if file == 'style.qss':
                                style_found = True
                                print(f"Found style.qss at: {os.path.join(root, file)}")
                            if file == 'default.json':
                                default_found = True
                                print(f"Found default.json at: {os.path.join(root, file)}")
                    if not style_found:
                        print("style.qss not found in the executable!")
                    if not default_found:
                        print("default.json not found in the executable!")
                    return style_found and default_found
            else:
                print(f"Error: Archive viewer not found at {archive_viewer}")
        except ImportError:
            print("PyInstaller not installed. Cannot check executable contents.")
        print("Trying to run executable with debug options...")
        try:
            env = os.environ.copy()
            env['PYTHONVERBOSE'] = '1'
            result = subprocess.run([exe_path, "--version"], env=env, timeout=2, capture_output=True, text=True)
            print("Executable output:")
            print(result.stdout)
        except subprocess.TimeoutExpired:
            print("Executable timed out, this is expected.")
        except Exception as e:
            print(f"Error running executable: {e}")
    finally:
        shutil.rmtree(temp_dir)
    return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unified Windows build script for Kirtan Processor")
    parser.add_argument('--clean', action='store_true', help='Perform a clean rebuild (remove old build/dist/spec)')
    parser.add_argument('--icon-only', action='store_true', help='Only create the icon, do not build executable')
    parser.add_argument('--check-inclusion', action='store_true', help='Check if style.qss and default.json are included in the built executable')
    parser.add_argument('--exe-path', type=str, default=None, help='Path to the executable to check (for --check-inclusion)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, '..'))
    icon_path = os.path.join(root_dir, 'app_icon.ico')

    if args.icon_only:
        create_icon(icon_path)
        print("Icon creation complete.")
        return

    if args.check_inclusion:
        exe_path = args.exe_path or os.path.join(root_dir, 'dist', 'KirtanProcessor.exe')
        check_exe_contents(exe_path)
        print("\nDone!")
        return

    # Always create icon before build
    create_icon(icon_path)
    build_executable(clean=args.clean)

if __name__ == "__main__":
    main()
