#!/usr/bin/env python
"""
Convert an image file to .ico format.
Prompts the user to select an image file via a file explorer dialog.
Supported input formats: PNG, JPG, JPEG, BMP, GIF
"""
import os
from PIL import Image
import tkinter as tk
from tkinter import filedialog

SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')


def convert_to_ico(image_path):
    base, _ = os.path.splitext(image_path)
    ico_path = base + '.ico'
    try:
        img = Image.open(image_path)
        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        # Prepare icon sizes: include all standard sizes up to the image's max size
        sizes = [16, 24, 32, 48, 64, 128, 256]
        max_dim = min(img.width, img.height)
        available_sizes = [s for s in sizes if s <= max_dim]
        if not available_sizes:
            available_sizes = [min(max_dim, 256)]
        icon_images = [img.resize((s, s), Image.LANCZOS) for s in available_sizes]
        # Save all sizes in the .ico file for best compatibility
        icon_images[0].save(ico_path, format='ICO', sizes=[(s, s) for s in available_sizes])
        print(f"Converted {image_path} -> {ico_path} (sizes: {available_sizes})")
    except Exception as e:
        print(f"Failed to convert {image_path}: {e}")

def select_image_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an image file to convert to .ico",
        filetypes=[
            ("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"),
            ("All Files", "*.*")
        ]
    )
    root.destroy()
    return file_path

def main():
    print("Select an image file to convert to .ico...")
    image_path = select_image_file()
    if not image_path:
        print("No file selected. Exiting.")
        return
    if not image_path.lower().endswith(SUPPORTED_FORMATS):
        print("Selected file is not a supported image format.")
        return
    convert_to_ico(image_path)
    print("Done.")

if __name__ == "__main__":
    main()
