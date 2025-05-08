#!/usr/bin/env python
"""
Command-line tool for managing singer voice profiles and identification.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.voice_recognition import VoiceProfiler, build_singer_database, identify_singers_in_file

def create_profile(args):
    """Create a singer profile from audio files"""
    profiler = VoiceProfiler()
    
    # Get audio files
    audio_files = []
    if os.path.isdir(args.input):
        # If input is a directory, include all audio files
        for root, _, files in os.walk(args.input):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.flac')):
                    audio_files.append(os.path.join(root, file))
    else:
        # If input is a file, use just that file
        if args.input.lower().endswith(('.wav', '.mp3', '.flac')):
            audio_files.append(args.input)
    
    if not audio_files:
        print(f"No audio files found in {args.input}")
        return
    
    print(f"Creating profile for '{args.name}' with {len(audio_files)} audio files...")
    success = profiler.create_voice_profile(
        args.name,
        audio_files,
        metadata={"source": args.input}
    )
    
    if success:
        print(f"✓ Successfully created profile for '{args.name}'")
    else:
        print(f"✗ Failed to create profile for '{args.name}'")

def list_profiles(args):
    """List all available profiles"""
    profiler = VoiceProfiler()
    profiles = profiler.get_profile_names()
    
    if not profiles:
        print("No profiles available. Create some profiles first.")
        return
    
    print(f"Found {len(profiles)} singer profiles:")
    for i, name in enumerate(profiles, 1):
        profile = profiler.get_profile(name)
        sample_count = profile.get('sample_count', 0)
        print(f"{i}. {name} ({sample_count} samples)")

def delete_profile(args):
    """Delete a singer profile"""
    profiler = VoiceProfiler()
    success = profiler.delete_profile(args.name)
    
    if success:
        print(f"✓ Successfully deleted profile for '{args.name}'")
    else:
        print(f"✗ Failed to delete profile for '{args.name}' (profile may not exist)")

def rename_profile(args):
    """Rename a singer profile"""
    profiler = VoiceProfiler()
    success = profiler.rename_profile(args.old_name, args.new_name)
    
    if success:
        print(f"✓ Successfully renamed profile from '{args.old_name}' to '{args.new_name}'")
    else:
        print(f"✗ Failed to rename profile")

def build_database(args):
    """Build a database of singer profiles from a folder structure"""
    if not os.path.isdir(args.folder):
        print(f"Error: {args.folder} is not a valid directory")
        return
    
    print(f"Building singer database from {args.folder}...")
    profiler = build_singer_database(args.folder)
    
    profiles = profiler.get_profile_names()
    print(f"✓ Successfully built database with {len(profiles)} singers")
    
    if profiles:
        print("Singers in database:")
        for i, name in enumerate(profiles, 1):
            print(f"{i}. {name}")

def identify_singer(args):
    """Identify the singer in an audio file"""
    profiler = VoiceProfiler()
    
    if not profiler.profiles:
        print("No singer profiles available. Create some profiles first.")
        return
    
    print(f"Analyzing '{args.audio}'...")
    matches = profiler.identify_singer(
        args.audio,
        top_n=args.top,
        threshold=args.threshold
    )
    
    if not matches:
        print("No singer matched with sufficient confidence.")
        return
    
    print("\nIdentified singers:")
    for i, (name, confidence) in enumerate(matches, 1):
        print(f"{i}. {name} - {confidence:.2%} confidence")

def analyze_file(args):
    """Analyze an audio file to identify singers in different segments"""
    print(f"Analyzing segments in '{args.audio}'...")
    results = identify_singers_in_file(
        args.audio,
        segment_duration=args.segment_duration
    )
    
    if not results:
        print("No singers identified in any segment.")
        return
    
    # Group results by singer
    singers = {}
    for result in results:
        singer = result['singer']
        if singer not in singers:
            singers[singer] = []
        singers[singer].append(result)
    
    print("\nIdentified singers by segment:")
    for singer, segments in singers.items():
        avg_confidence = sum(s['confidence'] for s in segments) / len(segments)
        total_duration = sum(s['end_ms'] - s['start_ms'] for s in segments) / 1000.0  # seconds
        
        print(f"\n{singer} - {avg_confidence:.2%} avg. confidence, {total_duration:.1f}s total duration")
        print("Segments:")
        for seg in segments:
            print(f"  {seg['start_time']} to {seg['end_time']} ({seg['confidence']:.2%})")

def main():
    parser = argparse.ArgumentParser(
        description="Singer Voice Recognition Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a profile for a singer
  python singer_recognition_cli.py create "Singer Name" path/to/audio/files/
  
  # List all profiles
  python singer_recognition_cli.py list
  
  # Identify the singer in an audio file
  python singer_recognition_cli.py identify path/to/audio.mp3
  
  # Build a database from a folder structure (one subfolder per singer)
  python singer_recognition_cli.py build path/to/singers/
  
  # Analyze a recording for different singers by segment
  python singer_recognition_cli.py analyze path/to/recording.mp3
"""
    )
    
    subparsers = parser.add_subparsers(title="commands", dest="command")
    
    # Create profile command
    create_parser = subparsers.add_parser("create", help="Create a singer profile")
    create_parser.add_argument("name", help="Name of the singer")
    create_parser.add_argument("input", help="Path to audio file or folder of audio files")
    create_parser.set_defaults(func=create_profile)
    
    # List profiles command
    list_parser = subparsers.add_parser("list", help="List all profiles")
    list_parser.set_defaults(func=list_profiles)
    
    # Delete profile command
    delete_parser = subparsers.add_parser("delete", help="Delete a singer profile")
    delete_parser.add_argument("name", help="Name of the singer to delete")
    delete_parser.set_defaults(func=delete_profile)
    
    # Rename profile command
    rename_parser = subparsers.add_parser("rename", help="Rename a singer profile")
    rename_parser.add_argument("old_name", help="Current name of the singer")
    rename_parser.add_argument("new_name", help="New name for the singer")
    rename_parser.set_defaults(func=rename_profile)
    
    # Build database command
    build_parser = subparsers.add_parser("build", help="Build a database from a folder structure")
    build_parser.add_argument("folder", help="Path to the base folder with singer subfolders")
    build_parser.set_defaults(func=build_database)
    
    # Identify singer command
    identify_parser = subparsers.add_parser("identify", help="Identify the singer in an audio file")
    identify_parser.add_argument("audio", help="Path to the audio file to analyze")
    identify_parser.add_argument("--top", type=int, default=3, help="Number of top matches to return")
    identify_parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold (0-1)")
    identify_parser.set_defaults(func=identify_singer)
    
    # Analyze file command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze an audio file for different singers by segment")
    analyze_parser.add_argument("audio", help="Path to the audio file to analyze")
    analyze_parser.add_argument("--segment-duration", type=int, default=30, help="Duration of each segment (seconds)")
    analyze_parser.set_defaults(func=analyze_file)
    
    args = parser.parse_args()
    
    # Create tools directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    
    try:
        if hasattr(args, 'func'):
            args.func(args)
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()