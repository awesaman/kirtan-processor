#!/usr/bin/env python
"""
Voice Recognition Module for Kirtan Processor.
This module provides functionality to create voice profiles for singers
and identify singers in new recordings.
"""

import os
import json
import pickle
import numpy as np
import librosa
from pydub import AudioSegment
from pathlib import Path
import warnings

# Suppress librosa's warning about PySoundFile
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")

# Default locations
DEFAULT_PROFILES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'voice_profiles')
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_PROFILES_DIR, 'voice_model.pkl')

# Ensure directories exist
os.makedirs(DEFAULT_PROFILES_DIR, exist_ok=True)

class VoiceProfiler:
    """Class for creating and managing voice profiles for singers"""
    
    def __init__(self, profiles_dir=None, model_path=None):
        """Initialize the voice profiler"""
        self.profiles_dir = profiles_dir or DEFAULT_PROFILES_DIR
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.profiles = {}
        self._load_profiles()
        
    def _load_profiles(self):
        """Load existing voice profiles from disk"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.profiles = pickle.load(f)
                print(f"Loaded {len(self.profiles)} voice profiles")
            else:
                print("No existing voice profile database found. Creating new.")
                self.profiles = {}
        except Exception as e:
            print(f"Error loading voice profiles: {str(e)}")
            self.profiles = {}
    
    def _save_profiles(self):
        """Save voice profiles to disk"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.profiles, f)
            print(f"Saved {len(self.profiles)} voice profiles")
            return True
        except Exception as e:
            print(f"Error saving voice profiles: {str(e)}")
            return False
    
    def extract_voice_features(self, audio_path, sample_duration=30):
        """
        Extract voice features from an audio file
        
        Args:
            audio_path: Path to the audio file
            sample_duration: Duration in seconds to analyze (from middle of file)
            
        Returns:
            Dictionary of voice features
        """
        try:
            # Use pydub to read audio file and convert to mono
            audio = AudioSegment.from_file(audio_path)
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Extract middle section for analysis if file is long enough
            if len(audio) > sample_duration * 1000:
                middle = len(audio) // 2
                half_sample = (sample_duration * 1000) // 2
                audio = audio[middle - half_sample:middle + half_sample]
            
            # Export to temporary file for librosa
            temp_file = os.path.join(os.path.dirname(audio_path), "temp_voice_analysis.wav")
            audio.export(temp_file, format="wav")
            
            # Load with librosa
            y, sr = librosa.load(temp_file, sr=None)
            
            # Extract features
            # MFCCs (Mel-frequency cepstral coefficients) - voice timbre characteristics
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # Chroma features - related to pitch class content
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            
            # Spectral contrast - difference between peaks and valleys in the spectrum
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            contrast_mean = np.mean(contrast, axis=1)
            
            # Pitch statistics using pyin algorithm for better pitch tracking
            # Handle potential errors in pitch estimation
            try:
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                pitch_mean = np.mean(pitches[pitches > 0])
                pitch_std = np.std(pitches[pitches > 0])
            except:
                pitch_mean = 0
                pitch_std = 0
            
            # Clean up temp file
            try:
                os.remove(temp_file)
            except:
                pass
            
            # Combine features into a profile
            profile = {
                'mfcc_mean': mfcc_mean.tolist(),
                'mfcc_std': mfcc_std.tolist(),
                'chroma_mean': chroma_mean.tolist(),
                'contrast_mean': contrast_mean.tolist(),
                'pitch_mean': float(pitch_mean),
                'pitch_std': float(pitch_std),
            }
            
            return profile
            
        except Exception as e:
            print(f"Error extracting voice features: {str(e)}")
            return None
    
    def create_voice_profile(self, singer_name, audio_files, metadata=None):
        """
        Create a voice profile for a singer from multiple audio files
        
        Args:
            singer_name: Name of the singer
            audio_files: List of paths to audio files containing the singer's voice
            metadata: Optional dictionary of additional information about the singer
            
        Returns:
            True if successful, False otherwise
        """
        if not audio_files:
            print(f"No audio files provided for {singer_name}")
            return False
            
        # Extract features from each file
        features_list = []
        for audio_path in audio_files:
            if os.path.exists(audio_path):
                features = self.extract_voice_features(audio_path)
                if features:
                    features_list.append(features)
            else:
                print(f"Audio file not found: {audio_path}")
        
        if not features_list:
            print(f"Could not extract features from any files for {singer_name}")
            return False
            
        # Average features across all samples
        profile = {
            'name': singer_name,
            'sample_count': len(features_list),
            'metadata': metadata or {},
            'features': self._average_features(features_list)
        }
        
        # Add to profiles dictionary
        self.profiles[singer_name] = profile
        
        # Save updated profiles
        return self._save_profiles()
    
    def _average_features(self, features_list):
        """Average features across multiple samples"""
        if not features_list:
            return {}
            
        # Initialize with the first feature set
        avg_features = features_list[0].copy()
        
        # Add all other feature sets
        for features in features_list[1:]:
            for key, value in features.items():
                if isinstance(value, list):
                    avg_features[key] = [a + b for a, b in zip(avg_features[key], value)]
                else:
                    avg_features[key] += value
        
        # Divide by the number of samples
        for key, value in avg_features.items():
            if isinstance(value, list):
                avg_features[key] = [v / len(features_list) for v in value]
            else:
                avg_features[key] /= len(features_list)
                
        return avg_features
    
    def identify_singer(self, audio_path, top_n=1, threshold=0.7):
        """
        Identify the singer in an audio file
        
        Args:
            audio_path: Path to the audio file
            top_n: Number of top matches to return
            threshold: Similarity threshold for a match (0 to 1)
            
        Returns:
            List of tuples with (singer_name, confidence_score)
        """
        if not self.profiles:
            print("No voice profiles available. Create profiles first.")
            return []
            
        # Extract features from the audio file
        features = self.extract_voice_features(audio_path)
        if not features:
            print(f"Could not extract features from {audio_path}")
            return []
            
        # Compare with each profile
        similarities = []
        for name, profile in self.profiles.items():
            similarity = self._calculate_similarity(features, profile['features'])
            similarities.append((name, similarity))
            
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold and take top_n
        matches = [(name, score) for name, score in similarities if score >= threshold]
        return matches[:top_n]
    
    def _calculate_similarity(self, features1, features2):
        """Calculate similarity between two sets of voice features"""
        try:
            # Get cosine similarity for each feature type and take weighted average
            similarities = []
            
            # MFCC mean vectors (higher weight as it's most important for voice identity)
            mfcc_sim = self._cosine_similarity(features1['mfcc_mean'], features2['mfcc_mean'])
            similarities.append((mfcc_sim, 0.5))  # 50% weight
            
            # MFCC standard deviation
            mfcc_std_sim = self._cosine_similarity(features1['mfcc_std'], features2['mfcc_std'])
            similarities.append((mfcc_std_sim, 0.1))  # 10% weight
            
            # Chroma mean
            chroma_sim = self._cosine_similarity(features1['chroma_mean'], features2['chroma_mean'])
            similarities.append((chroma_sim, 0.1))  # 10% weight
            
            # Spectral contrast
            contrast_sim = self._cosine_similarity(features1['contrast_mean'], features2['contrast_mean'])
            similarities.append((contrast_sim, 0.1))  # 10% weight
            
            # Pitch features (calculate relative difference)
            pitch_mean_diff = abs(features1['pitch_mean'] - features2['pitch_mean'])
            pitch_mean_sim = max(0, 1 - (pitch_mean_diff / max(features1['pitch_mean'], 1)))
            similarities.append((pitch_mean_sim, 0.2))  # 20% weight
            
            # Calculate weighted average
            weighted_sum = sum(sim * weight for sim, weight in similarities)
            total_weight = sum(weight for _, weight in similarities)
            
            return weighted_sum / total_weight
            
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
            
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norm_product == 0:
            return 0.0
            
        similarity = dot_product / norm_product
        # Normalize to 0-1 range (cosine similarity is between -1 and 1)
        similarity = (similarity + 1) / 2
        return similarity
    
    def get_profile_names(self):
        """Get names of all profiles in the database"""
        return list(self.profiles.keys())
    
    def get_profile(self, singer_name):
        """Get a specific voice profile"""
        return self.profiles.get(singer_name)
    
    def delete_profile(self, singer_name):
        """Delete a voice profile"""
        if singer_name in self.profiles:
            del self.profiles[singer_name]
            return self._save_profiles()
        return False
    
    def rename_profile(self, old_name, new_name):
        """Rename a voice profile"""
        if old_name in self.profiles and new_name not in self.profiles:
            self.profiles[new_name] = self.profiles[old_name]
            self.profiles[new_name]['name'] = new_name
            del self.profiles[old_name]
            return self._save_profiles()
        return False


# Helper functions for batch processing
def build_singer_database(base_folder, output_path=None):
    """
    Build a database of singer profiles from a folder structure
    Assumes each singer has a dedicated folder with their audio samples
    
    Args:
        base_folder: Base folder containing singer subfolders
        output_path: Path to save the database (default: use default path)
        
    Returns:
        VoiceProfiler instance with the built database
    """
    profiler = VoiceProfiler(model_path=output_path)
    
    # Find all subdirectories - each represents a singer
    singer_dirs = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    
    for singer_dir in singer_dirs:
        singer_name = singer_dir
        full_path = os.path.join(base_folder, singer_dir)
        
        # Find all audio files
        audio_files = []
        for root, _, files in os.walk(full_path):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.flac')):
                    audio_files.append(os.path.join(root, file))
        
        if audio_files:
            print(f"Creating profile for {singer_name} with {len(audio_files)} audio files")
            metadata = {
                'source_folder': full_path,
                'file_count': len(audio_files)
            }
            profiler.create_voice_profile(singer_name, audio_files, metadata)
    
    return profiler

def identify_singers_in_file(audio_path, profiler=None, segment_duration=30):
    """
    Identify singers in an audio file by analyzing segments
    
    Args:
        audio_path: Path to the audio file
        profiler: VoiceProfiler instance (creates a new one if None)
        segment_duration: Duration of each segment to analyze (in seconds)
        
    Returns:
        List of identified singers with confidence scores and timestamps
    """
    if profiler is None:
        profiler = VoiceProfiler()
    
    if not profiler.profiles:
        print("No voice profiles available. Create profiles first.")
        return []
    
    try:
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Segment the audio
        total_duration_ms = len(audio)
        segment_duration_ms = segment_duration * 1000
        
        # Create temporary directory for segments
        temp_dir = os.path.join(os.path.dirname(audio_path), "temp_segments")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Process segments
        results = []
        for start_ms in range(0, total_duration_ms, segment_duration_ms):
            end_ms = min(start_ms + segment_duration_ms, total_duration_ms)
            if end_ms - start_ms < 5000:  # Skip segments less than 5 seconds
                continue
                
            # Extract segment
            segment = audio[start_ms:end_ms]
            
            # Save to temporary file
            temp_file = os.path.join(temp_dir, f"segment_{start_ms}_{end_ms}.wav")
            segment.export(temp_file, format="wav")
            
            # Identify singer
            matches = profiler.identify_singer(temp_file, top_n=2, threshold=0.5)
            
            if matches:
                # Convert timestamps to readable format
                start_time = f"{start_ms // 60000}:{(start_ms % 60000) // 1000:02d}"
                end_time = f"{end_ms // 60000}:{(end_ms % 60000) // 1000:02d}"
                
                # Save results
                for singer, confidence in matches:
                    results.append({
                        'singer': singer,
                        'confidence': confidence,
                        'start_ms': start_ms,
                        'end_ms': end_ms,
                        'start_time': start_time,
                        'end_time': end_time
                    })
            
            # Clean up
            try:
                os.remove(temp_file)
            except:
                pass
        
        # Clean up temp directory
        try:
            os.rmdir(temp_dir)
        except:
            pass
            
        return results
        
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")
        return []


# Example usage
if __name__ == "__main__":
    # Create a voice profiler
    profiler = VoiceProfiler()
    
    # Example: Create a profile for a singer
    # profiler.create_voice_profile("Singer Name", ["path/to/audio1.wav", "path/to/audio2.mp3"])
    
    # Example: Identify a singer in a file
    # matches = profiler.identify_singer("path/to/unknown.wav")
    # if matches:
    #     print(f"Identified singer: {matches[0][0]} (confidence: {matches[0][1]:.2f})")
    # else:
    #     print("No match found")