import os
from pydub import AudioSegment

# Set path to your audio files
INPUT_FOLDER = "C:\\Users\\arshd\\Desktop\\FOLDER01\\edited"
OUTPUT_FOLDER = "C:\\Users\\arshd\\Desktop\\FOLDER01\\edited\\louder"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# dB change: positive = louder, negative = quieter
volume_increase_db = 6  

# Supported file formats
extensions = ('.mp3', '.wav')

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(extensions):
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        print(f"Processing: {filename}")
        audio = AudioSegment.from_file(input_path)
        louder_audio = audio + volume_increase_db
        louder_audio.export(output_path, format=filename.split('.')[-1])

print("Volume increase complete.")
