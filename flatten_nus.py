import os
import shutil

# ⚠️ CHANGE THIS PATH to where you extracted the downloaded dataset!
# Example Windows path: "C:/Users/mayan/Downloads/nus-smc-corpus_..."
SOURCE_DIR = r"C:\Users\mayan\OneDrive\Desktop\autolyrics\drive-download-20260519T211359Z-3-001"

TARGET_AUDIO_DIR = "audio"
TARGET_TRANSCRIPT_DIR = "transcripts"

# Create target directories
os.makedirs(TARGET_AUDIO_DIR, exist_ok=True)
os.makedirs(TARGET_TRANSCRIPT_DIR, exist_ok=True)

print(f"Scanning {SOURCE_DIR} for audio and lyrics...")

count = 0
for root, _, files in os.walk(SOURCE_DIR):
    for file in files:
        if file.endswith(".wav"):
            shutil.copy2(os.path.join(root, file), os.path.join(TARGET_AUDIO_DIR, file))
            count += 1
        elif file.endswith(".txt"):
            shutil.copy2(os.path.join(root, file), os.path.join(TARGET_TRANSCRIPT_DIR, file))

print(f"Done! Flattened {count} audio files and their matching transcripts.")