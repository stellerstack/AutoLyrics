import os
from datasets import Dataset, DatasetDict

def ingest_local_dataset(audio_dir, transcript_dir):
    print("Pairing local audio files with lyric transcripts...")
    
    data = {"audio_path": [], "text": []}
    
    for filename in os.listdir(audio_dir):
        if filename.endswith(".wav"):
            base_name = filename.replace(".wav", "")
            transcript_path = os.path.join(transcript_dir, f"{base_name}.txt")
            
            if os.path.exists(transcript_path):
                with open(transcript_path, "r", encoding="utf-8") as f:
                    lyrics = f.read().strip()
                
                # Store absolute paths (as standard strings) and lyrics
                data["audio_path"].append(os.path.join(os.path.abspath(audio_dir), filename))
                data["text"].append(lyrics)

    raw_dataset = Dataset.from_dict(data)
    
    print("Creating Train/Test split (80/20)...")
    train_test_split = raw_dataset.train_test_split(test_size=0.2, seed=42)
    
    dataset = DatasetDict({
        "train": train_test_split["train"],
        "test": train_test_split["test"]
    })
    
    output_dir = "prepared_singing_dataset"
    dataset.save_to_disk(output_dir)
    print(f"\nSuccess! Bypassed HF Audio and saved to '{output_dir}'.")

if __name__ == "__main__":
    ingest_local_dataset("audio", "transcripts")