import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def run_zero_shot_baseline(audio_path):
    # 1. Load the pre-trained Whisper model (we use 'small' to balance speed and accuracy)
    model_id = "openai/whisper-small"
    print(f"Loading base model: {model_id}...")
    
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    
    # Move to GPU if you have one, otherwise fall back to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on {device.upper()}")
    
    try:
        # 2. Load and resample audio to 16kHz (Whisper strictly requires 16,000 Hz)
        print(f"\nProcessing '{audio_path}'...")
        speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
        
        # 3. Extract features and generate tokens
        inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        
        # 4. Generate the transcription
        print("Transcribing...")
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
            
        # 5. Decode the tokens back into readable text
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        print("\n=== ZERO-SHOT BASELINE RESULT ===")
        print(transcription.strip())
        print("=================================\n")
        
    except FileNotFoundError:
        print(f"\nERROR: Could not find the file '{audio_path}'.")
        print("Please add a short singing audio file to your folder and try again.")

if __name__ == "__main__":
    # Point this to your test audio file
    TEST_AUDIO = "sample_singing.wav"
    run_zero_shot_baseline(TEST_AUDIO)