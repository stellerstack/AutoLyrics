import torch
import librosa
import gradio as gr
import jiwer
import warnings
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

warnings.filterwarnings("ignore")

# 1. Load the Base Model and Processor
model_id = "openai/whisper-small"
print("Loading processor and base model...")
processor = WhisperProcessor.from_pretrained(model_id, language="English", task="transcribe")
base_model = WhisperForConditionalGeneration.from_pretrained(model_id)

# 2. Try to Inject your trained LoRA Adapters
print("Looking for LoRA adapters...")
try:
    # If your training finished, it loads your custom brain
    model = PeftModel.from_pretrained(base_model, "./autolyrics_final_lora")
    print("✅ Success: AUTOLYRICS Fine-tuned model loaded!")
except Exception as e:
    # If training was interrupted, it falls back to the base model so the app still runs
    print("⚠️ LoRA adapters not found or incomplete. Falling back to base Whisper model.")
    model = base_model

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 3. The Inference & Evaluation Engine
def transcribe_and_evaluate(audio_path, ground_truth_lyrics):
    if not audio_path:
        return "Please upload an audio file.", ""
    
    speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
    
    inputs = processor(
        speech_array, 
        sampling_rate=sampling_rate, 
        return_tensors="pt",
        truncation=True
    )
    input_features = inputs.input_features.to(device)
    
    with torch.no_grad():
        predicted_ids = model.generate(input_features, max_new_tokens=400)
    
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    
    # Calculate Word Error Rate (WER) and Character Error Rate (CER)
    metrics_output = "No reference lyrics provided. Metrics skipped."
    if ground_truth_lyrics.strip():
        truth_clean = " ".join(ground_truth_lyrics.lower().split())
        trans_clean = " ".join(transcription.lower().split())
        
        try:
            wer = jiwer.wer(truth_clean, trans_clean)
            cer = jiwer.cer(truth_clean, trans_clean)
            metrics_output = f"📊 Word Error Rate (WER): {wer * 100:.2f}%\n"
            metrics_output += f"📉 Character Error Rate (CER): {cer * 100:.2f}%"
        except Exception:
            metrics_output = "Error calculating metrics. Ensure reference text is valid."

    return transcription, metrics_output

# 4. Build the UI
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🎙️ AUTOLYRICS: Singing-Voice ASR")
    gr.Markdown("Upload a singing audio clip. The system will transcribe the lyrics and calculate performance metrics.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Upload Singing Audio (.wav)")
            reference_input = gr.Textbox(lines=4, label="Ground Truth Lyrics (Optional - Used for WER calculation)", placeholder="Paste actual lyrics here...")
            submit_btn = gr.Button("Transcribe Audio", variant="primary")
            
        with gr.Column():
            transcription_output = gr.Textbox(lines=6, label="Model Transcription")
            metrics_output = gr.Textbox(lines=2, label="Evaluation Metrics (Lower is better)")

    submit_btn.click(
        fn=transcribe_and_evaluate,
        inputs=[audio_input, reference_input],
        outputs=[transcription_output, metrics_output]
    )

if __name__ == "__main__":
    print("\nStarting AUTOLYRICS web server...")
    demo.launch(server_name="127.0.0.1", server_port=7860)