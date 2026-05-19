---

### 3. AUTOLYRICS

````markdown
# 🎙️ AUTOLYRICS: Parameter-Efficient LoRA Fine-Tuning for ASR

Standard Automatic Speech Recognition (ASR) systems suffer from "infinite hallucination loops" when fed singing voices due to melodic pitch and elongated vowels. AUTOLYRICS is an ML engineering pipeline that patches this flaw in `openai/whisper-small` using surgical, parameter-efficient fine-tuning.

## 🧠 Engineering & Data Pipeline

- **PEFT / LoRA Injection:** Bypassed the massive VRAM constraints of full-parameter tuning by freezing the base model and injecting Low-Rank Adaptation (LoRA) adapters strictly into the `q_proj` and `v_proj` attention modules. **Reduced trainable parameters from 242M to just 884K (0.36%).**
- **Strict Memory Truncation:** Engineered the tokenizer and feature extractor to strictly enforce Whisper’s 30-second context window and 448-token generation limit to prevent out-of-memory (OOM) crashes during audio ingestion.
- **Custom C++ Dependency Bypass:** Wrote a custom data collator using `librosa` to directly parse and feed raw `.wav` paths, completely bypassing Hugging Face's reliance on fragile, system-level Windows FFmpeg installations.
- **Automated CI/Evaluation:** Integrated `jiwer` directly into the inference loop to mathematically calculate Word Error Rate (WER) and Character Error Rate (CER) in real-time against ground-truth transcripts.

## 🛠️ Deployment & Execution

- **Stack:** PyTorch, Hugging Face (Transformers, PEFT, Datasets), Librosa, Gradio

**To launch the automated inference UI:**

```bash
pip install -r requirements.txt
python app.py
# Access the local evaluation server at [http://127.0.0.1:7860](http://127.0.0.1:7860)
```
````
