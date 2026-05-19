import torch
import librosa
from datasets import load_from_disk
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import warnings

# Hides some verbose Hugging Face warnings to keep your terminal clean
warnings.filterwarnings("ignore") 

def run_training():
    print("Loading prepared dataset...")
    dataset = load_from_disk("prepared_singing_dataset")
    
    model_id = "openai/whisper-small"
    print(f"Loading base model: {model_id}...")
    processor = WhisperProcessor.from_pretrained(model_id, language="English", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)

    config = LoraConfig(
        r=8, 
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05,
        bias="none"
    )
    
    model = get_peft_model(model, config)
    print("\n--- MODEL PARAMETERS ---")
    model.print_trainable_parameters() 
    print("------------------------\n")

    def prepare_dataset(batch):
        # We load the audio manually using librosa
        speech_array, sampling_rate = librosa.load(batch["audio_path"], sr=16000)
        
        # FIX 1: Truncate audio to Whisper's 30-second limit
        batch["input_features"] = processor.feature_extractor(
            speech_array, 
            sampling_rate=sampling_rate,
            truncation=True
        ).input_features[0]
        
        # FIX 2: Truncate lyrics to Whisper's max token limit (448)
        batch["labels"] = processor.tokenizer(
            batch["text"],
            max_length=448,
            truncation=True
        ).input_ids
        
        return batch

    print("Extracting features and tokenizing lyrics (this takes a moment)...")
    tokenized_dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=1)

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./autolyrics_checkpoints",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        num_train_epochs=1, 
        fp16=False, 
        eval_strategy="steps",       # FIX 3: Updated to comply with newer Transformers library
        eval_steps=5,
        save_steps=5,
        logging_steps=2, 
        remove_unused_columns=False, 
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        processing_class=processor.feature_extractor, # <--- New Hugging Face syntax
    )

    print("\nStarting LoRA Fine-Tuning Engine...")
    trainer.train()

    model.save_pretrained("./autolyrics_final_lora")
    print("\nSuccess! LoRA adapters saved to './autolyrics_final_lora'.")

if __name__ == "__main__":
    run_training()