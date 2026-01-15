from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import json
from datetime import datetime
import torch
import os
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class SimpleDataCollator:
    """Simple collator that just converts to tensors"""
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract fields
        batch = {}
        
        # Stack input_ids
        batch['input_ids'] = torch.tensor(
            [f['input_ids'] for f in features], 
            dtype=torch.long
        )
        
        # Stack attention_mask
        batch['attention_mask'] = torch.tensor(
            [f['attention_mask'] for f in features],
            dtype=torch.long
        )
        
        # Stack labels
        batch['labels'] = torch.tensor(
            [f['labels'] for f in features],
            dtype=torch.long
        )
        
        return batch

def train_qwen_text2sql():
    """
    Simple, effective QLoRA fine-tuning
    Fine-tune Qwen2.5-Coder-3B-Instruct for Text2SQL using QLoRA
    Focus: Get the basics RIGHT
    """
    start_time = datetime.now()
    print("QWEN 2.5-CODER 3B - TEXT2SQL FINE-TUNING")

    output_dir = f"models/qwen3b_baseline_r16_{start_time.strftime('%Y%m%d')}"
    os.makedirs(output_dir, exist_ok=True)  # Create directory

    # 1. LOAD MODEL
    print("\n1. Loading model...")
    
    model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,  # QLoRA: 4-bit quantization
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare for QLoRA
    model = prepare_model_for_kbit_training(model)

    # 2. LORA CONFIG
    print("\n2. Setting up LoRA...")
    
    lora_config = LoraConfig(
        r=16,                    # Rank (16 is good balance)
        lora_alpha=32,           # Alpha (typically 2x rank)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
        lora_dropout=0.05,       # Small dropout
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 3. LOAD DATA
    print("\n3. Loading training data...")
    
    dataset = load_dataset(
        'json',
        data_files={
            'train': 'data/processed/train.jsonl',
            'validation': 'data/processed/val.jsonl'
        }
    )
    
    print(f"Train examples: {len(dataset['train'])}")
    print(f"Val examples: {len(dataset['validation'])}")
    
    # Tokenize
    def tokenize_function(examples):
        result= tokenizer(
            examples['text'],
            truncation=True,
            max_length=1024,  
            padding='max_length'
        )
        result["labels"] = list(result["input_ids"])
        return result
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=False,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing"

    )
    
    # 4. TRAINING ARGUMENTS
    print("\n4. Setting up training...")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # Training hyperparameters
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size = 16
        
        learning_rate=2e-4,
        warmup_steps=100,
        
        # Logging
        logging_steps=10,
        logging_dir=f"{output_dir}/logs",
        
        # Evaluation
        eval_steps=200,
        eval_strategy="steps",
        
        # Saving
        save_steps=400,
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        
        # Optimization
        fp16=True,
        optim="adamw_torch",
        
        # Reporting
        report_to="tensorboard"
    )
    

    # 5. DATA COLLATOR & TRAINER
    print("\n5. Creating trainer...")

    data_collator = SimpleDataCollator()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=data_collator
    )
    
    # 6. TRAIN!
    print("\n6. Starting training...")

    trainer.train()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 3600  # hours
    
    # 7. SAVE METADATA
    print("\n7. Saving metadata...")
    
    metadata = {
        'name': 'qwen3b_baseline_r16',
        'model': 'Qwen2.5-Coder-3B-Instruct',
        'hyperparameters': {
            'lora_r': 16,
            'lora_alpha': 32,
            'max_length': 1024,
            'batch_size': 4,
            'learning_rate': 2e-4,
            'epochs': 3
        },
        'training': {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_hours': round(duration, 2),
            'train_examples': len(dataset['train']),
            'val_examples': len(dataset['validation']),
            'final_train_loss': trainer.state.log_history[-1].get('loss', None),
            'final_val_loss': trainer.state.log_history[-1].get('eval_loss', None)
        },
        'model_path': output_dir,
        'notes': 'Initial training run with r=16'
    }

    # Save metadata alongside model
    with open(f'{output_dir}/training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to {output_dir}/training_metadata.json")
    
    # 8. SAVE MODEL
    print("\n8. Saving final model...")
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n✅ TRAINING COMPLETE!")
    print(f"Model saved to: {output_dir}")
    return trainer


if __name__ == "__main__":
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    else:
        print("⚠ WARNING: No GPU detected!")
        print("Training on CPU will be VERY slow (days instead of hours)\n")
    
    # Start training
    trainer = train_qwen_text2sql()