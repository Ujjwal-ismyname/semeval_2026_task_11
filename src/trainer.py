import os
import sys
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from dataset import SyllogismDataset
from model import get_model_and_tokenizer

def train():
    # Configuration
    MODEL_NAME = "CohereForAI/aya-23-8B" # or "meta-llama/Meta-Llama-3-8B-Instruct"
    # Note: User might need to login to HF Hub for Llama 3 provided they have access. 
    # Aya-23 is open weights usually.
    
    DATA_PATH = "semeval_2026_task_11/data/raw/train_data.json"
    OUTPUT_DIR = "semeval_2026_task_11/output"
    
    # 1. Load Model & Tokenizer
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, use_lora=True)
    
    # 2. Load Dataset
    print("Loading Dataset...")
    # Enable distractor augmentation for Subtask 2 simulation
    train_dataset = SyllogismDataset(
        file_path=DATA_PATH, 
        tokenizer=tokenizer, 
        max_length=512,
        augment_distractors=True,   # Critical for Subtask 2
        num_distractors=2           # Inject 2 irrelevant sentences
    )
    
    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=1, # Start with 1 epoch for demo
        save_steps=100,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none" # disable wandb for now
    )
    
    # 4. Data Collator
    # We use DataCollatorForLanguageModeling but with mlm=False (Causal LM)
    # Actually, our Dataset already returns 'input_ids', 'labels', 'attention_mask'.
    # We can use the default data collator or DataCollatorwithPadding.
    # Since we pad in dataset, default collator works if it stacks tensors.
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        # data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False) # Not needed if dataset yields labels
    )
    
    # 5. Train
    print("Starting Training...")
    trainer.train()
    
    # 6. Save
    print(f"Saving model to {OUTPUT_DIR}/final_model")
    trainer.model.save_pretrained(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")

if __name__ == "__main__":
    train()
