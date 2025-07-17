import os
import torch
import pandas as pd
import requests
import json
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback
)
from trl import SFTTrainer

# Configuration
MODEL_ID = "google/gemma-3-4b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LORA_RANK = 4
BATCH_SIZE = 3
EPOCHS = 4
LEARNING_RATE = 5e-5
OUTPUT_DIR = "./gemma-3-4b-it-lora-finetuned"
REPO_ID = "Sunchain/gemma-3-4b-it-dolly-alpaca-ro"

# Load environment variables
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME")
KAGGLE_KEY = os.environ.get("KAGGLE_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

assert HF_TOKEN, "Missing Hugging Face token in environment variable 'HF_TOKEN'"
assert KAGGLE_USERNAME and KAGGLE_KEY, "Missing Kaggle credentials in environment variables"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# Optional: quantization setup
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    # quantization_config=bnb_config,  # Uncomment if using quantization
    device_map="auto",
    token=HF_TOKEN
)
model = prepare_model_for_kbit_training(model)

# LoRA config
peft_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=16,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    bias="none"
)

# ---------------------------
# Data preparation functions
# ---------------------------

def download_file(url, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not os.path.exists(output_path):
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)

def extract_prompt_response(row):
    prompt = row["instruction"].strip()
    if row.get("context"):
        prompt += " " + str(row["context"]).strip()
    if row.get("input"):
        prompt += " " + str(row["input"]).strip()

    response = ""
    if "response" in row and pd.notna(row["response"]):
        response = str(row["response"]).strip()
    elif "output" in row and pd.notna(row["output"]):
        response = str(row["output"]).strip()

    return {"prompt": prompt, "response": response}

# ---------------------------
# Download and process datasets
# ---------------------------

# Dolly
dolly_path = "data/parquet/dolly/train.parquet"
download_file(
    "https://huggingface.co/api/datasets/databricks/databricks-dolly-15k/parquet/default/train/0.parquet",
    dolly_path
)
dolly_df = pd.read_parquet(dolly_path)
dolly_data = dolly_df.apply(extract_prompt_response, axis=1).tolist()

# Alpaca
alpaca_path = "data/parquet/alpaca/train.parquet"
download_file(
    "https://huggingface.co/api/datasets/yahma/alpaca-cleaned/parquet/default/train/0.parquet",
    alpaca_path
)
alpaca_df = pd.read_parquet(alpaca_path)
alpaca_data = alpaca_df.apply(extract_prompt_response, axis=1).tolist()

# Legal Summarization
legal_path = "legal-summarization-ro.jsonl"
download_file(
    "https://lxi-data.fra1.cdn.digitaloceanspaces.com/summarize-legal-ro-18k.jsonl",
    legal_path
)
legal_data = []
with open(legal_path) as f:
    for line in f:
        example = json.loads(line)
        legal_data.append({
            "prompt": example["instruction"],
            "response": example["response"]
        })

# Combine datasets
combined_data = dolly_data + alpaca_data + legal_data
print(f"Total training examples: {len(combined_data)}")

# Convert to HF dataset
dataset = Dataset.from_list(combined_data).train_test_split(test_size=0.1)

# Format for chat template
def format_chat(example):
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]}
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

dataset = dataset.map(format_chat, remove_columns=["prompt", "response"])

# ---------------------------
# Training setup
# ---------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    optim="adamw_torch",
    logging_steps=10,
    save_strategy="steps",
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    report_to="none",
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    max_grad_norm=0.3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    save_total_limit=2,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train
trainer.train()

# Save final model and tokenizer
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Push to Hub
trainer.model.push_to_hub(
    REPO_ID,
    use_temp_dir=False,
    token=HF_TOKEN,
    private=True
)

tokenizer.push_to_hub(
    REPO_ID,
    use_temp_dir=False,
    token=HF_TOKEN
)

print("Model pushed to Hub!")
