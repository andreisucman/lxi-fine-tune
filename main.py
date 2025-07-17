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

# ---------------------------
# Configuration
# ---------------------------
MODEL_ID = "google/gemma-3-4b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LORA_RANK = 8
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8  # Effective batch size = 4 * 8 = 32
EPOCHS = 8
LEARNING_RATE = 5e-5
OUTPUT_DIR = "./gemma-3-4b-it-lora-finetuned"
REPO_ID = "Sunchain/gemma-3-4b-it-dolly-alpaca-ro"

# ---------------------------
# Environment Variables
# ---------------------------
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME")
KAGGLE_KEY = os.environ.get("KAGGLE_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

assert HF_TOKEN, "Missing Hugging Face token in environment variable 'HF_TOKEN'"
assert KAGGLE_USERNAME and KAGGLE_KEY, "Missing Kaggle credentials"

# ---------------------------
# Tokenizer & Model (4-bit)
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN
)
model = prepare_model_for_kbit_training(model)

# ---------------------------
# LoRA Config
# ---------------------------
peft_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=16,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    bias="none"
)

# ---------------------------
# Dataset Download + Prep
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

    response = str(row.get("response") or row.get("output") or "").strip()
    return {"prompt": prompt, "response": response}

# Dolly
download_file("https://huggingface.co/api/datasets/databricks/databricks-dolly-15k/parquet/default/train/0.parquet",
              "data/parquet/dolly/train.parquet")
dolly_df = pd.read_parquet("data/parquet/dolly/train.parquet")
dolly_data = dolly_df.apply(extract_prompt_response, axis=1).tolist()

# Alpaca
download_file("https://huggingface.co/api/datasets/yahma/alpaca-cleaned/parquet/default/train/0.parquet",
              "data/parquet/alpaca/train.parquet")
alpaca_df = pd.read_parquet("data/parquet/alpaca/train.parquet")
alpaca_data = alpaca_df.apply(extract_prompt_response, axis=1).tolist()

# Legal
download_file("https://lxi-data.fra1.digitaloceanspaces.com/summarize-legal-ro-18k.jsonl",
              "legal-summarization-ro.jsonl")
legal_data = []
with open("legal-summarization-ro.jsonl") as f:
    for line in f:
        example = json.loads(line)
        legal_data.append({
            "prompt": example["instruction"],
            "response": example["response"]
        })

# Combine all
combined_data = dolly_data + alpaca_data + legal_data
print(f"Total training examples: {len(combined_data)}")

# ---------------------------
# Chat formatting + Tokenization
# ---------------------------
dataset = Dataset.from_list(combined_data).train_test_split(test_size=0.1)

def format_chat(example):
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}

formatted = dataset.map(format_chat, remove_columns=["prompt", "response"])

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

tokenized = formatted.map(tokenize, batched=True, remove_columns=["text"])
tokenized.set_format("torch")

# ---------------------------
# Training Arguments
# ---------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
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

# ---------------------------
# Trainer
# ---------------------------
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    peft_config=peft_config,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# ---------------------------
# Train & Save
# ---------------------------
trainer.train()
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

trainer.model.push_to_hub(REPO_ID, token=HF_TOKEN, private=True)
tokenizer.push_to_hub(REPO_ID, token=HF_TOKEN)

print("✅ Training complete and model pushed to Hub.")
