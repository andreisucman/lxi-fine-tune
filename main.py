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
LORA_RANK = 8  # Increased rank for better performance
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
GRAD_ACCUM_STEPS = 1
BATCH_SIZE = 8  # Increased for A100 80GB
MAX_SEQ_LENGTH = 4096  # Full context length
EPOCHS = 8
LEARNING_RATE = 2e-5  # Optimized learning rate
OUTPUT_DIR = "./gemma-3-4b-it-lora-finetuned"
REPO_ID = "Sunchain/gemma-3-4b-it-dolly-alpaca-ro"

# Load environment variables
HF_TOKEN = os.environ.get("HF_TOKEN")
assert HF_TOKEN, "Missing Hugging Face token in environment variable 'HF_TOKEN'"

# Configure tokenizer for Flash Attention
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    padding_side="right",  # Required for Flash Attention
    truncation_side="right"
)
tokenizer.pad_token = tokenizer.eos_token

# QLoRA Configuration (Double Quantization + NF4)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True  # Double quantization for memory efficiency
)

# Load model with Flash Attention
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
    attn_implementation="flash_attention_2",  # Enable Flash Attention
    torch_dtype=torch.bfloat16
)

# Prepare model for QLoRA training
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False  # Required for Flash Attention
model.config.pretraining_tp = 1  # Disable tensor parallelism

# LoRA Configuration (covering all linear layers)
peft_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=LORA_DROPOUT,
    task_type="CAUSAL_LM",
    bias="none"
)

# ---------------------------
# Data preparation functions
# ---------------------------

def download_file(url, output_path):
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
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
    "https://lxi-data.fra1.digitaloceanspaces.com/summarize-legal-ro-18k.jsonl",
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
# Training setup with Flash Attention optimizations
# ---------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    optim="paged_adamw_8bit",  # Memory-efficient optimizer
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    bf16=True,  # Force BF16 for A100
    fp16=False,  # Disable FP16 when using BF16
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="none",
    gradient_checkpointing=True,  # Enable for memory savings
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_steps=-1,
    group_by_length=True,  # Improves efficiency with packing
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

# Save final model
trainer.save_model(OUTPUT_DIR)

# Push to Hub (adapter only)
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