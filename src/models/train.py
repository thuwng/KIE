# train.py
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # đặt TRƯỚC khi import torch
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.models.dataset import build_dataset, make_collate_fn

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_SEQ_LENGTH = 4096

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

#model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto", attn_implementation="sdpa")
from liger_kernel.transformers import AutoLigerKernelForCausalLM

model = AutoLigerKernelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()      # THÊM — giảm mạnh memory activation, đánh đổi ~20-30% tốc độ
model.config.use_cache = False  
model.print_trainable_parameters()

train_dataset = build_dataset("data/processed/train.jsonl", tokenizer, MAX_SEQ_LENGTH)
val_dataset = build_dataset("data/processed/val.jsonl", tokenizer, MAX_SEQ_LENGTH)

training_args = TrainingArguments(
    output_dir="outputs/lora_v0",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    group_by_length=True,
    num_train_epochs=2,
    learning_rate=2e-4,
    warmup_steps=20,
    logging_steps=10,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    bf16=True,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="none",
    seed=3407,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=make_collate_fn(tokenizer.pad_token_id),
)
trainer.train()
model.save_pretrained("outputs/lora_v0")
tokenizer.save_pretrained("outputs/lora_v0")