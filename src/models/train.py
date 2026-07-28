# train.py
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from src.models.dataset import build_dataset
from src.structuring.serializer import RESPONSE_MARKER

MODEL_NAME ="meta-llama/Meta-Llama-3-8B-Instruct" 
MAX_SEQ_LENGTH = 4096

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

train_dataset = build_dataset("data/processed/train.jsonl", tokenizer, MAX_SEQ_LENGTH)
val_dataset = build_dataset("data/processed/val.jsonl", tokenizer, MAX_SEQ_LENGTH)

collator = DataCollatorForCompletionOnlyLM(
    response_template=RESPONSE_MARKER,
    tokenizer=tokenizer,
)

training_args = TrainingArguments(
    output_dir="outputs/lora_v0",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,   # effective batch = 16
    num_train_epochs=2,
    learning_rate=2e-4,
    warmup_steps=20,
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    save_strategy="epoch",
    eval_strategy="epoch",
    seed=3407,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    data_collator=collator,
    args=training_args,
)

trainer.train()
model.save_pretrained("outputs/lora_v0")
tokenizer.save_pretrained("outputs/lora_v0")