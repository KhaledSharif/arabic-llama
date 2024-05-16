import os
from glob import glob
import torch
import wandb
from datasets import Dataset
import json
from copy import deepcopy
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import ORPOConfig, ORPOTrainer, setup_chat_format

BATCH_SIZE = 1_000

# ref: https://huggingface.co/blog/mlabonne/orpo-llama-3

wandb.login()

if torch.cuda.get_device_capability()[0] >= 8:
    print("✅ using flash attention")
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
    print("❌ not using flash attention")
    attn_implementation = "eager"
    torch_dtype = torch.float16

base_model = "meta-llama/Meta-Llama-3-8B"
output_directory = ".model"

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "up_proj",
        "down_proj",
        "gate_proj",
        "k_proj",
        "q_proj",
        "v_proj",
        "o_proj",
    ],
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation,
)
model, tokenizer = setup_chat_format(model, tokenizer)
model = prepare_model_for_kbit_training(model)


# EXPECTED DATASET EXAMPLE
# --------------------------------------------------------------
#
# https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k
#
# --------------------------------------------------------------


def read_jsonl(file_path) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]
    return data


data = []
for data_file in glob("./jsonl/*.jsonl"):
    data += read_jsonl(data_file)


if len(data) < BATCH_SIZE:
    print(len(data), "is too small :(")
    exit()

data_dict = {"chosen": [], "rejected": [], "prompt": []}

for d in data:
    prompt = str(d["context"] + " Question: " + d["question"])
    new_prompt = {"content": prompt, "role": "user"}

    data_dict["chosen"].append(
        [
            deepcopy(new_prompt),
            {"content": d["answer_chosen"], "role": "assistant"},
        ]
    )

    data_dict["rejected"].append(
        [
            deepcopy(new_prompt),
            {"content": d["answer_rejected"], "role": "assistant"},
        ]
    )

    data_dict["prompt"].append(prompt)


dataset = Dataset.from_dict(data_dict)


dataset = dataset.shuffle(seed=42).select(range(BATCH_SIZE))


def format_chat_template(row):
    row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
    row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
    return row


dataset = dataset.map(
    format_chat_template,
    num_proc=os.cpu_count(),
)
dataset = dataset.train_test_split(test_size=0.01)


orpo_args = ORPOConfig(
    learning_rate=8e-6,
    beta=0.1,
    lr_scheduler_type="linear",
    max_length=1024,
    max_prompt_length=512,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    num_train_epochs=1,
    evaluation_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    report_to="wandb",
    output_dir="./results/",
)

trainer = ORPOTrainer(
    model=model,
    args=orpo_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model(output_directory)
