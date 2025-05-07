import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
from datasets import load_dataset, DatasetDict

# 1. Select an open-source model "Qwen3-4B"
model_name = "Qwen3-4B" # Changed as per your instruction

# 2. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. Configure 4-bit quantization with bitsandbytes (GPU-only)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NF4 quantization
    bnb_4bit_compute_dtype="float16" # Kept as "float16", common and usually works.
)

# 4. Load model with quantization and place on GPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": "cuda:0"}  # force all layers onto GPU
)

# 5. Apply LoRA adaptation
# Using the original target_modules. You may need to verify/adjust these
# for Qwen3-4B if they are not appropriate.
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"], # Kept original target_modules for minimal change
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 6. Prepare your dataset
# -----------------------
# Load the JSONL file as a single split and then train/test split
raw = load_dataset(
    "json",
    data_files="dataset.jsonl", # Ensure this file exists
    split="train"
)
# Split into train and validation
splits = raw.train_test_split(test_size=0.1, seed=42)
# Create DatasetDict
dataset = DatasetDict({
    "train": splits["train"],
    "validation": splits["test"]
})

# 7. Tokenization function
def tokenize_fn(examples):
    prompts = []
    for ins, inp, out in zip(
        examples.get("instruction", []),
        examples.get("input", [""] * len(examples.get("instruction", []))),
        examples.get("output", [])
    ):
        text = f"### Instruction:\n{ins}\n"
        if inp:
            text += f"### Input:\n{inp}\n"
        text += f"### Response:\n{out}{tokenizer.eos_token}" # Added EOS token for better sequence handling
        prompts.append(text)

    tokenized = tokenizer(
        prompts,
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# 8. Tokenize all splits
tokenized = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=dataset["train"].column_names
)
train_ds = tokenized["train"]
eval_ds = tokenized["validation"]

# 9. Define training arguments
training_args = TrainingArguments(
    output_dir="./lora_4bit_qwen3_hf_standard", # Changed output directory as per your instruction
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4, # Assuming eval batch size should also be 4
    num_train_epochs=3,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    remove_unused_columns=False # Kept as in original, map already handles removal
    # Removed evaluation_strategy and eval_steps to keep changes minimal to what was explicitly requested
)

# 10. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds, # eval_dataset is provided
    tokenizer=tokenizer
)

# 11. Train the model
trainer.train()

# 12. Save LoRA adapters
output_save_directory = "./lora_4bit_qwen3_hf_standard" # Changed save directory as per your instruction
model.save_pretrained(output_save_directory)
tokenizer.save_pretrained(output_save_directory)

# 13. Sample generation function
def generate_sample(current_model, current_tokenizer): # Renamed arguments for clarity
    prompt = (
        "### Instruction:\nDetermine if the following individual qualifies as a dependent under IRS §152."
        "\n### Input:\nSarah is a 19-year-old student living with her parents, earned $2,000 last year, and her parents provided 90% of her support."
        "\n### Response:\n"
    )
    inputs = current_tokenizer(prompt, return_tensors="pt").to(current_model.device)
    outputs = current_model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=current_tokenizer.eos_token_id # Good practice to include
    )
    print(current_tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    # Set model to evaluation mode for generation
    trainer.model.eval()
    generate_sample(trainer.model, tokenizer)