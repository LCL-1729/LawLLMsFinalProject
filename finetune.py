import os
import torch # Import torch explicitly for dtype
import unsloth
from unsloth import FastLanguageModel # Key import for Unsloth
from transformers import AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, PeftModel # PeftModel will be used for inference
from datasets import load_dataset, DatasetDict


# 1. Define Model Name (already set for Unsloth Qwen3 4-bit)
model_name = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
# It's good practice to define the output directory name based on the model
output_dir_name = "./lora_4bit_qwen3_unsloth"

# 2. Load tokenizer
# Qwen models might require trust_remote_code=True for the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    # Set pad_token_id for consistency if pad_token was initially None
    if tokenizer.pad_token_id is None: # Ensure pad_token_id is also set
        tokenizer.pad_token_id = tokenizer.eos_token_id


# 3. Load model with Unsloth's FastLanguageModel
# Unsloth handles BitsAndBytesConfig internally when load_in_4bit=True
# It also manages device_map automatically for GPU placement.
# For Qwen models, trust_remote_code=True is often necessary.
# Specify torch_dtype for the compute dtype during 4-bit loading.
# Common choices are torch.bfloat16 (if supported) or torch.float16.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 512, # Or your desired max_length
    dtype = None, # Unsloth will pick torch.bfloat16 if available, else torch.float16
    load_in_4bit = True,
    trust_remote_code = True, # Important for some models like Qwen
    # token = "hf_YOUR_TOKEN", # if using gated models or private repos
)

# 4. Apply LoRA adaptation using Unsloth's method
# model = FastLanguageModel.get_peft_model(...)
# Unsloth can often infer target_modules. If you want to be explicit:
# For Qwen models, common target modules include attention projections and MLP layers.
# The original ["q_proj", "v_proj"] is minimal. A more comprehensive set is better.
# --- CORRECTED LINE BELOW ---
model = FastLanguageModel.get_peft_model(
    model,
    r = 8, # LoRA rank
    lora_alpha = 16, # LoRA alpha
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj", # Attention projections
        "gate_proj", "up_proj", "down_proj"    # MLP linear layers
    ],
    lora_dropout = 0.05,
    bias = "none",    # Biases are usually not trained with LoRA.
    use_gradient_checkpointing = True, # Recommended for memory saving
    random_state = 3407, # For reproducibility
    use_rslora = False,  # Rank Stable LoRA (optional)
    loftq_config = None, # LoftQ configuration (optional)
    # task_type = "CAUSAL_LM", # <<< REMOVE THIS LINE >>>
)


# 5. Prepare your dataset (This section remains largely the same)
# Load the JSONL file as a single split and then train/test split
raw = load_dataset(
    "json",
    data_files="dataset.jsonl", # Ensure this file exists in your directory
    split="train"
)
# Split into train and validation
# Assuming your dataset.jsonl now has 100 examples, a 0.1 test_size gives 10 for validation.
# If your dataset size is different, adjust test_size or ensure num_validation_examples is appropriate.
num_total_examples = len(raw)
if num_total_examples >= 10: # Ensure there are enough samples for a reasonable split
    test_set_size = 0.1 # Use 10% for validation if dataset is large enough
    if num_total_examples * test_set_size < 1: # Ensure at least 1 validation sample
        test_set_size = 1 / num_total_examples if num_total_examples > 0 else 0.1
else: # If dataset is very small, might need a different strategy or fixed number
    test_set_size = 1 / num_total_examples if num_total_examples > 1 else 0.0 # Avoid error if only 1 sample

splits = raw.train_test_split(test_size=test_set_size, seed=42, shuffle=True) # Added shuffle

# Create DatasetDict
dataset = DatasetDict({
    "train": splits["train"],
    "validation": splits["test"]
})
print(f"Using {len(dataset['train'])} examples for training and {len(dataset['validation'])} for validation.")


# 6. Tokenization function (Remains the same, ensure prompt format matches Qwen3 expectations if specific)
def tokenize_fn(examples):
    prompts = []
    instructions = examples.get("instruction", [])
    inputs = examples.get("input", [""] * len(instructions))
    outputs = examples.get("output", [])

    for ins, inp, out in zip(instructions, inputs, outputs):
        text = f"### Instruction:\n{ins}\n"
        if inp and inp.strip():
            text += f"### Input:\n{inp}\n"
        text += f"### Response:\n{out}"
        text += tokenizer.eos_token
        prompts.append(text)

    tokenized = tokenizer(
        prompts,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# 7. Tokenize all splits
tokenized_dataset = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=dataset["train"].column_names
)
train_ds = tokenized_dataset["train"]
eval_ds = tokenized_dataset["validation"]


# 8. Define training arguments
# Note: Unsloth might recommend specific optimizers or settings for best performance.
# Check their documentation for Qwen3 finetuning examples.
training_args = TrainingArguments(
    output_dir=output_dir_name,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    save_steps=100, # Adjusted based on typical dataset size, review if total steps are very low
    # evaluation_strategy="steps", # REMOVED this line
    eval_steps=100, # Trainer will evaluate at these steps if eval_dataset is provided
    save_total_limit=2,
    remove_unused_columns=False,
    seed=42,
    # optim="adamw_torch_fused", # Consider if using Unsloth's recommended optimizers
    # report_to="wandb", # Example for logging to Weights & Biases
)

# Ensure evaluation is triggered if eval_ds and eval_steps are set
if eval_ds is not None and training_args.eval_steps is not None:
    print(f"Evaluation will be performed every {training_args.eval_steps} steps.")
    # For older transformers versions, explicitly setting do_eval might be necessary
    # if the Trainer doesn't infer it automatically.
    # However, usually providing eval_dataset and eval_steps is enough.
    # training_args.do_eval = True # Uncomment if evaluation doesn't run

# 9. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds, # Make sure eval_ds is not None for evaluation to occur
    tokenizer=tokenizer,
)

# 10. Train the model
print("Starting training...")
trainer.train()
print("Training finished.")

# 11. Save LoRA adapters
print(f"Saving LoRA adapter to {output_dir_name}")
model.save_pretrained(output_dir_name)

# 12. Inference (Example) - Ensure this part is updated if tokenizer was reloaded by FastLanguageModel
# from peft import PeftModel # Already imported at the top

print("Setting up model for inference...")

# It's crucial that the tokenizer used for inference is the same one
# that was active when the model was prepared with FastLanguageModel for training,
# especially if FastLanguageModel modified it (e.g., added special tokens or changed padding).
# The `tokenizer` variable from step 2 (and potentially modified by FastLanguageModel in step 3)
# should be the correct one to use here.

# Reloading the base model and tokenizer for inference:
# This ensures a clean state and applies Unsloth's optimizations correctly.
base_model_inf, tokenizer_inf = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 512,
    dtype = None,
    load_in_4bit = True,
    trust_remote_code = True,
    # token = "hf_YOUR_TOKEN",
)

# Apply the PEFT adapter
inference_model = PeftModel.from_pretrained(base_model_inf, output_dir_name)
inference_model.eval()

def generate_sample(model_to_use, tokenizer_to_use):
    prompt = (
        "### Instruction:\nDetermine if the following individual qualifies as a dependent under IRS §152."
        "\n### Input:\nSarah is a 19-year-old student living with her parents, earned $2,000 last year, and her parents provided 90% of her support."
        "\n### Response:\n"
    )
    # Ensure the tokenizer used here is the one compatible with the loaded inference_model
    inputs = tokenizer_to_use(prompt, return_tensors="pt", padding=True, truncation=True).to(model_to_use.device)

    print("Generating response...")
    with torch.no_grad():
        outputs = model_to_use.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            eos_token_id=tokenizer_to_use.eos_token_id,
            pad_token_id=tokenizer_to_use.pad_token_id if tokenizer_to_use.pad_token_id is not None else tokenizer_to_use.eos_token_id,
        )
    decoded_output = tokenizer_to_use.decode(outputs[0], skip_special_tokens=True)
    print("---PROMPT---")
    print(prompt)
    print("---RESPONSE---")
    response_only = decoded_output.split("### Response:\n")[-1].strip()
    print(response_only)

if __name__ == "__main__":
    print("Running sample generation post-training...")
    # Use the tokenizer that was loaded and potentially modified with the inference model
    generate_sample(inference_model, tokenizer_inf)

