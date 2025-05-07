import json
import os
import torch
import re # For regular expression matching
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from peft import PeftModel

# --- Model Configuration ---
# Base model name used for fine-tuning
base_model_name = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
# Path to your fine-tuned LoRA adapters
lora_model_path = "./lora_4bit_qwen3_unsloth"
# Max sequence length used during training
max_seq_length = 512 # This is for the input to the model

# --- Global variables for model and tokenizer ---
# These will be initialized once by load_model_and_tokenizer()
inference_model = None
tokenizer = None

def load_model_and_tokenizer():
    """
    Loads the fine-tuned model and tokenizer for inference.
    This function should be called once before generating outputs.
    """
    global inference_model, tokenizer

    if not os.path.exists(lora_model_path):
        print(f"Error: LoRA model path not found: {lora_model_path}")
        print("Please ensure the fine-tuned model adapters are at this location.")
        print("Generating summary table with placeholder 'Error: Model not found' for Model Outputs.")
        return False

    print(f"Loading base model '{base_model_name}' for inference...")
    try:
        base_model_loaded, tokenizer_loaded = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=max_seq_length,
            dtype=None,  # Unsloth will pick torch.bfloat16 if available, else torch.float16
            load_in_4bit=True,
            trust_remote_code=True,
            # token = "hf_YOUR_TOKEN", # if using gated models or private repos
        )
        print("Base model loaded.")

        print(f"Loading LoRA adapter from '{lora_model_path}'...")
        # Apply the PEFT adapter
        inference_model_loaded = PeftModel.from_pretrained(base_model_loaded, lora_model_path)
        inference_model_loaded.eval()  # Set the model to evaluation mode
        print("LoRA adapter applied. Model ready for inference.")

        # Assign to global variables
        inference_model = inference_model_loaded
        tokenizer = tokenizer_loaded

        # Set pad_token if not already set (common practice)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return True

    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        print("Generating summary table with placeholder 'Error: Model loading failed' for Model Outputs.")
        inference_model = None # Ensure it's None if loading fails
        tokenizer = None
        return False

def generate_model_output(fact_pattern_text):
    """
    Generates a response from the fine-tuned model for a given fact pattern.
    """
    if inference_model is None or tokenizer is None:
        print("Model not loaded. Skipping generation.")
        return "Error: Model not loaded or loading failed"

    # Construct the prompt in the format the model expects
    # This should match the format used during fine-tuning
    instruction = "Determine if the following individual qualifies as a dependent under IRS §152."
    prompt = (
        f"### Instruction:\n{instruction}\n"
        f"### Input:\n{fact_pattern_text}\n"
        f"### Response:\n"
    )

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length # Max length for the input prompt
        ).to(inference_model.device)

        print(f"Generating response for input: '{fact_pattern_text[:100]}...'")
        with torch.no_grad(): # Disable gradient calculations for inference
            outputs = inference_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=1200, # This parameter controls the max length of the generated output
                do_sample=False,     # Use greedy decoding (no sampling)
                temperature=0.0,     # Effectively ignored when do_sample=False, but set low
                top_p=1.0,           # Effectively ignored when do_sample=False
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        response_only = decoded_output.strip()
            
        # print(f"Full raw generated output (after input): '{tokenizer.decode(outputs[0], skip_special_tokens=True)}'")
        print(f"Extracted response (decoded generated tokens only): '{response_only}'")
        return response_only

    except Exception as e:
        print(f"Error during model generation for '{fact_pattern_text[:50]}...': {e}")
        return "Error: Generation failed"

def create_summary_table():
    """
    Creates a list of dictionaries representing the fact pattern evaluation summary,
    gets model outputs dynamically, processes them to Yes/No/Non-Conforming,
    and saves it to a JSON file.
    """
    fact_patterns_data = [
        {
            "id": "Case 1",
            "text": "Maria is 20 years old, a full-time university student, and lives with her parents, David and Linda, for the entire year. Maria earned $3,000 from a part-time job. David and Linda provide over 75% of her financial support. Maria has not filed a joint tax return.",
            "SMT Verdict": "Yes",
        },
        {
            "id": "Case 2",
            "text": "John's mother, Alice, is 75 years old and lives in an assisted living facility. Alice's gross income for the year is $2,500 from a small pension. John provides over 60% of Alice's total support for the year, including the costs of the facility. Alice is a U.S. citizen and is not a qualifying child of any taxpayer.",
            "SMT Verdict": "Yes",
        },
        {
            "id": "Case 3",
            "text": "Michael is 25 years old, not a student, and lives with his sister, Sarah, for the entire year. Michael earned $10,000 from a full-time job. Sarah provides about 40% of Michael's support, with Michael covering the rest. Michael has not filed a joint return.",
            "SMT Verdict": "No",
        },
        {
            "id": "Case 4",
            "text": "Kevin is 18 years old and lives with his parents. He graduated high school and started a full-time job, earning $25,000. Kevin uses his earnings to pay for all his own expenses, including rent to his parents, food, and clothing, covering 100% of his own support. He has not filed a joint return.",
            "SMT Verdict": "No",
        },
        {
            "id": "Case 5",
            "text": "Laura is 22, a full-time student, lives with her parents, and her parents provide all her support. Laura got married during the year to Tom, and they filed a joint tax return solely to claim a refund of income tax withheld, as neither would have had any tax liability if they had filed separately.",
            "SMT Verdict": "Yes",
        },
        {
            "id": "Case 6",
            "text": "David (68) lives alone and has an annual income of $3,000. His three children, Amy, Beth, and Charles, collectively provide his support. Amy contributes 40%, Beth 30%, and Charles 30%. No single child provides more than half. David is not a qualifying child of anyone. All are US citizens. Can Amy claim David as a dependent?",
            "SMT Verdict": "Ambiguous",
        }
    ]

    summary_output_data = []
    model_loaded_successfully = load_model_and_tokenizer()

    for item in fact_patterns_data:
        print(f"\nProcessing {item['id']}...")
        
        raw_model_output_text = "Error: Model not run prior to processing" # Initial default
        if model_loaded_successfully:
            raw_model_output_text = generate_model_output(item["text"])
        elif inference_model is None and os.path.exists(lora_model_path):
             raw_model_output_text = "Error: Model loading failed"
        elif not os.path.exists(lora_model_path):
             raw_model_output_text = "Error: Model not found at path"

        # --- Process raw model output to derive "Yes", "No", or specific error/non-conforming status ---
        # This `final_answer_for_table` is now primarily for the Status calculation
        final_answer_for_table = "Non-Conforming Output" # Default for actual model responses

        if raw_model_output_text.startswith("Error:"):
            final_answer_for_table = raw_model_output_text # Propagate specific error messages
        else:
            temp_normalized_output = raw_model_output_text.lower().strip()
            # Check if the response starts with "yes" or "no" as a whole word.
            if re.match(r"^\s*\byes\b.*", temp_normalized_output):
                final_answer_for_table = "Yes"
            elif re.match(r"^\s*\bno\b.*", temp_normalized_output):
                final_answer_for_table = "No"
            # If neither, it remains "Non-Conforming Output"

        # --- Determine status based on `final_answer_for_table` (the processed output) ---
        status = "Error" # Default status
        
        if item["SMT Verdict"].lower() == "ambiguous":
            status = "Ambiguous"
        elif final_answer_for_table == "Non-Conforming Output":
            status = "Error (Non-Conforming Model Output)"
        elif final_answer_for_table.startswith("Error:"): # Catches errors from model loading/generation
            status = "Error (Model Operation Issue)"
        elif final_answer_for_table.lower() == item["SMT Verdict"].lower(): # Handles "Yes"=="Yes", "No"=="No"
            status = "Correct"
        else: 
            # This case means final_answer_for_table is "Yes" and SMT is "No", or vice-versa.
            status = "Error (Content Mismatch)"

        summary_output_data.append({
            "Fact Pattern": item["id"] + ": " + item["text"],
            # MODIFIED LINE: Use raw_model_output_text for the "Model Output" field
            "Model Output": raw_model_output_text,
            "SMT Verdict": item["SMT Verdict"],
            "Status": status
            # The commented-out "Raw Model Response" is no longer strictly necessary here,
            # as "Model Output" now serves this purpose.
        })

    output_filename = "summary_table_with_model_outputs.json"
    with open(output_filename, 'w') as f:
        json.dump(summary_output_data, f, indent=4)

    print(f"\nSuccessfully created '{output_filename}'")
    if not model_loaded_successfully:
        print("WARNING: Summary table generated with placeholder or error messages for Model Outputs due to issues loading/running the model.")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Model loading/inference might be very slow or fail if GPU is required.")
    create_summary_table()