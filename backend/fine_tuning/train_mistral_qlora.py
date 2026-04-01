# ── CELL 1: Install Dependencies ─────────────────────────────────────────────
"""
Paste this into Colab Cell 1 and run it.
This installs Unsloth (which makes fine-tuning 2x faster with 60% less VRAM).
Takes about 3-4 minutes.
"""

CELL_1 = """
# Install Unsloth and dependencies
# Unsloth makes fine-tuning on free Colab GPU possible
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes -q
!pip install mistralai huggingface_hub datasets python-dotenv -q
print("✓ All dependencies installed")
"""

# ── CELL 2: Configuration ────────────────────────────────────────────────────
"""
Paste this into Colab Cell 2.
Fill in your HuggingFace token and username.
"""

CELL_2 = """
# ── Configuration ────────────────────────────────────────────────────────────
# Fill in YOUR values here before running

HF_TOKEN    = "hf_your_token_here"       # From https://huggingface.co/settings/tokens
HF_USERNAME = "your_hf_username"          # Your HuggingFace username

# These match config.py (single source of truth in the main project)
BASE_MODEL_NAME        = "mistralai/Mistral-7B-v0.1"
HF_REPO_ID             = f"{HF_USERNAME}/fincomply-mistral-qlora"
FINE_TUNE_MAX_SEQ_LENGTH = 2048
FINE_TUNE_BATCH_SIZE     = 2
FINE_TUNE_GRAD_ACCUM     = 4
FINE_TUNE_EPOCHS         = 3
FINE_TUNE_LR             = 2e-4
FINE_TUNE_LORA_R         = 16
FINE_TUNE_LORA_ALPHA     = 32
FINE_TUNE_LORA_DROPOUT   = 0.05
OUTPUT_DIR               = "/content/fincomply-checkpoints"

print(f"✓ Config set — model: {BASE_MODEL_NAME}")
print(f"✓ Will save adapter to: {HF_REPO_ID}")
"""

# ── CELL 3: Load Model with Unsloth ─────────────────────────────────────────
"""
Paste this into Colab Cell 3.
Loads Mistral-7B with 4-bit quantization (fits in free Colab GPU).
"""

CELL_3 = """
from unsloth import FastLanguageModel
import torch

print("Loading model... (this takes 3-5 minutes on first run)")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name  = BASE_MODEL_NAME,
    max_seq_length = FINE_TUNE_MAX_SEQ_LENGTH,
    dtype       = None,       # Auto-detect: float16 on T4, bfloat16 on A100
    load_in_4bit = True,      # 4-bit quantization = fits in 15GB VRAM (free T4)
)

print("✓ Base model loaded")

# Apply LoRA (Low-Rank Adaptation)
# This adds ~0.1% extra parameters that we actually train
# The 6.7 billion original parameters stay frozen → fast + cheap
model = FastLanguageModel.get_peft_model(
    model,
    r                = FINE_TUNE_LORA_R,
    target_modules   = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    lora_alpha       = FINE_TUNE_LORA_ALPHA,
    lora_dropout     = FINE_TUNE_LORA_DROPOUT,
    bias             = "none",
    use_gradient_checkpointing = "unsloth",   # Saves memory
    random_state     = 42,
    use_rslora       = False,
    loftq_config     = None,
)

print("✓ LoRA adapters applied")
print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
"""

# ── CELL 4: Load Dataset ─────────────────────────────────────────────────────
"""
Paste this into Colab Cell 4.
Upload your .jsonl dataset files to Colab Files panel first,
OR the script will create a small sample dataset for testing.
"""

CELL_4 = """
from datasets import load_dataset, Dataset
import json
import os

# ── Instruction tuning format ─────────────────────────────────────────────────
# Each example = {"instruction": "...", "input": "...", "output": "..."}
#
# instruction = the user's compliance question
# input       = regulatory context (from gov circular)
# output      = the structured JSON answer

ALPACA_PROMPT = \"\"\"You are FinComply AI, an Indian financial regulatory intelligence expert.

### Instruction:
{instruction}

### Input (Regulatory Context):
{input}

### Response:
{output}\"\"\"

EOS_TOKEN = tokenizer.eos_token  # Important: must add EOS token

def format_example(example):
    text = ALPACA_PROMPT.format(
        instruction = example.get("instruction", ""),
        input       = example.get("input", ""),
        output      = example.get("output", ""),
    ) + EOS_TOKEN
    return {"text": text}


# Try to load from file — if not present, use sample data
DATASET_PATH = "/content/gst_rbi_sebi_mca_dataset.jsonl"

if os.path.exists(DATASET_PATH):
    # Load your actual dataset
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"✓ Loaded dataset from file: {len(dataset)} examples")
else:
    # Sample dataset for testing the pipeline
    # In production: replace this with your real scraped data
    sample_data = [
        {
            "instruction": "What is the GST rate on residential apartments?",
            "input": "GST Notification 11/2017-CT(Rate) dated 28-06-2017 specifies construction services. Affordable residential apartments attract 1% GST. Other residential apartments attract 5% GST without ITC.",
            "output": '{"summary": "1% GST for affordable housing, 5% for other residential apartments", "circular_number": "Notification 11/2017-CT(Rate)", "source_url": "https://www.cbic.gov.in", "domain": "gst", "action_required": "Verify category (affordable vs other) to apply correct rate"}'
        },
        {
            "instruction": "What is the RBI digital lending guideline for interest rates?",
            "input": "RBI Circular RBI/2022-23/111 dated 10-08-2022 on digital lending. All-inclusive cost of digital loans must be disclosed as Annual Percentage Rate (APR). No additional hidden charges allowed.",
            "output": '{"summary": "All digital loan costs must be disclosed as APR with no hidden charges", "circular_number": "RBI/2022-23/111", "source_url": "https://www.rbi.org.in", "domain": "rbi", "action_required": "Disclose all fees as APR in loan agreement"}'
        },
        {
            "instruction": "What are SEBI disclosure requirements for listed companies?",
            "input": "SEBI LODR Regulations 2015 Regulation 29: Companies must notify stock exchange 2 days before board meeting where financial results, dividends, or buybacks are discussed.",
            "output": '{"summary": "Notify stock exchange 2 days before board meeting discussing financials or dividends", "circular_number": "SEBI LODR Regulation 29", "source_url": "https://www.sebi.gov.in", "domain": "sebi", "action_required": "File intimation on BSE/NSE portal 2 working days before board meeting"}'
        },
        {
            "instruction": "What is MCA annual filing deadline for private companies?",
            "input": "Companies Act 2013 Section 92 and Rule 11 of Companies (Management) Rules 2014. Annual Return (MGT-7) must be filed within 60 days of AGM. AOC-4 (financial statements) within 30 days.",
            "output": '{"summary": "MGT-7 within 60 days of AGM, AOC-4 within 30 days of AGM", "circular_number": "Companies Act 2013 Section 92", "source_url": "https://www.mca.gov.in", "domain": "mca", "action_required": "File MGT-7 and AOC-4 on MCA21 portal within deadlines"}'
        },
    ]
    dataset = Dataset.from_list(sample_data)
    print(f"✓ Using sample dataset: {len(dataset)} examples")
    print("  (Upload your real JSONL file to /content/ for production training)")


# Format all examples for instruction tuning
dataset = dataset.map(format_example, remove_columns=dataset.column_names)
print(f"✓ Dataset formatted. First example preview:")
print(dataset[0]["text"][:300])
"""

# ── CELL 5: Train ─────────────────────────────────────────────────────────────
"""
Paste this into Colab Cell 5.
This is the actual training. With 4 examples takes <5 min.
With your full dataset (~500 examples) takes ~30-45 min on T4.
"""

CELL_5 = """
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model                = model,
    tokenizer            = tokenizer,
    train_dataset        = dataset,
    dataset_text_field   = "text",
    max_seq_length       = FINE_TUNE_MAX_SEQ_LENGTH,
    dataset_num_proc     = 2,
    packing              = False,
    args = TrainingArguments(
        per_device_train_batch_size  = FINE_TUNE_BATCH_SIZE,
        gradient_accumulation_steps  = FINE_TUNE_GRAD_ACCUM,
        warmup_steps                 = 5,
        num_train_epochs             = FINE_TUNE_EPOCHS,
        learning_rate                = FINE_TUNE_LR,
        fp16                         = not is_bfloat16_supported(),
        bf16                         = is_bfloat16_supported(),
        logging_steps                = 1,
        optim                        = "adamw_8bit",
        weight_decay                 = 0.01,
        lr_scheduler_type            = "linear",
        seed                         = 42,
        output_dir                   = OUTPUT_DIR,
        report_to                    = "none",    # Disable wandb
    ),
)

print("Starting training...")
trainer_stats = trainer.train()
print(f"✓ Training complete!")
print(f"  Training loss: {trainer_stats.training_loss:.4f}")
print(f"  Runtime: {trainer_stats.metrics['train_runtime']:.0f} seconds")
"""

# ── CELL 6: Test the fine-tuned model ────────────────────────────────────────
"""
Paste this into Colab Cell 6.
Test the model before saving to HuggingFace.
"""

CELL_6 = """
# Switch to inference mode (faster)
FastLanguageModel.for_inference(model)

test_query = "What is the GST rate for software services?"
prompt = ALPACA_PROMPT.format(
    instruction = test_query,
    input       = "GST Notification 11/2017-CT(Rate) — Software services fall under SAC 9983. GST rate is 18%.",
    output      = "",   # Leave empty — model will fill this
)

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens = 300,
    use_cache      = True,
    temperature    = 0.1,
)
response = tokenizer.batch_decode(outputs)[0]
# Extract only the generated part (after "### Response:")
generated = response.split("### Response:")[-1].strip()
print("Model response:")
print(generated[:500])
"""

# ── CELL 7: Save adapter to HuggingFace Hub ──────────────────────────────────
"""
Paste this into Colab Cell 7.
Saves ONLY the LoRA adapter (tiny file ~50MB) to HuggingFace Hub.
NOT the full 7B model — just the small adapter that gets merged at inference.
"""

CELL_7 = """
from huggingface_hub import login

# Login to HuggingFace
login(token=HF_TOKEN)
print(f"✓ Logged in to HuggingFace")

# Save adapter to Hub
print(f"Uploading adapter to: {HF_REPO_ID} ...")
model.push_to_hub(
    HF_REPO_ID,
    token = HF_TOKEN,
)
tokenizer.push_to_hub(
    HF_REPO_ID,
    token = HF_TOKEN,
)
print(f"✓ Adapter saved!")
print(f"  View at: https://huggingface.co/{HF_REPO_ID}")
print()
print("NEXT STEP: Copy the repo ID above and paste it into your .env file as:")
print(f"  MISTRAL_FINE_TUNED_MODEL={HF_REPO_ID}")
"""

# ── HOW TO USE OUTPUT ─────────────────────────────────────────────────────────
print("""
============================================================
HOW TO USE THIS FILE IN GOOGLE COLAB
============================================================
1. Go to https://colab.research.google.com
2. Click File → New Notebook
3. Runtime → Change runtime type → GPU (T4 free tier)
4. Create 7 cells and paste CELL_1 through CELL_7 content
5. Run each cell top to bottom

The CELL_X = \"\"\" ... \"\"\" strings above contain the code.
Copy the content between the triple quotes into Colab.

IMPORTANT BEFORE RUNNING:
- In CELL 2: Replace HF_TOKEN and HF_USERNAME with your values
- Upload your .jsonl training data to Colab Files panel
  (or use the sample data for testing)
============================================================
""")