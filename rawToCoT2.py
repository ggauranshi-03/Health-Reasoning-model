# ==============================================================
#  Tabular → CoT Reasoning Dataset
# ==============================================================

import json
import re

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------
df = pd.read_csv("indian_liver_patient.csv")
df = df[
    [
        "Age",
        "Gender",
        "Total_Bilirubin",
        "Direct_Bilirubin",
        "Alkaline_Phosphotase",
        "Dataset",
    ]
].dropna()
df["tabular_row"] = df.apply(lambda r: r.to_dict(), axis=1).astype(str)
print(f"Loaded {len(df)} rows")

data = Dataset.from_pandas(df[["tabular_row"]])

# --------------------------------------------------------------
# 2. Model: Phi-3-mini-4k-instruct
# --------------------------------------------------------------
model_name = "microsoft/Phi-3-mini-4k-instruct"  

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_cfg,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# --------------------------------------------------------------
# 3. Prompt (Phi-3 format; adjust for Mistral if switching)
# --------------------------------------------------------------
PROMPT = """
<|system|>You are a bilingual Indian rural doctor (Hindi + English). Convert patient data to Chain-of-Thought Q&A. Always use exact format. Example:
Data: {{'Age': 45, 'Gender': 'Male', 'Total_Bilirubin': 2.5, 'Direct_Bilirubin': 1.2, 'Alkaline_Phosphotase': 300, 'Dataset': 1}}

Question: 45 वर्षीय पुरुष में बिलीरुबिन 2.5 – यकृत समस्या? (45yo male, bilirubin 2.5 – liver issue?)

Reasoning:
Step 1: Age 45, male – common for fatty liver in India.
Step 2: Total bilirubin 2.5 elevated (normal 0.3-1.2), direct 1.2 suggests obstruction.
Step 3: ALP 300 high – biliary issue; rural diet (oily food) risk.
Step 4: Ayurveda: Kutki for detox; modern: Ultrasound.

Final Answer: LFT repeat, avoid oil, consult PHC. (LFT दोहराएं, तेल कम करें, PHC जाएं।)<|end|>

Now convert: Data: {row}<|end|>
<|user|>Output exactly:
Question: <Hindi (English)>
Reasoning:
Step 1: <vitals>
Step 2: <differential/India prevalence>
Step 3: <risks/cultural>
Step 4: <Ayurveda + modern>
Final Answer: <advice + Hindi><|end|>
<|assistant|>
"""


# --------------------------------------------------------------
# 4. Manual Generation + Parse
# --------------------------------------------------------------
def extract(txt, pat):
    m = re.search(pat, txt, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else "N/A"


def make_cot(batch):
    batch_size = len(batch["tabular_row"])
    outs = [None] * batch_size
    skip_count = 0
    for i, row in enumerate(batch["tabular_row"]):
        row_escaped = row.replace("{", "{{").replace("}", "}}")
        prompt = PROMPT.format(row=row_escaped)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,  
                temperature=0.3,  # Lower for consistency
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        q = extract(gen, r"Question:\s*(.*?)\nReasoning:")
        rea = extract(gen, r"Reasoning:\s*(.*?)\nFinal Answer:")
        ans = extract(gen, r"Final Answer:\s*(.*)")

        full_out = (
            f"{rea}\n\nFinal Answer: {ans}" if rea != "N/A" and ans != "N/A" else gen
        )
        instruction = q if q != "N/A" else "Patient liver assessment query."

        # Skip only if <150 chars or no "Step"
        if len(full_out) < 150 or "Step" not in full_out:
            print(f"Skipping bad output for row {i}: {row[:50]}...")  # Log
            skip_count += 1
            outs[i] = {"instruction": "", "output": ""}  # Dummy to maintain length
            continue

        outs[i] = {"instruction": instruction, "output": full_out}

    print(f"Processed batch: {batch_size - skip_count} good, {skip_count} skipped")
    return {"cot": outs}


# --------------------------------------------------------------
# 5. Run (20 rows test)
# --------------------------------------------------------------
test_data = data.select(range(min(20, len(data))))
converted = test_data.map(make_cot, batched=True, batch_size=4)

cot_df = pd.DataFrame(converted["cot"])
# Filter out dummies before saving
cot_df = cot_df[cot_df["output"] != ""]  # Remove empty/skipped

cot_df.to_json(
    "reasoning_cot_dataset2.jsonl", orient="records", lines=True, force_ascii=False
)

print(f"\nSaved {len(cot_df)} good examples (after filtering)")
print("\n--- Sample ---")
print(json.dumps(cot_df.iloc[0].to_dict(), indent=2, ensure_ascii=False))
