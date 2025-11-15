# ==============================================================
#  Tabular → CoT Reasoning Dataset (FIXED: Manual Generate)
# ==============================================================

import pandas as pd, json, re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# --------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------
df = pd.read_csv('indian_liver_patient.csv')
df = df[['Age','Gender','Total_Bilirubin','Direct_Bilirubin',
         'Alkaline_Phosphotase','Dataset']].dropna()
df['tabular_row'] = df.apply(lambda r: r.to_dict(), axis=1).astype(str)
print(f"Loaded {len(df)} rows")

data = Dataset.from_pandas(df[['tabular_row']])

# --------------------------------------------------------------
# 2. Model: Phi-3-mini-4k-instruct (or swap to Mistral if downloaded)
# --------------------------------------------------------------
model_name = "microsoft/Phi-3-mini-4k-instruct"  # Keep Phi-3 (fast); or "mistralai/Mistral-7B-Instruct-v0.3"

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_cfg,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# --------------------------------------------------------------
# 3. Prompt (Phi-3 format; adjust for Mistral if switching)
# --------------------------------------------------------------
PROMPT = """
<|system|>You are a bilingual Indian rural doctor (Hindi + English). Convert the patient data into a Chain-of-Thought Q&A.<|end|>
<|user|>Data: {row}

Output exactly:
Question: <Hindi (English)>
Reasoning:
Step 1: ...
Step 2: ...
Step 3: ...
Step 4: ...
Final Answer: <advice + Hindi><|end|>
<|assistant|>
"""

# --------------------------------------------------------------
# 4. Manual Generation + Parse (FIXES Cache Error)
# --------------------------------------------------------------
def extract(txt, pat):
    m = re.search(pat, txt, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else "N/A"

def make_cot(batch):
    outs = []
    for row in batch['tabular_row']:
        prompt = PROMPT.format(row=row)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=260,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        gen = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        q   = extract(gen, r'Question:\s*(.*?)\nReasoning:')
        rea = extract(gen, r'Reasoning:\s*(.*?)\nFinal Answer:')
        ans = extract(gen, r'Final Answer:\s*(.*)')

        full_out = f"{rea}\n\nFinal Answer: {ans}" if rea != "N/A" else gen

        outs.append({"instruction": q, "output": full_out})
    return {"cot": outs}

# --------------------------------------------------------------
# 5. Run (20 rows test)
# --------------------------------------------------------------
test_data = data.select(range(min(20, len(data))))
converted = test_data.map(make_cot, batched=True, batch_size=4)

cot_df = pd.DataFrame(converted['cot'])
cot_df.to_json("reasoning_cot_dataset.jsonl", orient="records", lines=True, force_ascii=False)

print(f"\nSaved {len(cot_df)} examples")
print("\n--- Sample ---")
print(json.dumps(cot_df.iloc[0].to_dict(), indent=2, ensure_ascii=False))