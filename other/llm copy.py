from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM
import json
import time


# Load context from a JSON file.
# This assumes a file named 'combined_context.json' exists in the same directory.
# If your file has a different name or is in another folder, update the path below.
with open('carburetor_assembly.json', 'r', encoding='utf-8') as f:
    memory = json.load(f)

with open('scene2.json', 'r', encoding='utf-8') as f:
    scene = json.load(f)

# Convert the Python dictionary to a JSON string for the prompt
assembly = json.dumps(memory, indent=2)
scene = json.dumps(scene, indent=2)

model_name = "qwen3_4B_INT4"

print("Loading model and compiling for GPU (this may take a moment)...")
start_setup = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)
model = OVModelForCausalLM.from_pretrained(model_name, device="GPU")
print(f"Setup Time (Load & Compile): {time.time() - start_setup:.2f} seconds")

system_prompt = """You are an expert industrial task planner.
Your goal is to determine the current state of the assembly process based on the provided scene and assembly steps.
You must output the result in strict JSON format without any additional text or explanations."""

# user_prompt = f"""
# ASSEMBLY DEFINITION:
# {assembly}

# CURRENT SCENE:
# {scene}

# Task:
# 1. Analyze the 'semantic action' in the CURRENT SCENE.
# 2. Match these with the 'step description' and 'objects_involved' in the ASSEMBLY DEFINITION to find the CURRENT step.
# 3. Identify the NEXT operation (CURRENT step number + 1). If the CURRENT step is the last one, the NEXT step is step 1.
# 4. Set OBJECTS REQUIRED: objects involved in the NEXT operation.
# 5. Find objects to bring: 'remote objects' in CURRENT SCENE that are also in REQUIRED
# 6. Find objects to remove: 'shared objects' in CURRENT SCENE that are NOT in REQUIRED

# Return ONLY a JSON object with this structure:
# {{
#   "stage of assembly": "<current step description>",
#   "next operation": "<next step description>",
#   "objects required": [list of objects involved in NEXT]
#   "objects to bring": [list of objects to bring],
#   "objects to remove": [list of objects to remove]
# }}
# """

user_prompt = f"""
ASSEMBLY DEFINITION:
{assembly}

CURRENT SCENE:
{scene}

Task:
1. Analyze the 'semantic action' in the CURRENT SCENE.
2. Match this with the 'step description' and 'objects_involved' in the ASSEMBLY DEFINITION to find the CURRENT step.
3. Identify the NEXT operation (CURRENT step number + 1). If the CURRENT step is the last one in ASSEMBLY DEFINITION, the NEXT step is step 1.
4. Set OBJECTS REQUIRED: objects involved in NEXT operation.

Return ONLY a JSON object with this structure:
{{
  "stage of assembly": "<current step description>",
  "next operation": "<next step description>",
  "objects required": [list of objects involved in NEXT]
}}

Do not show the thinking process.
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
)

print("Generating response...")
start_inf = time.time()
outputs = model.generate(**inputs, max_new_tokens=150)
print(f"Inference Time: {time.time() - start_inf:.2f} seconds")
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))