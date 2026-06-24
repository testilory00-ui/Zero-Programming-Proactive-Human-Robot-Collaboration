from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig

model_name = "Qwen/Qwen3-4B-Instruct-2507"

# Configura quantizzazione INT4
quantization_config = OVWeightQuantizationConfig(bits=8, sym=True, group_size=-1)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Carica e converte il modello
model = OVModelForCausalLM.from_pretrained(
    model_name,
    export=True,                     # converte la prima volta
    quantization_config=quantization_config,
    trust_remote_code=True,
    device="AUTO"
)

# Salva modello già convertito in locale
model.save_pretrained("Qwen3_4B_INT8")
tokenizer.save_pretrained("Qwen3_4B_INT8")

print("Modello salvato!")