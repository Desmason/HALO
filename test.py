from llama_infer import LlamaInference

BASE = "converted/meta-llama_Llama-3.2-1B-Instruct.f16.gguf"
LORA = "finetuned_adapters/meta-llama_Llama-3.2-1B-Instruct/peft_lora/hf_trainer_output/checkpoint-9/meta-llama_Llama-3.2-1B-Instruct-lora-f16.gguf"  # adapter GGUF from convert-lora-to-gguf



prompt = "Explain the HALO system in 1 sentence."

base = LlamaInference(BASE)
lora  = LlamaInference(BASE, LORA)

b, _ = base.infer(prompt, max_tokens=64)
l, _ = lora.infer(prompt, max_tokens=64)

print("BASE:", b)
print("LORA:", l)
