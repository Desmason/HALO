# hf_lora_to_gguf.py

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, PeftModel


class HfLoraFinetuneError(RuntimeError):
    pass


def _load_text_dataset(txt_path: str, block_size: int = 1024) -> Dataset:
    """
    Very simple text loader: reads a .txt file and chunks it into blocks
    of tokens for causal LM training.
    """
    txt_path = str(Path(txt_path).expanduser().resolve())
    if not Path(txt_path).exists():
        raise FileNotFoundError(f"Training data not found at: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    # naive split into paragraphs; you can customize later
    samples = [s for s in text.split("\n\n") if s.strip()]
    return Dataset.from_dict({"text": samples})


def train_lora_adapter(
    base_model_id: str,
    train_txt_path: str,
    adapter_out_dir: str,
    *,
    hf_token: Optional[str] = None,
    max_seq_len: int = 1024,
    batch_size: int = 1,
    grad_accum_steps: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 1e-4,
    save_steps: int = 1000,
    logging_steps: int = 50,
    use_4bit: bool = False,
) -> str:
    """
    Fine-tune a HuggingFace causal LM with LoRA and save a PEFT adapter.

    Returns:
        adapter_out_dir (str): path to directory containing adapter_model.*
    """

    use_mps = torch.backends.mps.is_available()

    adapter_dir = Path(adapter_out_dir).expanduser().resolve()
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        use_auth_token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model loading options
    model_kwargs: Dict[str, Any] = {
        "use_auth_token": hf_token,
    }

    if use_4bit:
        # QLoRA-style loading (requires bitsandbytes, may not work on CPU-only / M1)
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        **model_kwargs,
    )

    # Set up LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

       # Load dataset
    raw_ds = _load_text_dataset(train_txt_path, block_size=max_seq_len)

    ASSISTANT_MARKER = "Assistant:"

    def preprocess_example(example):
        """
        Turn a single text block like

        <SFT>
        User: ...
        Assistant: ...

        into:
          - input_ids: prompt (User + 'Assistant:') + answer
          - labels:    -100 on prompt tokens, real ids on answer tokens
        """
        text = example["text"]

        # Find the Assistant part
        if ASSISTANT_MARKER in text:
            idx = text.index(ASSISTANT_MARKER)
            idx_after = idx + len(ASSISTANT_MARKER)
            # Prompt includes everything up THROUGH the 'Assistant:' marker
            prompt_text = text[:idx_after]
            # Answer is everything after 'Assistant:'
            answer_text = text[idx_after:]
        else:
            # Fallback: if no Assistant: marker, treat whole thing as prompt (no loss)
            prompt_text = text
            answer_text = ""

        # Full text that the model actually sees
        full_text = prompt_text + answer_text

        # Tokenize full text (this is what will be fed to the model)
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
        )

        input_ids = tokenized["input_ids"]

        # Tokenize only the prompt to know how many tokens to ignore in loss
        prompt_tokens = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,
        )["input_ids"]

        prompt_len = len(prompt_tokens)

        # Build labels: copy input_ids, but mask prompt tokens with -100
        labels = input_ids.copy()
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100  # ignored by CrossEntropyLoss

        tokenized["labels"] = labels
        return tokenized

    # map over dataset example-by-example (not batched)
    tokenized_ds = raw_ds.map(
        preprocess_example,
        remove_columns=["text"],
    )
    tokenized_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    # Just pad and batch; do NOT touch labels
    data_collator = default_data_collator


    training_args = TrainingArguments(
        output_dir=str(adapter_dir / "hf_trainer_output"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=1,
        bf16= False,
        fp16= False,  # on CPU fp16 won't actually be used
        optim="adamw_torch",
        report_to=[],
    )

    # Disable fp16 mixed precision for accelerate (important on MPS / CPU)
    os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "no")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )

    trainer.train()

    # Save only the LoRA adapter weights
    model: PeftModel
    model.save_pretrained(str(adapter_dir))

    return str(adapter_dir)


def convert_lora_dir_to_gguf_adapter(
    llama_cpp_dir: str,
    lora_dir: str,
    *,
    base_model_id: Optional[str] = None,
    base_config_dir: Optional[str] = None,
    outfile: Optional[str] = None,
    outtype: str = "f16",
) -> str:
    """
    Wraps llama.cpp's convert_lora_to_gguf.py.

    You must have llama.cpp checked out, and this script inside it.

    Args:
        llama_cpp_dir: path to llama.cpp repo root.
        lora_dir:      directory containing adapter_model.json + adapter_model.(safetensors|bin)
        base_model_id: HF model id for base (e.g. meta-llama/Llama-3.1-8B-Instruct)
        base_config_dir: local directory with base HF config (config.json, tokenizer.json);
                         used if base_model_id is None.
        outfile:       final .gguf adapter path. If None, placed next to lora_dir.
        outtype:       "f32", "f16", "bf16", "q8_0", or "auto"
    """
    llama_cpp_dir = str(Path(llama_cpp_dir).expanduser().resolve())
    lora_dir = str(Path(lora_dir).expanduser().resolve())

    if outfile is None:
        outfile_path = Path(lora_dir) / "adapter.gguf"
    else:
        outfile_path = Path(outfile).expanduser().resolve()
        outfile_path.parent.mkdir(parents=True, exist_ok=True)

    script_path = Path(llama_cpp_dir) / "convert_lora_to_gguf.py"
    if not script_path.exists():
        raise HfLoraFinetuneError(
            f"convert_lora_to_gguf.py not found at {script_path}. "
            "Make sure your llama.cpp checkout matches the GitHub repo."
        )

    cmd = [
        sys.executable,
        str(script_path),
        "--outfile", str(outfile_path),
        "--outtype", outtype,
    ]

    if base_model_id is not None:
        cmd += ["--base-model-id", base_model_id]
    elif base_config_dir is not None:
        cmd += ["--base", str(Path(base_config_dir).expanduser().resolve())]
    # else: script will try to read base_model_name_or_path from adapter_config.json

    cmd.append(lora_dir)

    proc = subprocess.run(
        cmd,
        cwd=llama_cpp_dir,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise HfLoraFinetuneError(
            f"convert_lora_to_gguf.py failed with code {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )

    if not outfile_path.exists():
        raise HfLoraFinetuneError(
            f"Expected GGUF adapter not found at {outfile_path} after conversion."
        )

    return str(outfile_path)


def hf_lora_finetune_to_gguf_adapter(
    hf_model_id: str,
    train_txt_path: str,
    llama_cpp_dir: str,
    *,
    hf_token: Optional[str] = None,
    out_dir: str = "finetuned_adapters",
    max_seq_len: int = 1024,
    batch_size: int = 1,
    grad_accum_steps: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 1e-4,
    use_4bit: bool = False,
    outtype: str = "f16",
) -> str:
    """
    Convenience function:

    1) LoRA-finetune HF model `hf_model_id` on `train_txt_path` → PEFT adapter dir
    2) Run convert_lora_to_gguf.py → GGUF LoRA adapter

    Returns:
        path to GGUF LoRA adapter file (.gguf) usable with llama.cpp.
    """
    out_dir_path = Path(out_dir).expanduser().resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    adapter_dir = out_dir_path / hf_model_id.replace("/", "_") / "peft_lora"
    adapter_dir_str = train_lora_adapter(
        base_model_id=hf_model_id,
        train_txt_path=train_txt_path,
        adapter_out_dir=str(adapter_dir),
        hf_token=hf_token,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        use_4bit=use_4bit,
    )

    gguf_adapter_out = out_dir_path / hf_model_id.replace("/", "_") / "adapter.gguf"
    gguf_adapter_path = convert_lora_dir_to_gguf_adapter(
        llama_cpp_dir=llama_cpp_dir,
        lora_dir=adapter_dir_str,
        base_model_id=hf_model_id,
        outfile=str(gguf_adapter_out),
        outtype=outtype,
    )

    return gguf_adapter_path
