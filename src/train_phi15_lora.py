
import os, math, json, argparse, warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType

warnings.filterwarnings("ignore", category=UserWarning)

PROMPT_TPL = "{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
PROMPT_TPL_NO_INPUT = "{instruction}\n\n### Response:\n"

def format_example(ex: Dict[str, Any]) -> str:
    # Support common keys: instruction/input/output OR prompt/response
    if "instruction" in ex and "output" in ex:
        if ex.get("input"):
            return PROMPT_TPL.format(**ex)
        else:
            return PROMPT_TPL_NO_INPUT.format(**ex)
    elif "prompt" in ex and "response" in ex:
        return f"{ex['prompt']}\n\n### Response:\n"
    else:
        # fallback: join all fields for demo
        parts = [f"{k}: {v}" for k, v in ex.items()]
        return "\n".join(parts) + "\n\n### Response:\n"

def find_lora_targets(model) -> List[str]:
    """
    Heuristic: pick leaf Linear module names likely to be attention/MLP projections.
    Falls back to all Linear leaves if nothing matches.
    """
    import torch.nn as nn
    leaf_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            leaf = name.split(".")[-1]
            leaf_names.append(leaf)
    # prefer common names
    preferred = [n for n in set(leaf_names) if any(x in n.lower() for x in ["q", "k", "v", "o", "proj", "gate", "up", "down", "fc"])]
    return list(sorted(set(preferred))) or list(sorted(set(leaf_names)))

def main():
    ap = argparse.ArgumentParser(description="LoRA fine-tune Phi-1.5 locally")
    ap.add_argument("--model_name", default="microsoft/phi-1_5")
    ap.add_argument("--train_file", type=str, required=True, help="Path to JSON/JSONL training data")
    ap.add_argument("--val_file", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default="results/phi15-lora")
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--use_4bit", action="store_true", help="Load in 4-bit with bitsandbytes")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    bnb_cfg = None
    device_map = "auto"
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)

    if args.use_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype or torch.float16,
        )
        torch_dtype = None  # handled by bnb
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        quantization_config=bnb_cfg,
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # PEFT LoRA
    target_modules = find_lora_targets(model)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Datasets
    ext = "jsonl" if args.train_file.endswith("jsonl") else "json"
    ds_train = load_dataset("json", data_files=args.train_file, split="train")
    ds_val = load_dataset("json", data_files=args.val_file, split="train") if args.val_file else None

    def tokenize_fn(ex):
        prompt = format_example(ex)
        text = prompt + (ex.get("output") or ex.get("response") or "")
        toks = tok(text, truncation=True, padding="max_length", max_length=1024)
        toks["labels"] = toks["input_ids"].copy()
        return toks

    ds_train = ds_train.map(tokenize_fn, remove_columns=ds_train.column_names)
    if ds_val is not None:
        ds_val = ds_val.map(tokenize_fn, remove_columns=ds_val.column_names)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args_train = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        weight_decay=args.wd,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="steps" if ds_val is not None else "no",
        eval_steps=100,
        save_total_limit=2,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        data_collator=collator,
        train_dataset=ds_train,
        eval_dataset=ds_val,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
