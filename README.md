# Phi-1.5 Local Fine-Tuning (LoRA)

A clean, reproducible template to **fine-tune `microsoft/phi-1_5` locally** with LoRA (PEFT) using Hugging Face `transformers`, `datasets`, and (optionally) 4-bit loading via `bitsandbytes`. This repo packages a working training notebook plus a robust CLI training script.


## Quickstart
```bash
git clone https://github.com/mannybuff/phi15-finetune-lora.git
cd phi15-finetune-lora

python -m venv .venv && source .venv/bin/activate
# (Windows) .venv\Scripts\activate

pip install -r requirements.txt

# Put your data
# data/train.jsonl, data/val.jsonl  (see data/README.md for schema)
# Then run training (adjust flags as needed):
bash scripts/train.sh

# Inference
bash scripts/infer.sh
```
If you prefer `accelerate`, run `accelerate config` and set `--device_map auto`/`--mixed_precision` accordingly.

## Project layout
```
phi15-finetune-lora/
├── src/
│   ├── train_phi15_lora.py      # training (LoRA)
│   └── infer.py                 # quick generation helper
├── notebooks/
│   └── phi15_local_finetune.ipynb  # cleaned training notebook (your original, polished)
├── data/                        # put JSON/JSONL here (see data/README.md)
├── results/                     # model checkpoints and logs
├── config/train_config.yaml     # example config
├── scripts/train.sh             # convenience wrapper
├── scripts/infer.sh
├── requirements.txt
├── .gitignore
├── LICENSE (MIT)
└── README.md
```

## Tuning checklist for “mixed results”
- **Data**: ≥5–20k high-quality pairs often beats 500 noisy pairs; prefer *instruction → response* style if you prompt that way.
- **Prompt formatting**: Consistent templates (“Instruction/Input/Response”) reduce drift.
- **Steps & LR**: Start with ~500–3k steps; watch **loss curve**; too high LR → instability, too low → underfit.
- **LoRA rank**: `r=16` is a good start; bump to 32 for more capacity if VRAM allows.
- **4-bit**: Enable `--use_4bit` to fit on smaller GPUs; disable when you can to improve stability.
- **Eval**: Add a small **held‑out dev set** and simple metrics/SxS comparisons; qualitative checks matter.

## Notes
- This trains **only LoRA adapters**; base weights remain frozen (safer & faster).
- Tokenizer: set `pad_token = eos_token` if needed and use right padding.
- Target modules for LoRA are discovered heuristically to be robust across transformer versions.

## License
MIT © Manuel Buffa, 2025
