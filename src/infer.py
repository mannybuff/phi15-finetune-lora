
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    inputs = tok(args.prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
    print(tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
