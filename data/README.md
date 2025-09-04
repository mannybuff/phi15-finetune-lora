# Data format

Use JSONL (one object per line) or JSON list with one of these schemas:

**Instruct format**
```json
{"instruction": "Summarize this text.", "input": "Lorem ipsum...", "output": "..."}
```

**Prompt/response format**
```json
{"prompt": "Write a haiku about data.", "response": "..."}
```

Place your files under `data/`. Example:
- `data/train.jsonl`
- `data/val.jsonl` (optional)
