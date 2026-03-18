bebraAI project archive

Files included:
- main.py          (FastAPI inference + queue + local RAG)
- knowledge.txt    (your small KB)
- data.txt         (small example Q/A pairs)
- prepare_data.py  (creates train.jsonl from data.txt)
- finetune_peft.py (template for LoRA fine-tuning)
- requirements.txt

Quick start (local):
1) python -m venv venv && source venv/bin/activate
2) pip install -r requirements.txt
3) python prepare_data.py  # creates train.jsonl
4) uvicorn main:app --host 0.0.0.0 --port 8000

Notes:
- main.py uses tiny model 'sshleifer/tiny-gpt2' lazily (downloads on first request).
- For real fine-tuning use GPU, accelerate and finetune_peft.py template (edit BASE_MODEL).
- Do NOT commit secrets (no API keys in repo).
