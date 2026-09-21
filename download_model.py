"""下载 Qwen2.5-1.5B-Instruct GGUF (Q4_K_M) 到 ./models/。"""
import os
from huggingface_hub import hf_hub_download

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
LOCAL_DIR = os.getenv("MODEL_DIR", "./models")

path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    local_dir=LOCAL_DIR,
)
print(f"OK: {path}")
