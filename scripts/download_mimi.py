import os
from huggingface_hub import hf_hub_download

def download_mimi():
    repo_id = "nvidia/personaplex-7b-v1"
    filename = "tokenizer-e351c8d8-checkpoint125.safetensors"
    local_dir = "models"
    local_filename = "mimi.safetensors"

    print(f"Downloading {filename} from {repo_id}...")

    os.makedirs(local_dir, exist_ok=True)

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
    )

    # Rename to mimi.safetensors for consistency in Elara
    target_path = os.path.join(local_dir, local_filename)
    if os.path.exists(target_path):
        os.remove(target_path)
    os.rename(path, target_path)

    print(f"Mimi model downloaded to {target_path}")

if __name__ == "__main__":
    download_mimi()
