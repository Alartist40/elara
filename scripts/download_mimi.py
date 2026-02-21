import os
import shutil
import time
import logging
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_with_retry(repo_id, filename, local_dir, max_retries=3):
    """Download a file from HuggingFace with exponential backoff retries."""
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}: Downloading {filename} from {repo_id}...")
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
            )
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to download after {max_retries} attempts: {e}")
                raise
            wait_time = 2 ** attempt
            logger.warning(f"Download failed: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

def main():
    repo_id = "nvidia/personaplex-7b-v1"
    filename = "tokenizer-e351c8d8-checkpoint125.safetensors"
    local_dir = "models"
    local_filename = "mimi.safetensors"

    os.makedirs(local_dir, exist_ok=True)

    try:
        path = download_with_retry(repo_id, filename, local_dir)

        # Move/Rename to mimi.safetensors for consistency in Elara
        target_path = os.path.join(local_dir, local_filename)
        if os.path.exists(target_path):
            os.remove(target_path)

        shutil.move(path, target_path)
        logger.info(f"Mimi model successfully downloaded and moved to {target_path}")
    except Exception as e:
        logger.error(f"Mimi download failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
