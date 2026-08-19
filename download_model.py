import os
import sys

MODEL_DIR = "./models"
MODEL_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
REPO_ID = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)


def download():
    if os.path.exists(MODEL_PATH):
        size_gb = os.path.getsize(MODEL_PATH) / (1024 ** 3)
        print(f"Model already exists at {MODEL_PATH} ({size_gb:.1f} GB)")
        return MODEL_PATH

    print(f"Downloading {MODEL_FILENAME} from HuggingFace...")
    print("File size: ~0.6 GB — this will take a minute on a good connection.\n")

    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILENAME,
            local_dir=MODEL_DIR,
        )
        print(f"\nModel downloaded successfully to: {path}")
        return path
    except Exception as e:
        print(f"\nERROR: Download failed: {e}")
        print("\n--- MANUAL DOWNLOAD INSTRUCTIONS ---")
        print("If HuggingFace is blocked, download the model manually on another machine:")
        print(f"  URL: https://huggingface.co/{REPO_ID}/resolve/main/{MODEL_FILENAME}")
        print(f"  Then copy the file to: {os.path.abspath(MODEL_PATH)}")
        print("-------------------------------------")
        sys.exit(1)


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    download()