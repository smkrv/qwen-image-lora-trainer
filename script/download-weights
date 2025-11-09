#!/usr/bin/env python3
"""Download Qwen Image base model weights during build time"""

import os
import subprocess
import sys
from pathlib import Path

MODEL_CACHE = "model_cache"
BASE_URL = "https://weights.replicate.delivery/default/qwen-image-lora/model_cache/"

# Set environment variables for model caching
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

def download_weights(url: str, dest: str) -> None:
    """Download weights from CDN using pget"""
    print(f"[!] Downloading from URL: {url}")
    print(f"[~] Destination path: {dest}")
    
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
        print("[+] Download completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to download weights. Command returned exit status {e.returncode}.")
        sys.exit(1)

def main():
    # Create model cache directory
    os.makedirs(MODEL_CACHE, exist_ok=True)
    
    # Model files to download
    model_files = [
        "models--Qwen--Qwen-Image.tar",
        "xet.tar",
    ]
    
    for model_file in model_files:
        url = BASE_URL + model_file
        filename = url.split("/")[-1]
        dest_path = os.path.join(MODEL_CACHE, filename)
        
        # Check if the extracted directory exists (without .tar extension)
        extracted_name = filename.replace(".tar", "")
        extracted_path = os.path.join(MODEL_CACHE, extracted_name)
        
        if os.path.exists(extracted_path):
            print(f"[✓] Model already exists at {extracted_path}, skipping download")
            continue
        
        print(f"\n[~] Downloading {model_file}...")
        download_weights(url, dest_path)
    
    print("\n[✓] All model weights downloaded successfully!")

if __name__ == "__main__":
    main()
