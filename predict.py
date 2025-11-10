#!/usr/bin/env python3
"""Qwen Image predictor with optimized LoRA support and caching"""

import os
from pathlib import Path
import time

MODEL_CACHE = "model_cache"

# Set environment variables for model caching BEFORE any imports
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

import sys
import torch
import tempfile
import zipfile
import tarfile
import gzip
import shutil
import subprocess
import hashlib
from typing import Optional, List
from cog import BasePredictor, Input, Path as CogPath
from diffusers import DiffusionPipeline
from pathlib import Path as PathlibPath

from helpers.billing.metrics import record_billing_metric


# Aspect ratio configurations
ASPECT_RATIOS = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1136),
    "3:4": (1136, 1472),
    "3:2": (1536, 1024),
    "2:3": (1024, 1536),
}

# Speed dimensions for optimize_for_speed mode
SPEED_DIMENSIONS = {
    "1:1": (1024, 1024),
    "16:9": (1024, 576),
    "9:16": (576, 1280),
    "4:3": (1024, 768),
    "3:4": (768, 1024),
    "3:2": (1152, 768),
    "2:3": (768, 1152),
}


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download weights on first run if not present
        if not os.path.exists(MODEL_CACHE):
            print(f"Model cache directory '{MODEL_CACHE}' not found. Downloading weights on first run...")
            try:
                subprocess.run([sys.executable, "download_weights.py"], check=True)
                print("Weights downloaded successfully!")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to download model weights: {e}")
        
        print("Loading Qwen Image model from pre-downloaded weights...")
        start_time = time.time()
        
        # Load using optimized DiffusionPipeline
        self.pipe = DiffusionPipeline.from_pretrained(
            "Qwen/Qwen-Image",
            dtype=torch.bfloat16,
        )
        self.pipe.to("cuda")
        
        print(f"Model loaded in {time.time() - start_time:.2f}s")
        print("Model ready for inference!")
        
        # Initialize LoRA cache for maximum performance
        self.loaded_loras = {}
    
    def _get_lora_hash(self, lora_url: str) -> str:
        """Generate unique hash for LoRA URL"""
        return hashlib.md5(lora_url.encode()).hexdigest()[:12]
    
    def _detect_archive_type(self, url: str) -> Optional[str]:
        """Detect archive type from URL
        
        Returns:
            str: Archive type ('zip', 'tar', 'tar.gz', 'tar.bz2', 'tar.xz', 'gz') or None for direct file
        """
        url_lower = url.lower()
        
        if url_lower.endswith('.zip'):
            return 'zip'
        elif url_lower.endswith(('.tar.gz', '.tgz')):
            return 'tar.gz'
        elif url_lower.endswith(('.tar.bz2', '.tbz2')):
            return 'tar.bz2'
        elif url_lower.endswith(('.tar.xz', '.txz')):
            return 'tar.xz'
        elif url_lower.endswith('.tar'):
            return 'tar'
        elif url_lower.endswith('.gz') and not url_lower.endswith('.tar.gz'):
            return 'gz'
        elif url_lower.endswith('.safetensors'):
            return None  # Direct safetensors file
        else:
            return None  # Unknown format, treat as direct file
    
    def _extract_archive(self, archive_path: str, extract_dir: str, archive_type: str) -> None:
        """Extract archive based on type
        
        Args:
            archive_path: Path to the archive file
            extract_dir: Directory to extract to
            archive_type: Type of archive ('zip', 'tar', 'tar.gz', etc.)
        """
        os.makedirs(extract_dir, exist_ok=True)
        
        if archive_type == 'zip':
            with zipfile.ZipFile(archive_path, 'r') as archive:
                archive.extractall(extract_dir)
        
        elif archive_type in ('tar', 'tar.gz', 'tar.bz2', 'tar.xz'):
            # tarfile automatically handles compression detection with 'r:*' mode
            with tarfile.open(archive_path, 'r:*') as archive:
                archive.extractall(extract_dir)
        
        elif archive_type == 'gz':
            # Single gzip file - extract directly
            output_path = os.path.join(extract_dir, 'extracted.safetensors')
            with gzip.open(archive_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        else:
            raise ValueError(f"Unsupported archive type: {archive_type}")
    
    def _load_lora_weights(self, lora_path: str, adapter_name: str, lora_scale: float) -> Optional[str]:
        """Load LoRA weights with intelligent caching
        
        Returns:
            str: Unique adapter name for this LoRA, or None if no LoRA provided
        """
        if not lora_path:
            return None
        
        # Generate unique adapter name based on URL/path hash
        url_hash = self._get_lora_hash(lora_path)
        unique_adapter_name = f"{adapter_name}_{url_hash}"
        
        # Check if already loaded in pipe - maximum performance optimization
        if unique_adapter_name in self.loaded_loras:
            cached = self.loaded_loras[unique_adapter_name]
            print(f"✓ LoRA {adapter_name} (hash: {url_hash}) already loaded in pipe")
            print(f"  Cached at: {cached['path']}")
            # Return immediately without reloading for maximum performance
            return unique_adapter_name
        
        print(f"Loading LoRA {adapter_name} from {lora_path}")
        print(f"Using unique identifier: {url_hash}")
        
        # Download if URL, otherwise use local path
        if lora_path.startswith(("http://", "https://")):
            import requests
            
            # Detect archive type
            archive_type = self._detect_archive_type(lora_path)
            
            if archive_type:
                # Handle archive files
                ext_map = {
                    'zip': '.zip',
                    'tar': '.tar',
                    'tar.gz': '.tar.gz',
                    'tar.bz2': '.tar.bz2',
                    'tar.xz': '.tar.xz',
                    'gz': '.gz'
                }
                ext = ext_map.get(archive_type, '.archive')
                
                archive_path = f"/tmp/lora_{url_hash}{ext}"
                extract_dir = f"/tmp/lora_{url_hash}_extracted"
                
                # Check if already downloaded and extracted
                if not os.path.exists(extract_dir):
                    print(f"Downloading and extracting {archive_type.upper()} archive...")
                    # Download archive
                    response = requests.get(lora_path)
                    with open(archive_path, "wb") as f:
                        f.write(response.content)
                    
                    # Extract archive
                    self._extract_archive(archive_path, extract_dir, archive_type)
                    
                    # Clean up archive file after extraction
                    os.remove(archive_path)
                    print(f"{archive_type.upper()} archive extracted to {extract_dir}")
                else:
                    print(f"Using cached extracted files from {extract_dir}")
                
                # Find .safetensors file
                safetensors_files = list(PathlibPath(extract_dir).rglob("*.safetensors"))
                if not safetensors_files:
                    raise ValueError(f"No .safetensors file found in {lora_path}")
                
                final_lora_path = str(safetensors_files[0])
                print(f"Using safetensors file: {os.path.basename(final_lora_path)}")
            else:
                # Direct download for safetensors with unique name
                final_lora_path = f"/tmp/lora_{url_hash}.safetensors"
                
                # Download only if not exists
                if not os.path.exists(final_lora_path):
                    print(f"Downloading safetensors file...")
                    response = requests.get(lora_path)
                    with open(final_lora_path, "wb") as f:
                        f.write(response.content)
                else:
                    print(f"Using cached file: {final_lora_path}")
        else:
            final_lora_path = lora_path
        
        # Load LoRA with unique adapter name using optimized diffusers API
        self.pipe.load_lora_weights(final_lora_path, adapter_name=unique_adapter_name)
        self.loaded_loras[unique_adapter_name] = {
            'url': lora_path,
            'path': final_lora_path
        }
        print(f"✓ LoRA {adapter_name} loaded successfully as {unique_adapter_name}")
        print(f"  Cached at: {final_lora_path}")
        
        return unique_adapter_name

    def _get_dimensions(self, aspect_ratio: str, image_size: str) -> tuple:
        """Get dimensions based on aspect ratio and image size preset"""
        
        if image_size == "optimize_for_quality":
            dims = ASPECT_RATIOS
        else:  # optimize_for_speed
            dims = SPEED_DIMENSIONS
        
        width, height = dims.get(aspect_ratio, (1328, 1328))
        
        # Ensure divisible by 16
        adjusted_width = (width // 16) * 16
        adjusted_height = (height // 16) * 16
        
        # Log if adjustment was needed (shouldn't happen with our presets)
        if adjusted_width != width or adjusted_height != height:
            print(f"`height` and `width` have to be divisible by 16 but are {width} and {height}.")
            print(f"Dimensions will be resized to {adjusted_width}x{adjusted_height}")
        
        return adjusted_width, adjusted_height

    def predict(
        self,
        prompt: str = Input(
            description="The main prompt for image generation"
        ),
        enhance_prompt: bool = Input(
            default=False,
            description="Automatically enhance the prompt for better image generation"
        ),
        negative_prompt: str = Input(
            default="",
            description="Things you do not want to see in your image"
        ),
        aspect_ratio: str = Input(
            default="16:9",
            choices=["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"],
            description="Aspect ratio for the generated image. Ignored if width and height are both provided."
        ),
        image_size: str = Input(
            default="optimize_for_quality",
            choices=["optimize_for_quality", "optimize_for_speed"],
            description="Image size preset (quality = larger, speed = faster). Ignored if width and height are both provided."
        ),
        width: int = Input(
            default=0,
            ge=0,
            le=2048,
            description="Custom width in pixels. Provide both width and height for custom dimensions (overrides aspect_ratio/image_size). 0 to use presets."
        ),
        height: int = Input(
            default=0,
            ge=0,
            le=2048,
            description="Custom height in pixels. Provide both width and height for custom dimensions (overrides aspect_ratio/image_size). 0 to use presets."
        ),
        go_fast: bool = Input(
            default=False,
            description="Use LCM-LoRA to accelerate image generation (trades quality for 8x speed)"
        ),
        num_inference_steps: int = Input(
            default=50,
            ge=1,
            le=100,
            description="Number of denoising steps. More steps = higher quality."
        ),
        guidance: float = Input(
            default=4.0,
            ge=0.0,
            le=10,
            description="Guidance scale for image generation (true_cfg_scale)."
        ),
        seed: int = Input(
            default=-1,
            description="Set a seed for reproducibility (-1 for random)."
        ),
        output_format: str = Input(
            default="webp",
            choices=["webp", "jpg", "png"],
            description="Format of the output images"
        ),
        output_quality: int = Input(
            default=80,
            ge=0,
            le=100,
            description="Quality when saving images (0-100, higher is better, 100 = lossless)"
        ),
        replicate_weights: CogPath = Input(
            default=None,
            description="Path to LoRA weights file. Leave blank to use base model."
        ),
        lora_scale: float = Input(
            default=1.0,
            ge=0,
            le=3,
            description="Scale for LoRA weights (0 = base model, 1 = full LoRA)"
        ),
        extra_lora_weights: CogPath = Input(
            default=None,
            description="Path to additional LoRA weights file (for combining multiple LoRAs)."
        ),
        extra_lora_scale: float = Input(
            default=1.0,
            ge=0,
            le=3,
            description="Scale for extra LoRA weights"
        )
    ) -> CogPath:
        """Run a single prediction on the model"""
        # Determine dimensions with smart handling
        if width > 0 and height > 0:
            # User provided explicit dimensions - validate and adjust if needed
            orig_w, orig_h = width, height
            
            # Ensure divisible by 16 (round to nearest)
            width = max(512, round(width / 16) * 16)
            height = max(512, round(height / 16) * 16)
            
            # Cap at max dimensions
            width = min(width, 2048)
            height = min(height, 2048)
            
            if (orig_w, orig_h) != (width, height):
                print(f"📐 Adjusted dimensions from {orig_w}x{orig_h} to {width}x{height} (divisible by 16)")
            else:
                print(f"📐 Using custom dimensions: {width}x{height}")
        
        elif width > 0 or height > 0:
            # Only one dimension provided - error
            raise ValueError(
                "Both width and height must be provided together for custom dimensions. "
                "Otherwise, use aspect_ratio and image_size presets."
            )
        
        else:
            # Use preset dimensions based on aspect_ratio and image_size
            width, height = self._get_dimensions(aspect_ratio, image_size)
            mode_name = "quality" if image_size == "optimize_for_quality" else "speed"
            print(f"📐 Using {mode_name} preset for {aspect_ratio}: {width}x{height}")

        # Override steps for go_fast mode (not used with Qwen, but kept for API compatibility)
        if go_fast and num_inference_steps > 28:
            num_inference_steps = 28
        
        # Load LoRA weights with intelligent caching
        adapters = []
        adapter_weights = []
        
        if replicate_weights:
            unique_name = self._load_lora_weights(str(replicate_weights), "main_lora", lora_scale)
            if unique_name:
                adapters.append(unique_name)
                adapter_weights.append(lora_scale)
        
        if extra_lora_weights:
            unique_name = self._load_lora_weights(str(extra_lora_weights), "extra_lora", extra_lora_scale)
            if unique_name:
                adapters.append(unique_name)
                adapter_weights.append(extra_lora_scale)
        
        # Set adapters if LoRAs are loaded
        if adapters:
            self.pipe.set_adapters(adapters, adapter_weights=adapter_weights)
            print(f"Using LoRAs: {adapters} with weights {adapter_weights}")
        
        # Set seed
        if seed == -1:
            seed = int.from_bytes(os.urandom(4), "big")
            print(f"Using random seed: {seed}")
        else:
            print(f"Using seed: {seed}")
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # Enhance prompt if requested
        if enhance_prompt:
            prompt = f"{prompt}, highly detailed, crisp focus, studio lighting, photorealistic"
        
        # Add quality magic prompt suffix
        positive_magic = ", Ultra HD, 4K, cinematic composition."
        full_prompt = prompt + positive_magic
        
        # Prepare negative prompt
        if not negative_prompt.strip():
            negative_prompt = " "
        
        # Generate
        print(f"Generating: {prompt} ({width}x{height}, steps={num_inference_steps}, seed={seed})")
        
        prediction_start = time.time()
        
        # Optimized generation using native diffusers pipeline
        output = self.pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=guidance,
            num_images_per_prompt=1,
            generator=generator,
        )
        
        prediction_end = time.time()
        prediction_time = prediction_end - prediction_start
        
        # Get the generated image
        img = output.images[0]
        
        # Save
        output_path = f"/tmp/output.{output_format}"
        save_kwargs = {"quality": output_quality} if output_format in ("jpg", "webp") else {}
        
        if output_format == "jpg":
            # Convert RGBA to RGB if necessary
            if img.mode == "RGBA":
                img = img.convert("RGB")
            save_kwargs["optimize"] = True
            img.save(output_path, format="JPEG", **save_kwargs)
        elif output_format == "webp":
            img.save(output_path, format="WEBP", **save_kwargs)
        else:  # png
            img.save(output_path, format="PNG")
        
        # Record billing metric after successful image generation
        record_billing_metric("image_output_count", 1)
        
        print(f"Generation took {prediction_time:.2f} seconds")
        print(f"Total safe images: 1/1")
        
        # Keep LoRAs loaded in pipe for maximum performance on subsequent requests
        
        return CogPath(output_path)
