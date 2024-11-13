import hashlib
import json
from PIL.PngImagePlugin import PngInfo
import typing
import folder_paths
import torch
import os
import re
import datetime
from typing import Optional
from PIL import Image
import numpy as np
from .types import MetadataOutput
from comfy.cli_args import args
import comfy.sd
import comfy.utils


class AnyType(str):
    """A special type that can be connected to any other types. Credit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")


def format_filename(filename: str, index: int, timestamp_format: Optional[str] = None):
    timestamp = datetime.datetime.now().strftime(timestamp_format or "%Y%m%d-%H%M%S")
    match = re.search(r"\$\{(\d*)\}", filename)
    if match:
        padding = int(match.group(1) or 0)
        filename = filename.replace(match.group(0), f"{index:0{padding}d}")
    filename = filename.replace("$timestamp", timestamp)
    return filename


def calculate_hash(
    cache: typing.Dict[str, str],
    name: str,
    file_type: typing.Literal["model", "vae", "lora"],
):
    cached = cache.get(name)
    if cached:
        return cached
    if file_type == "model":
        filename = folder_paths.get_full_path("checkpoints", name)
    elif file_type == "vae":
        filename = folder_paths.get_full_path("vae", name)
    elif file_type == "lora":
        filename = folder_paths.get_full_path("loras", name)
    if not filename:
        raise FileNotFoundError(f"File not found: {file_type} {name}")
    sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            sha256.update(chunk)

    hash_value = sha256.hexdigest()[:10]
    cache[name] = hash_value

    return hash_value


def save_image(
    image: torch.Tensor,
    directory: str,
    filename: str,
    prompt,
    extra_pnginfo,
    metadata: MetadataOutput,
    compress_level: int = 4,
    timestamp_format: Optional[str] = None,
):
    output_path = os.path.join(folder_paths.get_output_directory(), directory)
    os.makedirs(output_path, exist_ok=True)
    file_count = len(os.listdir(output_path))
    filename = format_filename(filename, file_count, timestamp_format)
    i = 255.0 * image.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    mdata = None
    mdata = PngInfo()
    if not args.disable_metadata:
        if prompt is not None:
            mdata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                mdata.add_text(x, json.dumps(extra_pnginfo[x]))

    mdata.add_text("metadata", json.dumps(metadata))
    img.save(
        os.path.join(output_path, filename),
        pnginfo=mdata,
        compress_level=compress_level,
    )
    return filename


def load_lora(model, clip, lora_name, model_strength, clip_strength):
    lora_path = folder_paths.get_full_path("loras", lora_name)
    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

    model_lora, clip_lora = comfy.sd.load_lora_for_models(
        model, clip, lora, model_strength, clip_strength
    )
    return model_lora, clip_lora
