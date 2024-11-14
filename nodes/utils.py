import hashlib
import piexif
import piexif.helper
import json
from PIL.PngImagePlugin import PngInfo
import typing
import folder_paths
import torch
import os
import re
import datetime
from typing import Optional
from PIL import Image, ExifTags
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
    final: bool = False,
    compress_level: int = 4,
    timestamp_format: Optional[str] = None,
):
    output_path = os.path.join(folder_paths.get_output_directory(), directory)
    os.makedirs(output_path, exist_ok=True)
    file_count = len(os.listdir(output_path))
    filename = format_filename(filename, file_count, timestamp_format)
    i = 255.0 * image.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    save_path = os.path.join(output_path, filename)
    civit_metadata = format_civit_metadata(metadata)
    metadata_str = json.dumps(metadata)
    prompt_str = json.dumps(prompt)

    # png tEXt metadata
    if filename.endswith(".png"):
        mdata = None
        mdata = PngInfo()
        mdata.add_text("parameters", civit_metadata)
        mdata.add_text("metadata", metadata_str)
        if not final:
            if not args.disable_metadata:
                if prompt is not None:
                    mdata.add_text("prompt", prompt_str)
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        mdata.add_text(x, json.dumps(extra_pnginfo[x]))
        img.save(
            save_path,
            pnginfo=mdata,
            compress_level=compress_level,
        )

    # general exif metadata
    elif filename.endswith(".webp"):
        exif = {
            "Exif": {
                piexif.ExifIFD.UserComment: (
                    piexif.helper.UserComment.dump(civit_metadata, encoding="unicode")
                )
            },
            "0th": {
                piexif.ImageIFD.Software: metadata_str,
            },
        }

        if not final:
            if prompt is not None:
                exif["0th"][0x0110] = f"prompt:{prompt_str}"
            if extra_pnginfo is not None:
                inital_exif = 0x010F
                for x in extra_pnginfo:
                    exif["0th"][inital_exif] = "{}:{}".format(
                        x, json.dumps(extra_pnginfo[x])
                    )
                    inital_exif -= 1

        img.save(
            save_path,
            lossless=True,
        )
        exif_bytes = piexif.dump(exif)
        piexif.insert(exif_bytes, save_path)
    return filename


def load_lora(model, clip, lora_name, model_strength, clip_strength):
    lora_path = folder_paths.get_full_path("loras", lora_name)
    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

    model_lora, clip_lora = comfy.sd.load_lora_for_models(
        model, clip, lora, model_strength, clip_strength
    )
    return model_lora, clip_lora


def format_civit_metadata(metadata: MetadataOutput):
    positive = ", ".join(metadata["positive"])
    for w in ["loli", "lolita", "underage", "child", "children"]:
        positive = positive.replace(w, "")
    negative = "Negative prompt: " + ", ".join(metadata["negative"])
    lora_hashes = ", ".join(
        [f"{lora['name']}: {lora['sha']}" for lora in metadata["loras"]]
    )
    hashes_dict = {
        "model": metadata["model"]["sha"],
    }
    for lora in metadata["loras"]:
        hashes_dict[f'lora:{lora["name"]}'] = lora["sha"]
    if metadata["vae"]["name"] != "built-in":
        hashes_dict[metadata["vae"]["name"]] = metadata["vae"]["sha"]
    hashes = json.dumps(hashes_dict)
    settings = ", ".join(
        [
            f"Steps: {metadata['steps']}",
            f"Sampler: {metadata['sampler']}_{metadata['scheduler']}",
            f"CFG scale: {metadata['cfg']}",
            f"Seed: {metadata['seed']}",
            f"Size: {metadata['width']}x{metadata['height']}",
            f"Model hash: {metadata['model']['sha']}",
            f"Model: {metadata['model']['name']}",
            f'Lora hashes: "{lora_hashes}"',
            f'TI hashes: ""',
            f"Version: ComfyUI",
            f"Hashes: {hashes}",
        ]
    )
    return "\n".join([positive, negative, settings])
