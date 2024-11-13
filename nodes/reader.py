import json
import numpy as np
import os
from PIL import Image, ImageOps
import torch
import folder_paths
from .utils import any_type
from .types import MetadataOutput
from .pure_utils import log
import comfy.samplers


class SQImageReader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filepath": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Image file to read metadata from. Paths starting with '.' are relative to the output directory",
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        any_type,
        "INT",
        "INT",
        "FLOAT",
        comfy.samplers.KSampler.SAMPLERS,
        comfy.samplers.KSampler.SCHEDULERS,
        "INT",
        "INT",
        any_type,
        any_type,
        any_type,
        "IMAGE",
        "STRING",
    )
    RETURN_NAMES = (
        "model_name",
        "vae_name",
        "loras",
        "seed",
        "steps",
        "cfg",
        "sampler",
        "scheduler",
        "width",
        "height",
        "positive",
        "negative",
        "forward",
        "image",
        "filename",
    )
    OUTPUT_NODE = True
    CATEGORY = "SQNodes"
    FUNCTION = "read"
    DESCRIPTION = "Save images with reusable generation metadata"

    def read(self, filepath: str):
        log(f"File: {filepath}")
        if filepath.startswith("."):
            filepath = os.path.join(folder_paths.get_output_directory(), filepath)
        with open(filepath, "rb") as f:
            img = Image.open(f)
            img.load()
            img = ImageOps.exif_transpose(img)
            image = img.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            info = img.info
        metadata_json = info.get("metadata")
        if metadata_json is None:
            raise ValueError("No compatible metadata found")
        metadata: MetadataOutput = json.loads(metadata_json)
        log(metadata)

        model_name = metadata["model"]["name"]
        vae_name = metadata["vae"]["name"]
        loras = metadata["loras"]
        filename = os.path.basename(filepath)

        return (
            model_name,
            vae_name,
            loras,
            metadata["seed"],
            metadata["steps"],
            metadata["cfg"],
            metadata["sampler"],
            metadata["scheduler"],
            metadata["width"],
            metadata["height"],
            metadata["positive"],
            metadata["negative"],
            metadata,
            image,
            filename,
        )
