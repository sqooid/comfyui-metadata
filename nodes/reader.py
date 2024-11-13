from typing import Optional
from PIL import Image
import folder_paths
import nodes
from .utils import any_type
from .types import MetadataOutput
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
                        "tooltip": "The name of the checkpoint (model) to load.",
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
    )
    OUTPUT_NODE = True
    CATEGORY = "SQNodes"
    FUNCTION = "read"
    DESCRIPTION = "Save images with reusable generation metadata"

    def read(self, filepath: str):
        with open(filepath, "rb") as f:
            img = Image.open(f)
            img.load()
            info = img.info
            metadata: Optional[MetadataOutput] = info.get("metadata")
            if metadata is None:
                raise ValueError("No compatible metadata found")
            model_name = metadata["model"]["name"]
            vae_name = metadata["vae"]["name"]
            loras = metadata["loras"]
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
            )
