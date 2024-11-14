from typing import Dict, Literal, Optional

import torch
from .utils import any_type, calculate_hash, save_image
from .types import MetadataOutput, GeneratorForward, LoraMetadata
from .pure_utils import log


class SQImageWriter:
    hash_cache: Dict[str, str] = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "directory": (
                    "STRING",
                    {"default": ".", "tooltip": "Directory to save images"},
                ),
                "filename": (
                    "STRING",
                    {
                        "default": "image_${3}.png",
                        "tooltip": "png and webp supported. Use ${n} to label index with padding n. Use $timestamp to insert timestamp as formatted",
                    },
                ),
                "timestamp_format": ("STRING", {"default": "%Y%m%d-%H%M%S"}),
                "final": (
                    ["true", "false"],
                    {"default": "false", "tooltip": "Remove ConfyUI workflow metadata"},
                ),
            },
            "optional": {
                "loras": (any_type,),
                "seed": ("INT", {"default": 0, "defaultInput": True}),
                "steps": ("INT", {"default": 0, "defaultInput": True}),
                "cfg": ("FLOAT", {"default": 0.0, "defaultInput": True}),
                "width": ("INT", {"default": 0, "defaultInput": True}),
                "height": ("INT", {"default": 0, "defaultInput": True}),
                "positive": (
                    any_type,
                    {
                        "tooltip": "List of positive prompts (obtained through chain prompter node)",
                        "defaultInput": True,
                    },
                ),
                "negative": (
                    any_type,
                    {
                        "tooltip": "List of negative prompts (obtained through chain prompter node)",
                        "defaultInput": True,
                    },
                ),
                "generator_forward": (
                    any_type,
                    {"tooltip": "Pass in forwarded metadata from generator node"},
                ),
                "reader_forward": (
                    any_type,
                    {
                        "tooltip": "Pass in forwarded metadata from reader node. Seed, steps and cfg can be overriden by their respective inputs"
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    CATEGORY = "SQNodes"
    FUNCTION = "write"
    DESCRIPTION = "Save images with reusable generation metadata"

    def write(
        self,
        image: torch.Tensor,
        directory: str,
        filename: str,
        timestamp_format: str,
        final: Literal["true", "false"] = "false",
        loras: Optional[list[LoraMetadata]] = None,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        cfg: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        positive: Optional[list[str]] = None,
        negative: Optional[list[str]] = None,
        generator_forward: Optional[GeneratorForward] = None,
        reader_forward: Optional[MetadataOutput] = None,
        prompt=None,
        extra_pnginfo=None,
    ):

        if reader_forward is not None:
            metadata = reader_forward

            def apply_value(key, val):
                if val is not None:
                    metadata[key] = val

            apply_value("seed", seed)
            apply_value("steps", steps)
            apply_value("cfg", cfg)
        else:
            if (
                loras is None
                or seed is None
                or steps is None
                or cfg is None
                or width is None
                or height is None
                or positive is None
                or negative is None
                or generator_forward is None
            ):
                raise ValueError(
                    "all inputs must be provided if reader forward is not provided"
                )
            metadata: MetadataOutput = {
                "model": {
                    "name": generator_forward["model_name"],
                    "sha": calculate_hash(
                        self.hash_cache, generator_forward["model_name"], "model"
                    ),
                },
                "vae": (
                    {
                        "name": generator_forward["vae_name"],
                        "sha": (
                            calculate_hash(
                                self.hash_cache, generator_forward["vae_name"], "vae"
                            )
                            if generator_forward["vae_name"] != "built-in"
                            else ""
                        ),
                    }
                ),
                "loras": [
                    {
                        "name": lora["name"],
                        "sha": calculate_hash(self.hash_cache, lora["name"], "lora"),
                        "clip_strength": lora["clip_strength"],
                        "model_strength": lora["model_strength"],
                    }
                    for lora in loras
                ],
                "cfg": cfg,
                "seed": seed,
                "steps": steps,
                "width": width,
                "height": height,
                "positive": positive,
                "negative": negative,
                "sampler": generator_forward["sampler"],
                "scheduler": generator_forward["scheduler"],
            }  # type: ignore
        filename = save_image(
            image[0],
            directory,
            filename,
            prompt,
            extra_pnginfo,
            metadata,
            final=final == "true",
            timestamp_format=timestamp_format,
        )
        log(f"Saved image to {filename}")

        return ()
