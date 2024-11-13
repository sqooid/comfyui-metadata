import folder_paths
import comfy.sd
import comfy.utils
import torch
from .utils import any_type, load_lora
from .types import LoraMetadata, LoraMetadataOutput, GeneratorForward
import comfy.samplers

builtin_vae = "built-in"


def list_vaes():
    vaes = [builtin_vae]
    vaes.extend(folder_paths.get_filename_list("vae"))
    approx_vaes = folder_paths.get_filename_list("vae_approx")
    sdxl_taesd_enc = False
    sdxl_taesd_dec = False
    sd1_taesd_enc = False
    sd1_taesd_dec = False
    sd3_taesd_enc = False
    sd3_taesd_dec = False
    f1_taesd_enc = False
    f1_taesd_dec = False

    for v in approx_vaes:
        if v.startswith("taesd_decoder."):
            sd1_taesd_dec = True
        elif v.startswith("taesd_encoder."):
            sd1_taesd_enc = True
        elif v.startswith("taesdxl_decoder."):
            sdxl_taesd_dec = True
        elif v.startswith("taesdxl_encoder."):
            sdxl_taesd_enc = True
        elif v.startswith("taesd3_decoder."):
            sd3_taesd_dec = True
        elif v.startswith("taesd3_encoder."):
            sd3_taesd_enc = True
        elif v.startswith("taef1_encoder."):
            f1_taesd_dec = True
        elif v.startswith("taef1_decoder."):
            f1_taesd_enc = True
    if sd1_taesd_dec and sd1_taesd_enc:
        vaes.append("taesd")
    if sdxl_taesd_dec and sdxl_taesd_enc:
        vaes.append("taesdxl")
    if sd3_taesd_dec and sd3_taesd_enc:
        vaes.append("taesd3")
    if f1_taesd_dec and f1_taesd_enc:
        vaes.append("taef1")
    return vaes


def load_taesd(name):
    sd = {}
    approx_vaes = folder_paths.get_filename_list("vae_approx")

    encoder = next(
        filter(lambda a: a.startswith("{}_encoder.".format(name)), approx_vaes)
    )
    decoder = next(
        filter(lambda a: a.startswith("{}_decoder.".format(name)), approx_vaes)
    )

    enc = comfy.utils.load_torch_file(
        folder_paths.get_full_path_or_raise("vae_approx", encoder)
    )
    for k in enc:
        sd["taesd_encoder.{}".format(k)] = enc[k]

    dec = comfy.utils.load_torch_file(
        folder_paths.get_full_path_or_raise("vae_approx", decoder)
    )
    for k in dec:
        sd["taesd_decoder.{}".format(k)] = dec[k]

    if name == "taesd":
        sd["vae_scale"] = torch.tensor(0.18215)
        sd["vae_shift"] = torch.tensor(0.0)
    elif name == "taesdxl":
        sd["vae_scale"] = torch.tensor(0.13025)
        sd["vae_shift"] = torch.tensor(0.0)
    elif name == "taesd3":
        sd["vae_scale"] = torch.tensor(1.5305)
        sd["vae_shift"] = torch.tensor(0.0609)
    elif name == "taef1":
        sd["vae_scale"] = torch.tensor(0.3611)
        sd["vae_shift"] = torch.tensor(0.1159)
    return sd


class SQParameterGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"tooltip": "The name of the checkpoint (model) to load."},
                ),
                "vae_name": (
                    list_vaes(),
                    {"tooltip": "The name of the vae to load."},
                ),
                "sampler": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {
                        "tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."
                    },
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {
                        "tooltip": "The scheduler controls how noise is gradually removed to form the image."
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        comfy.samplers.KSampler.SAMPLERS,
        comfy.samplers.KSampler.SCHEDULERS,
        any_type,
    )
    RETURN_NAMES = ("model_name", "vae_name", "sampler", "scheduler", "forward")
    CATEGORY = "SQNodes"
    FUNCTION = "generate"
    DESCRIPTION = "Generate reusable generation parameters"

    def generate(self, ckpt_name, vae_name, sampler, scheduler):
        forward: GeneratorForward = {
            "model_name": ckpt_name,
            "vae_name": vae_name,
            "sampler": sampler,
            "scheduler": scheduler,
        }
        return ckpt_name, vae_name, sampler, scheduler, forward


class SQCheckpointLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (
                    "STRING",
                    {"tooltip": "The name of the checkpoint (model) to load."},
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    CATEGORY = "SQNodes"
    FUNCTION = "load"
    DESCRIPTION = "Load a checkpoint from parameter generator output"

    def load(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return out[:3]


class SQVaeLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (
                    "STRING",
                    {"tooltip": "The name of the vae to load.", "defaultInput": True},
                ),
                "built_in": (
                    "VAE",
                    {
                        "tooltip": "Built-in vae that might be loaded by the checkpoint",
                        "defaultInput": True,
                    },
                ),
            },
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    CATEGORY = "SQNodes"
    FUNCTION = "load"
    DESCRIPTION = "Load a VAE from parameter generator output"

    def load(self, vae_name, built_in):
        if vae_name == builtin_vae:
            return (built_in,)
        if vae_name in ["taesd", "taesdxl", "taesd3", "taef1"]:
            sd = load_taesd(vae_name)
        else:
            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            sd = comfy.utils.load_torch_file(vae_path)
        vae = comfy.sd.VAE(sd=sd)
        return (vae,)


class SQLoraChainLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (
                    folder_paths.get_filename_list("loras"),
                    {"tooltip": "The name of the vae to load."},
                ),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "model_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01},
                ),
                "clip_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01},
                ),
            },
            "optional": {"chain": (any_type,)},
        }

    RETURN_TYPES = ("MODEL", "CLIP", any_type)
    RETURN_NAMES = ("model", "clip", "chain")
    CATEGORY = "SQNodes"
    FUNCTION = "load"
    DESCRIPTION = "Load a Lora and chain outputs for metadata"

    def load(self, lora_name, model, clip, model_strength, clip_strength, chain=None):
        if model_strength == 0 and clip_strength == 0:
            return (model, clip, lora_name)

        model_lora, clip_lora = load_lora(
            model, clip, lora_name, model_strength, clip_strength
        )

        param: LoraMetadata = {
            "name": lora_name,
            "clip_strength": clip_strength,
            "model_strength": model_strength,
        }

        if chain is None:
            chain = [param]
        else:
            chain = chain + [param]

        return (model_lora, clip_lora, chain)


class SQLoraAutoLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loras": (
                    any_type,
                    {"tooltip": "Loras output from metadata reader"},
                ),
                "model": ("MODEL",),
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    CATEGORY = "SQNodes"
    FUNCTION = "load"
    DESCRIPTION = "Load multiple loras as specified in metadata"

    def load(self, loras, model, clip):
        ls: list[LoraMetadataOutput] = loras
        model_lora = model
        clip_lora = clip
        for l in ls:
            if l["model_strength"] == 0 and l["clip_strength"] == 0:
                continue
            model_lora, clip_lora = load_lora(
                model_lora,
                clip_lora,
                l["name"],
                l["model_strength"],
                l["clip_strength"],
            )

        return (model_lora, clip_lora)
