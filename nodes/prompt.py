import copy
import re
from typing import Any, Optional

import torch
from .utils import any_type
from .pure_utils import hash_var, parse_text, log
from .types import PromptChain


def concat_cond(cond1, cond2):
    out = []
    for i in range(len(cond1)):
        t1 = cond1[i][0]
        tw = torch.cat((t1, cond2[0][0]), 1)
        n = [tw, cond1[i][1].copy()]
        out.append(n)
    return out


def encode_cond(clip, text):
    tokens = clip.tokenize(text)
    output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    cond = output.pop("cond")
    return [[cond, output]]


class SQChainPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Prompt text. {|} expressions can be nested",
                    },
                ),
            },
            "optional": {
                "chain": (any_type,),
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = (any_type, "CONDITIONING", any_type)
    RETURN_NAMES = ("chain", "conditioning", "prompts")
    CATEGORY = "SQNodes"
    FUNCTION = "parse"
    DESCRIPTION = "Chain prompts with conditioning concat for longer attention. Pass prompts output to pos/neg writer input"

    @classmethod
    def IS_CHANGED(cls, prompt, clip=None, chain=None):
        if re.search(r"[{}]", prompt):
            return float("NaN")
        return 0

    def parse(
        self,
        prompt: str,
        clip: Any = None,
        chain: Optional[PromptChain] = None,
    ):
        # parse prompt
        text = parse_text(prompt)
        # get clip
        if chain is not None and clip is None:
            clip = chain["clip"]
        conditioning = encode_cond(clip, text)
        # chain prompts
        if chain is None:
            chain = {"prompts": [text], "conditioning": conditioning, "clip": clip}
            new_cond = conditioning
        else:
            new_cond = concat_cond(chain["conditioning"], conditioning)
            chain = {
                # shallow copy is fine because strings are immutable
                "prompts": chain["prompts"].copy(),
                "conditioning": new_cond,
                "clip": clip,
            }
            chain["prompts"].append(text)

        log(f"prompt loaded {text[:8]}... {hash_var(str(new_cond))}")

        return chain, new_cond, chain["prompts"]


class SQAutoPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": (any_type,),
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    CATEGORY = "SQNodes"
    FUNCTION = "parse"
    DESCRIPTION = "Load all prompts as specified in reader metadata. Pass in pos/neg output from reader"

    def parse(self, prompts: list[str], clip):
        conditioning = encode_cond(clip, prompts[0])
        log(f"prompt loaded: {prompts[0][:8]}... {hash_var(str(conditioning))}")
        for p in prompts[1:]:
            conditioning = concat_cond(conditioning, encode_cond(clip, p))

            log(f"prompt loaded: {p[:8]}... {hash_var(str(conditioning))}")

        return (conditioning,)
