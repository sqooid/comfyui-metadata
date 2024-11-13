from typing import Any, Optional

import torch
from .utils import any_type
from .pure_utils import parse_text
from .types import PromptChain


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
    DESCRIPTION = "Chain prompts with conditioning concat"

    def parse(
        self,
        prompt: str,
        clip: Any = None,
        chain: Optional[PromptChain] = None,
    ):
        text = parse_text(prompt)
        if chain is not None and clip is None:
            clip = chain["clip"]
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        conditioning = [[cond, output]]
        if chain is None:
            chain = {"prompts": [prompt], "conditioning": conditioning, "clip": clip}
            new_cond = conditioning
        else:
            chain["prompts"] = chain["prompts"] + [prompt]
            new_cond = []
            conditioning_to = chain["conditioning"]
            for i in range(len(conditioning_to)):
                t1 = conditioning_to[i][0]
                tw = torch.cat((t1, cond), 1)
                n = [tw, conditioning_to[i][1].copy()]
                new_cond.append(n)
        return chain, new_cond, chain["prompts"]
