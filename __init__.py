from .nodes.reader import *
from .nodes.writer import *
from .nodes.generator import *
from .nodes.prompt import *
from .server.routes import *

NODE_CLASS_MAPPINGS = {
    "SQ Image Writer": SQImageWriter,
    "SQ Image Reader": SQImageReader,
    "SQ Parameter Generator": SQParameterGenerator,
    "SQ Checkpoint Loader": SQCheckpointLoader,
    "SQ VAE Loader": SQVaeLoader,
    "SQ Lora Chain Loader": SQLoraChainLoader,
    "SQ Lora Auto Loader": SQLoraAutoLoader,
    "SQ Prompt Chain": SQChainPrompt,
    "SQ Prompt Auto": SQAutoPrompt,
}

WEB_DIRECTORY = "./js"


__all__ = ["NODE_CLASS_MAPPINGS", "WEB_DIRECTORY"]
