from .nodes.reader import *
from .nodes.writer import *

NODE_CLASS_MAPPINGS = {
    "SQ Image Writer": "SQImageWriter",
    "SQ Image Reader": "SQImageReader",
}

__all__ = ["NODE_CLASS_MAPPINGS"]
