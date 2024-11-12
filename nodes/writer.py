class SQImageWriter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {

            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }
    
    RETURN_TYPES = ("",)
    RETURN_NAMES = ("",)
    OUTPUT_NODE = True
    CATEGORY = "SQNodes"
    FUNCTION = "write"
    DESCRIPTION = "Save images with reusable generation metadata"

    def write(self, prompt=None, extra_pnginfo=None):
        return ()