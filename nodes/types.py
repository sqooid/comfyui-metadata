import typing


class ModelMetadata(typing.TypedDict):
    name: str


class ModelMetadataOutput(ModelMetadata):
    sha: str


class LoraMetadata(typing.TypedDict):
    name: str
    clip_strength: float
    model_strength: float


class LoraMetadataOutput(LoraMetadata):
    sha: str


class GeneratorForward(typing.TypedDict):
    model_name: str
    vae_name: str
    sampler: str
    scheduler: str


class PromptChain(typing.TypedDict):
    prompts: list[str]
    conditioning: typing.Any
    clip: typing.Any


class MetadataOutput(typing.TypedDict):
    model: ModelMetadataOutput
    vae: ModelMetadataOutput
    loras: list[LoraMetadataOutput]
    seed: int
    steps: int
    cfg: float
    sampler: str
    scheduler: str
    width: int
    height: int
    positive: list[str]
    negative: list[str]
