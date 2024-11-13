# SQ Nodes

Creates nodes for saving and reading images while storing generation metadata such that generation parameters can be perfectly replicated completely automatically in a different workflow while allowing flexibility to change any generation parameter.

Things like loras, sampler, scheduler, VAE etc. (and basically all KSampler inputs) can be replicated fully automatically just by reading in an image using the reader node.

Other auxilliary nodes like the lora and vae loaders, and prompt chainers are used to automate this.