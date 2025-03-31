##Stable Diffusion – Research Paper Implementation


This repository contains the full implementation of the Stable Diffusion model as described in the original research paper. Stable Diffusion is a type of latent diffusion model that enables high-quality image generation through denoising in a compressed latent space, significantly reducing computational requirements without sacrificing output fidelity.



Overview-

This project is structured to mirror the key components of the original paper, including:

Variational Autoencoder (VAE): For encoding and decoding images to and from the latent space.

UNet Backbone: The core denoising network, optionally enhanced with attention mechanisms.

Scheduler/Noise Sampler: Handles noise injection and guides the denoising steps during training and inference.

Text Encoder: Uses CLIP (or similar transformer-based encoders) to condition image generation on text prompts.

Latent Diffusion Pipeline: Ties all modules together for end-to-end training and inference.




Features-
Modular and clean architecture

Rewritten and refactored PyTorch codebase for clarity and originality

Easily extendable for experiments and fine-tuning

Support for prompt-based image generation

This project is for educational and research purposes only.



## Deployment

To deploy this project run

```bash
  run the demo.py file
```

python version  3.11.3


## weights and tokenizers

in the data directory download the following weights and enbeddings

from  https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/tokenizer   download tokenizer_config.json and vocab.json

from https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main download
