# JAX Tutorial

This repository provides a collection of machine learning examples implemented using JAX and Flax.
The code is written as part of my personal learning of these frameworks.

| Task        | Framework | Progress |
| ----------- | --------- | -------- |
| MNIST + MLP | Pure JAX  | ✔        |
| DDPM        | Flax      | ✔        |
| LLaMa3      | Flax      | -        |

## Setup

Install the required packages:

```bash
uv sync
```

## MNIST + MLP with Pure JAX

```bash
uv run -m mnist_mlp.train
```

## DDPM

Denoising diffusion probabilistic model implementation using Flax.

```bash
uv run -m ddpm.train
```

The training time for one iteration (batch size = 128) is as follows:

| Device | Training Time in One Iteration |
| ------ | ------------------------------ |
| CPU    | 3.2776 s                       |
| GPU    | 0.0148 s                       |

CPU time is measured on Apple M5, and GPU time is measured on single NVIDIA GH200 Grace Hopper Superchip containing an H100 GPU.

The training time is measured for the subsequent iterations after the first jit compilation.

## Llama3

I initially planned to implement Llama 3 using Flax to get familiar with LLM experiments.
However, official jax support in the Hugging Face transformers library appears to be deprecated from 2025.

There are some approaches to bridge the jax features and PyTorch models, such as [torchax](https://github.com/google/torchax) (see also [this blog post](https://huggingface.co/blog/qihqi/huggingface-jax-01)).
While these approaches are interesting and worth considering, I have decided not to use jax for LLM experiments for now.
If there are any strong advantages of jax-based LLM experiments that I don't know yet, I will reconsider this decision.
