# JAX Tutorial

This repository provides a collection of machine learning examples implemented using JAX and Flax.
The code is written as part of my personal learning of these frameworks.

| Task        | Framework | Progress |
| ----------- | --------- | -------- |
| MNIST + MLP | Pure JAX  | ✔        |
| DDPM        | Flax      | ✘        |
| LLaMa3      | Flax      | ✘        |

## Setup

Install the required packages:

```bash
uv sync
```

## MNIST + MLP with Pure JAX

```bash
uf run -m mnist_mlp.train
```

Training

| Device | Training Time in One Epoch |
| ------ | -------------------------- |
| CPU    | 1.87 s                     |
| GPU    |                            |

CPU time is measured on Apple M3 Pro.
The training time is measured for the subsequent epochs after the first jit compilation.
