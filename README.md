# Train and export MNIST weights with PyTorch

This project trains a small neural network for handwritten-digit recognition and exports its weights and biases. The trained model is a floating-point PyTorch checkpoint; `quantize.py` can also produce integer versions at several weight precisions.

## Model and data

The official MNIST images are 28 × 28 grayscale pixels. `model.py` averages each non-overlapping 2 × 2 block, with half-up rounding, to make a 14 × 14 input. The network is:

```text
196 inputs → 16 hidden units → 10 digit scores
```

The hidden activation is clipped to the range 0–1. The model has 3,296 weights and 26 biases. The output digit is the index of the highest score.

Training uses 55,000 images from MNIST's training set and reserves 5,000 for validation. The best validation checkpoint is saved. `quantize.py` evaluates the saved checkpoint on the separate 10,000-image test set. The dataset is loaded with [Torchvision's MNIST class](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST).

## Train the weights

On this Ubuntu 26.04 workspace, install Python's missing environment tools first, then install the CPU builds of PyTorch and Torchvision:

```bash
sudo apt update
sudo apt install python3.14-venv python3-pip
python3 -m venv .venv
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
.venv/bin/python train.py --epochs 5 --seed 1 --output artifacts/model.pt
```

The [PyTorch install selector](https://pytorch.org/get-started/locally/) provides a CUDA command if GPU access is available. `train.py` uses CUDA when available, or the CPU otherwise. The resulting `artifacts/model.pt` contains the four trained tensors:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `fc1.weight` | 16 × 196 | Input-to-hidden weights |
| `fc1.bias` | 16 | Hidden biases |
| `fc2.weight` | 10 × 16 | Hidden-to-output weights |
| `fc2.bias` | 10 | Output biases |

Inspect them directly with PyTorch:

```python
import torch

weights = torch.load("artifacts/model.pt", map_location="cpu", weights_only=True)
for name, tensor in weights.items():
    print(name, tuple(tensor.shape), tensor.flatten()[:5].tolist())
```

For example, `weights["fc1.weight"][3, 7]` is the weight from input 7 to hidden unit 3. `weights["fc2.bias"][5]` is the bias for digit 5's output score.

## Export quantized weights

```bash
.venv/bin/python quantize.py \
  --checkpoint artifacts/model.pt \
  --bits 2 4 8 \
  --output-dir artifacts/quantized
```

This writes `weights_2bit.json`, `weights_4bit.json`, and `weights_8bit.json`. Each file contains two layers of integer `weights` and `biases`, the scale for each layer, and floating-point and quantized test accuracy. The same trained checkpoint is used for every precision.

Each layer's `fraction_bits` value `F` defines the weight scale: approximately `floating_weight = integer_weight / 2^F`. The integer biases use a scale of `1 / (255 × 2^F)` because model inputs are pixel values divided by 255. The script chooses `F` to minimize weight reconstruction error and checks that its integer calculations fit a signed 32-bit accumulator. Hidden outputs are rounded and clamped to eight bits when measuring quantized accuracy.

This is post-training quantization: train once, then convert the saved weights. If low-bit accuracy is poor, quantization-aware training is a possible next experiment; [PyTorch's quantization documentation](https://docs.pytorch.org/docs/main/quantization-support.html) describes that approach.

The MNIST download and generated files live under `data/` and `artifacts/`, which Git ignores.
