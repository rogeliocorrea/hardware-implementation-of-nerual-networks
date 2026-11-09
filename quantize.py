"""Export quantized PyTorch weights and biases and measure MNIST accuracy."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DigitMLP, pooled_pixels


def quantize_layer(weights: torch.Tensor, biases: torch.Tensor, bits: int) -> dict:
    if bits < 2 or bits > 8:
        raise ValueError("this experiment supports 2 through 8 weight bits")
    qmax = (1 << (bits - 1)) - 1
    # A few outliers can make maximum-range scaling poor at 2 bits. Choose
    # the power-of-two scale minimizing weight reconstruction error instead.
    source = weights.detach().cpu()
    candidates = range(17)
    fraction_bits = min(
        candidates,
        key=lambda candidate: (
            (torch.round(source * (1 << candidate)).clamp(-qmax, qmax) / (1 << candidate) - source)
            .square()
            .mean()
            .item()
        ),
    )
    scale = 1 << fraction_bits
    integer_weights = torch.round(source * scale).clamp(-qmax, qmax).to(torch.int32)
    integer_biases = torch.round(biases.detach().cpu() * 255 * scale).to(torch.int64)
    if integer_biases.abs().max().item() >= 2**31:
        raise OverflowError("quantized bias exceeds signed 32-bit range")
    worst_case = integer_biases.abs() + integer_weights.abs().to(torch.int64).sum(dim=1) * 255
    if worst_case.max().item() >= 2**31:
        raise OverflowError("worst-case dot product exceeds signed 32-bit range")
    return {
        "fraction_bits": fraction_bits,
        "weights": integer_weights.tolist(),
        "biases": integer_biases.tolist(),
    }


def integer_logits(pixels: torch.Tensor, layers: list[dict]) -> torch.Tensor:
    inputs = pixels.to(torch.int64)
    for layer_number, layer in enumerate(layers):
        weights = torch.tensor(layer["weights"], dtype=torch.int64)
        biases = torch.tensor(layer["biases"], dtype=torch.int64)
        accumulators = inputs @ weights.T + biases
        if accumulators.abs().max().item() >= 2**31:
            raise OverflowError("dot product exceeds signed 32-bit accumulator")
        if layer_number == 0:
            nonnegative = accumulators.clamp_min(0)
            shift = layer["fraction_bits"]
            inputs = ((nonnegative + (1 << (shift - 1))) >> shift) if shift else nonnegative
            inputs = inputs.clamp_max(255)
        else:
            return accumulators
    raise ValueError("expected two layers")


def evaluate(model: DigitMLP, layers: list[dict], loader: DataLoader) -> tuple[float, float]:
    float_correct = integer_correct = total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            pixels = pooled_pixels(images)
            float_correct += (model(pixels).argmax(dim=1) == labels).sum().item()
            integer_correct += (integer_logits(pixels, layers).argmax(dim=1) == labels).sum().item()
            total += labels.numel()
    return float_correct / total, integer_correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/quantized"))
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1 or any(bits < 2 or bits > 8 for bits in args.bits):
        parser.error("batch size must be positive and weight bits must be from 2 to 8")

    model = DigitMLP()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=transforms.PILToTensor())
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for bits in args.bits:
        layers = [quantize_layer(layer.weight, layer.bias, bits) for layer in (model.fc1, model.fc2)]
        float_accuracy, integer_accuracy = evaluate(model, layers, test_loader)
        artifact = {
            "format": "mnist_quantized_v1",
            "architecture": "196-16-10",
            "weight_bits": bits,
            "input_bits": 8,
            "activation_bits": 8,
            "bias_bits": 32,
            "accumulator_bits": 32,
            "rounding": "half_up_after_relu",
            "checkpoint_sha256": checkpoint_hash,
            "float_test_accuracy": float_accuracy,
            "integer_test_accuracy": integer_accuracy,
            "layers": layers,
        }
        destination = args.output_dir / f"weights_{bits}bit.json"
        destination.write_text(json.dumps(artifact, separators=(",", ":")) + "\n")
        print(f"bits={bits} float_accuracy={float_accuracy:.5f} integer_accuracy={integer_accuracy:.5f} output={destination}")


if __name__ == "__main__":
    main()
