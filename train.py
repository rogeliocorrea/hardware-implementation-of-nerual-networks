"""Train the floating-point MNIST baseline; save the best validation model."""

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import DigitMLP, pooled_pixels


def accuracy(model: DigitMLP, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            pixels = pooled_pixels(images).to(device)
            predictions = model(pixels).argmax(dim=1)
            correct += (predictions == labels.to(device)).sum().item()
            total += labels.numel()
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(device_name)

    dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transforms.PILToTensor())
    split_rng = torch.Generator().manual_seed(args.seed)
    train_set, validation_set = random_split(dataset, [55_000, 5_000], generator=split_rng)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size)

    model = DigitMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in train_loader:
            pixels = pooled_pixels(images).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(pixels), labels.to(device))
            loss.backward()
            optimizer.step()
        validation_accuracy = accuracy(model, validation_loader, device)
        print(f"epoch={epoch} validation_accuracy={validation_accuracy:.5f}")
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            cpu_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            torch.save(cpu_state, args.output)

    manifest = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": device_name,
        "validation_accuracy": best_accuracy,
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
