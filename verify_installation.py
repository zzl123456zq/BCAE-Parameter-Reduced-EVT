"""Data-free smoke test for the released BCAE and EVT implementation."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from computational_complexity import (
    count_model_parameters,
    measure_inference_latency,
)
from evt import SPOT
from model import FlexibleConv1dAutoencoder, contrastive_loss


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    model = FlexibleConv1dAutoencoder(
        in_channels=8,
        conv_channels=[16, 8],
        kernel_size=3,
        latent_dim=4,
        activation="relu",
    )
    inputs = torch.randn(6, 8, 20)
    reconstructions = model(inputs)

    if reconstructions.shape != inputs.shape:
        raise AssertionError(
            "Unexpected reconstruction shape: "
            f"{reconstructions.shape} != {inputs.shape}"
        )

    loss = contrastive_loss(inputs, reconstructions)
    if not torch.isfinite(loss):
        raise AssertionError("Contrastive loss is not finite.")

    parameter_counts = count_model_parameters(model)
    if parameter_counts["total_params"] <= 0:
        raise AssertionError("Model parameter count must be positive.")

    latency = measure_inference_latency(
        model,
        DataLoader(inputs, batch_size=3),
        torch.device("cpu"),
        warmup=1,
        repeat=2,
    )
    if not np.isfinite(latency) or latency <= 0:
        raise AssertionError("Inference latency must be finite and positive.")

    errors = np.random.default_rng(42).gamma(
        shape=2.0, scale=0.01, size=1000
    )
    threshold, _, _, _, shape, scale = SPOT()._pot(
        errors, anomaly_ratio=0.02, initial_thr_ratio=0.05
    )
    if not np.all(np.isfinite([threshold, shape, scale])):
        raise AssertionError("EVT fitting returned a non-finite value.")

    print("Smoke test passed.")
    print(f"Model output shape: {tuple(reconstructions.shape)}")
    print(f"Contrastive loss: {loss.item():.6f}")
    print(f"Model parameters: {parameter_counts['total_params']}")
    print(f"CPU inference latency: {latency:.6f} ms/sample")
    print(
        "Synthetic EVT fit: "
        f"threshold={threshold:.6f}, shape={shape:.6f}, scale={scale:.6f}"
    )


if __name__ == "__main__":
    main()
