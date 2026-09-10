"""Utilities for reporting model size and inference latency."""

import time

import torch


def _extract_inputs(batch):
    """Return the input tensor from a tensor or a DataLoader batch."""
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def _synchronize(device):
    """Synchronize CUDA work when timing GPU execution."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_inference_latency(
    model,
    test_loader,
    device,
    x_reshape=False,
    warmup=20,
    repeat=100,
):
    """Measure mean forward-pass latency in milliseconds per sample.

    One batch is transferred to the target device before timing. CUDA is
    synchronized after warm-up and after the timed loop so that asynchronous
    kernel execution is included in the measurement.
    """
    if warmup < 0:
        raise ValueError("warmup must be non-negative.")
    if repeat <= 0:
        raise ValueError("repeat must be positive.")

    try:
        batch = next(iter(test_loader))
    except StopIteration as exc:
        raise ValueError("test_loader must contain at least one batch.") from exc

    inputs = _extract_inputs(batch).to(device)
    if x_reshape:
        inputs = inputs.reshape(inputs.size(0), -1)

    was_training = model.training
    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            model(inputs)
        _synchronize(device)

        start = time.perf_counter()
        for _ in range(repeat):
            model(inputs)
        _synchronize(device)
        elapsed = time.perf_counter() - start

    model.train(was_training)
    return elapsed * 1000.0 / (repeat * inputs.size(0))


def count_model_parameters(model):
    """Return total and trainable parameter counts."""
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_M": total_params / 1e6,
        "trainable_params_M": trainable_params / 1e6,
    }
