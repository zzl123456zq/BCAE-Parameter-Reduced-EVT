"""Synthetic training-set contamination used in the robustness experiment."""

import numpy as np


def build_drift_contaminated_dataset(
    x_train,
    y_train,
    x_test,
    y_test,
    contamination_ratio=0.01,
    contamination_type="mixed",
    seed=0,
    channel_ratio=1.0,
    scale_std=0.05,
    bias_level=0.05,
    trend_level=0.05,
    noise_level=0.01,
    return_info=False,
):
    """Inject reproducible perturbations into selected training sequences.

    Parameters
    ----------
    x_train : array-like, shape (N, L, D)
        Training sequences with sample, time, and sensor-channel dimensions.
    contamination_type : {"sensor_drift", "working_condition", "mixed"}
        In mixed mode, a perturbation family is sampled independently for each
        selected sequence.
    """

    rng = np.random.default_rng(seed)
    x_train_new = np.asarray(x_train, dtype=np.float32).copy()
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    if x_train_new.ndim != 3:
        raise ValueError(
            f"x_train must have shape (N, L, D); got {x_train_new.shape}."
        )

    n_samples, seq_len, n_channels = x_train_new.shape
    n_contam = int(np.floor(contamination_ratio * n_samples))

    if n_contam <= 0:
        if return_info:
            info = {
                "contaminated_indices": np.array([], dtype=int),
                "n_contaminated": 0,
                "contamination_ratio": contamination_ratio,
                "contamination_type": contamination_type,
                "applied_types": [],
            }
            return x_train_new, y_train, x_test, y_test, info
        return x_train_new, y_train, x_test, y_test

    contaminated_indices = rng.choice(
        n_samples, size=n_contam, replace=False
    )

    channel_std = np.std(x_train_new, axis=(0, 1))
    channel_std = np.maximum(channel_std, 1e-8)
    time_coordinate = np.linspace(
        -0.5, 0.5, seq_len, dtype=np.float32
    ).reshape(seq_len, 1)
    n_affected_channels = max(
        1, int(np.round(channel_ratio * n_channels))
    )
    applied_types = []

    for idx in contaminated_indices:
        affected_channels = rng.choice(
            n_channels, size=n_affected_channels, replace=False
        )
        cur_type = (
            rng.choice(["sensor_drift", "working_condition"])
            if contamination_type == "mixed"
            else contamination_type
        )
        applied_types.append(cur_type)

        sample = x_train_new[idx].copy()
        block = sample[:, affected_channels]
        std = channel_std[affected_channels].reshape(1, -1)

        scale = rng.normal(
            loc=1.0,
            scale=scale_std,
            size=(1, len(affected_channels)),
        ).astype(np.float32)
        bias = rng.normal(
            loc=0.0,
            scale=bias_level * std,
            size=(1, len(affected_channels)),
        ).astype(np.float32)

        if cur_type == "sensor_drift":
            trend_amp = rng.normal(
                loc=0.0,
                scale=trend_level * std,
                size=(1, len(affected_channels)),
            ).astype(np.float32)
            perturbed_block = (
                block * scale + bias + time_coordinate * trend_amp
            )
        elif cur_type == "working_condition":
            noise = rng.normal(
                loc=0.0,
                scale=noise_level * std,
                size=(seq_len, len(affected_channels)),
            ).astype(np.float32)
            perturbed_block = block * scale + bias + noise
        else:
            raise ValueError(
                "contamination_type must be sensor_drift, "
                "working_condition, or mixed."
            )

        sample[:, affected_channels] = perturbed_block
        x_train_new[idx] = sample

    if return_info:
        info = {
            "contaminated_indices": contaminated_indices,
            "n_contaminated": n_contam,
            "contamination_ratio": contamination_ratio,
            "contamination_type": contamination_type,
            "applied_types": applied_types,
        }
        return x_train_new, y_train, x_test, y_test, info

    return x_train_new, y_train, x_test, y_test
