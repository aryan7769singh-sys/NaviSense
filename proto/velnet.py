"""VelNet — forward speed from IMU alone.

The central idea of the submission: the training label is vehicle speed taken from
wheel-speed sensors (IO-VNBD) or GPS Doppler (our own drives), but the NETWORK ONLY
EVER SEES IMU. So the shipped system needs no OBD-II and no speedometer, while
training still gets dense, high-quality supervision.

Causal by construction — the window is [t-W, t], never centred — because a
real-time navigator cannot use future samples.

The second output head is log-variance, not a point estimate. That uncertainty is
what a Kalman filter needs for R; a speed estimate without one cannot be fused
properly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .align import to_vehicle_frame
from .trip import Trip

# 10.0 s at 10 Hz. Measured on real IO-VNBD (exp_window.py), all else held fixed:
#
#     4 s -> 4.582 m/s    7 s -> 4.181    10 s -> 3.912    15 s -> 4.026
#
# The speed cue is a vibration STATISTIC, not an instantaneous value, so more
# samples estimate it better — up to a point. Past ~10 s the vehicle's speed
# changes appreciably *within* the window, which makes "speed at the window's end"
# harder to pin down, and the extra parameters have no more data to learn from.
# arXiv:2505.18490 also settles on 10 s.
WINDOW = 100
# 6 vehicle-frame axes + 2 rotation-INVARIANT magnitudes. The magnitudes matter on
# real data: mount-yaw estimation scored 0.92 confidence on synthetic drives but
# only 0.10-0.25 on real IO-VNBD ones, so the vehicle frame is not reliable there.
# |acc| and |gyr| carry the vibration signature regardless of how the phone is lying.
CHANNELS = 8


class CausalConv1d(nn.Module):
    """Dilated convolution that can only see backwards in time.

    Symmetric padding would let the estimate at time t depend on samples after t,
    which inflates offline metrics and is impossible to ship in a real-time navigator.
    """

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = dilation * (kernel - 1)
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation, padding=self.pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)[:, :, : -self.pad] if self.pad else self.conv(x)


def dilations_for(window: int, kernel: int = 5) -> tuple[int, ...]:
    """Doubling dilations until the receptive field covers the whole window.

    Receptive field = 1 + (kernel-1) * sum(dilations). A stack that cannot see the
    whole window silently ignores its oldest samples, which would make a
    window-length comparison meaningless.
    """
    ds: list[int] = []
    d = 1
    while 1 + (kernel - 1) * sum(ds) < window:
        ds.append(d)
        d *= 2
    return tuple(ds) or (1,)


class VelNet(nn.Module):
    """Dilated causal TCN, sized to the requested window."""

    def __init__(self, channels: int = CHANNELS, width: int = 48, window: int = WINDOW):
        super().__init__()
        self.window = window
        layers: list[nn.Module] = []
        c_in = channels
        for dilation in dilations_for(window):
            layers += [CausalConv1d(c_in, width, 5, dilation), nn.GELU()]
            c_in = width
        self.net = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, C, W) -> (mu, log_var), each (B,)."""
        h = self.net(x)[:, :, -1]  # last step = causal estimate at the window's end
        out = self.head(h)
        mu = torch.nn.functional.softplus(out[:, 0])  # speed cannot be negative
        log_var = out[:, 1].clamp(-6.0, 6.0)
        return mu, log_var


@dataclass
class Bundle:
    """A trained model plus the normalisation it was fitted with."""

    model: VelNet
    mean: np.ndarray
    std: np.ndarray
    window: int = WINDOW

    def predict(self, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """feats: (N, W, C) -> (speed, sigma), each (N,)."""
        x = (feats - self.mean) / self.std
        self.model.eval()
        with torch.no_grad():
            mu, log_var = self.model(torch.tensor(x, dtype=torch.float32).transpose(1, 2))
        return mu.numpy(), np.exp(0.5 * log_var.numpy())


def trip_features(trip: Trip, v_ref: np.ndarray | None = None, window: int = WINDOW) -> np.ndarray:
    """(N, WINDOW, CHANNELS) causal windows. Row i covers samples [i-W+1 .. i].

    The accelerometer channels are mean-removed WITHIN each window. This matters
    more than it looks:

      * The speed cue is road-roughness vibration — an AC quantity. The DC level is
        gravity leakage plus sensor bias, i.e. pure nuisance.
      * That DC level is near-constant per device and per mount, so leaving it in
        lets the model identify which recording it is looking at and recall that
        recording's speed profile. It scores well in training and collapses on any
        held-out trip.

    Gyro is left raw, because its mean IS informative — yaw rate combines with
    lateral acceleration to give speed directly during turns.
    """
    aligned = to_vehicle_frame(trip, v_ref=v_ref)
    acc, gyr = aligned["acc"], aligned["gyr"]
    raw = np.concatenate(
        [
            acc,
            gyr,
            np.linalg.norm(acc, axis=1, keepdims=True),  # rotation-invariant
            np.linalg.norm(gyr, axis=1, keepdims=True),  # rotation-invariant
        ],
        axis=1,
    )  # (N, 8)
    padded = np.vstack([np.repeat(raw[:1], window - 1, axis=0), raw])
    idx = np.arange(len(raw))[:, None] + np.arange(window)[None, :]
    windows = padded[idx].astype(np.float32)  # (N, WINDOW, 8)
    # Mean-remove the accelerometer channels and |acc|: their DC is gravity plus
    # sensor bias, which identifies the recording rather than the speed. Gyro and
    # |gyr| keep their mean, which is genuine turn-rate information.
    windows[:, :, :3] -= windows[:, :, :3].mean(axis=1, keepdims=True)
    windows[:, :, 6:7] -= windows[:, :, 6:7].mean(axis=1, keepdims=True)
    return windows


def build_dataset(
    trips: list[Trip], stride: int = 1, window: int = WINDOW
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack trips into (features, labels, trip_id).

    `stride` subsamples the window START positions. Consecutive 4-second windows at
    10 Hz overlap by 97.5%, so a stride of 5 (0.5 s) discards almost no information
    while cutting memory 5x. Real IO-VNBD trips run to 176 minutes, where stride 1
    would need ~100 MB of float32 per trip.
    """
    feats, labels, ids = [], [], []
    for i, trip in enumerate(trips):
        f = trip_features(trip, window=window)[::stride]
        feats.append(f)
        labels.append(trip.v_true[::stride])
        ids.append(np.full(len(f), i))
    return np.concatenate(feats), np.concatenate(labels), np.concatenate(ids)


def train(
    train_trips: list[Trip],
    val_trips: list[Trip],
    epochs: int = 30,
    batch: int = 256,
    lr: float = 2e-3,
    seed: int = 0,
    stride: int = 1,
    window: int = WINDOW,
    width: int = 48,
    verbose: bool = True,
) -> tuple[Bundle, dict]:
    """Fit VelNet. Splits must be BY TRIP — see note below."""
    torch.manual_seed(seed)

    xtr, ytr, _ = build_dataset(train_trips, stride=stride, window=window)
    xva, yva, _ = build_dataset(val_trips, stride=max(stride, 2), window=window)

    mean = xtr.reshape(-1, CHANNELS).mean(0)
    std = xtr.reshape(-1, CHANNELS).std(0) + 1e-6
    xtr_n = ((xtr - mean) / std).transpose(0, 2, 1)
    xva_n = ((xva - mean) / std).transpose(0, 2, 1)

    xt = torch.tensor(xtr_n, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    xv = torch.tensor(xva_n, dtype=torch.float32)
    yv = torch.tensor(yva, dtype=torch.float32)

    model = VelNet(window=window, width=width)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = {"train_nll": [], "val_rmse": []}
    best_rmse, best_state = float("inf"), None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        # Gaussian NLL has a well-known failure mode: early on, the cheapest way to
        # cut the loss is to inflate log_var rather than fit mu, and once sigma is
        # large the gradient on mu is divided by it and effectively vanishes. The
        # model then "converges" while getting worse. Fit mu with MSE first, and
        # only then let the variance head start learning.
        warmup = epoch < int(0.5 * epochs)

        for i in range(0, len(perm), batch):
            sel = perm[i : i + batch]
            mu, log_var = model(xt[sel])
            sq_err = (yt[sel] - mu) ** 2
            if warmup:
                loss = sq_err.mean()
            else:
                nll = 0.5 * (log_var + sq_err / log_var.exp())
                # Keep a direct MSE term so mu never stops being supervised.
                loss = nll.mean() + 0.5 * sq_err.mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(sel)
        sched.step()

        model.eval()
        with torch.no_grad():
            mu_v, _ = model(xv)
            rmse = float(torch.sqrt(((mu_v - yv) ** 2).mean()))
        history["train_nll"].append(total / len(xt))
        history["val_rmse"].append(rmse)

        if rmse < best_rmse:
            best_rmse = rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  nll {total/len(xt):8.3f}   val speed RMSE {rmse:6.3f} m/s")

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_rmse"] = best_rmse
    return Bundle(model, mean, std, window), history


def split_by_trip(trips: list[Trip], val_fraction: float = 0.3, seed: int = 0):
    """Hold out WHOLE TRIPS, never random windows.

    Windows overlap by 19/20 samples, so a random split would put near-identical
    rows in train and test and report a number that means nothing.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(trips))
    n_val = max(1, int(round(len(trips) * val_fraction)))
    val_idx = set(idx[:n_val].tolist())
    train = [t for i, t in enumerate(trips) if i not in val_idx]
    val = [t for i, t in enumerate(trips) if i in val_idx]
    return train, val
