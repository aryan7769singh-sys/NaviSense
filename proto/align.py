"""In-vehicle alignment: work out how the phone is sitting, whatever the mount.

Three degrees of freedom, solved in two stages:

  roll + pitch   the low-passed accelerometer IS the up-axis, so this is direct.
  yaw            the hard one. The horizontal acceleration during straight-line
                 braking and acceleration points along the vehicle's forward axis,
                 so its principal component recovers the mount yaw.

Note that vehicle yaw RATE is available straight after levelling, before mount yaw
is known — rotating about the vertical axis does not change the vertical component
of the gyro. That is what keeps heading available even when alignment is still
converging.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .trip import Trip

UP = np.array([0.0, 0.0, 1.0])


def lowpass(x: np.ndarray, hz: float, cutoff_hz: float) -> np.ndarray:
    """Zero-phase Butterworth low-pass along axis 0."""
    nyq = hz / 2.0
    wn = min(cutoff_hz / nyq, 0.99)
    sos = butter(2, wn, btype="low", output="sos")
    return sosfiltfilt(sos, x, axis=0)


def estimate_up(acc: np.ndarray, hz: float, cutoff_hz: float = 0.15) -> np.ndarray:
    """Per-sample unit up-vector in the device frame.

    A phone at rest reads +g along its up axis, so the heavily low-passed
    accelerometer is the gravity direction. The cutoff is deliberately low: vehicle
    manoeuvres last seconds, gravity does not change.
    """
    g = lowpass(acc, hz, cutoff_hz)
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    norms = np.where(norms < 1e-6, 1.0, norms)
    return g / norms


def _rotations_to_level(up: np.ndarray) -> np.ndarray:
    """(N,3) unit up-vectors -> (N,3,3) rotations mapping each onto +Z.

    Rodrigues, vectorised. Degenerate cases (already level, or upside down) fall
    back to identity / a 180-degree flip about X.
    """
    n = len(up)
    axis = np.cross(up, UP)
    s = np.linalg.norm(axis, axis=1)
    c = up @ UP

    out = np.tile(np.eye(3), (n, 1, 1))
    ok = s > 1e-8
    if np.any(ok):
        k = axis[ok] / s[ok, None]
        kx = np.zeros((ok.sum(), 3, 3))
        kx[:, 0, 1], kx[:, 0, 2] = -k[:, 2], k[:, 1]
        kx[:, 1, 0], kx[:, 1, 2] = k[:, 2], -k[:, 0]
        kx[:, 2, 0], kx[:, 2, 1] = -k[:, 1], k[:, 0]
        # Rodrigues with a UNIT axis needs BOTH trig factors:
        #     R = I + sin(theta) [k]x + (1 - cos(theta)) [k]x^2
        # with sin(theta) = |a x b| and cos(theta) = a . b.
        #
        # Dropping the sin(theta) (and folding an s^2 into the second term) happens
        # to be correct at theta = 90 deg and catastrophically wrong near theta = 0:
        # it adds a ~1 radian rotation to an already-level device. Synthetic drives
        # used random SO(3) mounts, i.e. large angles, and hid it. Real phones lie
        # nearly flat, which is precisely the failing case.
        sin_t = s[ok][:, None, None]
        one_minus_cos = (1.0 - c[ok])[:, None, None]
        out[ok] = np.eye(3) + sin_t * kx + one_minus_cos * (kx @ kx)

    flipped = (~ok) & (c < 0)
    if np.any(flipped):
        out[flipped] = np.diag([1.0, -1.0, -1.0])
    return out


def level(acc: np.ndarray, gyr: np.ndarray, hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate device-frame data into a gravity-levelled frame (Z up, yaw arbitrary)."""
    rot = _rotations_to_level(estimate_up(acc, hz))
    return np.einsum("nij,nj->ni", rot, acc), np.einsum("nij,nj->ni", rot, gyr)


def estimate_mount_yaw(
    acc_level: np.ndarray,
    gyr_level: np.ndarray,
    v_ref: np.ndarray,
    hz: float,
    min_accel: float = 0.35,
    max_yaw_rate: float = 0.06,
    min_speed: float = 2.0,
) -> tuple[float, float]:
    """Recover the yaw offset between the levelled frame and vehicle-forward.

    Returns (yaw_rad, confidence) where confidence is the PCA eigenvalue ratio —
    near 1 means the acceleration was strongly one-dimensional and the estimate is
    trustworthy; near 0 means there was nothing to work with (e.g. constant speed
    the whole trip) and the caller should keep the previous estimate.
    """
    a_h = acc_level[:, :2]
    a_h = a_h - a_h.mean(axis=0)  # residual tilt leaks a DC term into horizontal accel

    dv = np.gradient(np.asarray(v_ref, dtype=float), 1.0 / hz)
    mask = (
        (np.linalg.norm(a_h, axis=1) > min_accel)
        & (np.abs(gyr_level[:, 2]) < max_yaw_rate)
        & (v_ref > min_speed)
    )
    if mask.sum() < 20:
        return 0.0, 0.0

    sel = a_h[mask]
    cov = np.cov(sel.T)
    evals, evecs = np.linalg.eigh(cov)
    principal = evecs[:, np.argmax(evals)]
    confidence = float(1.0 - evals.min() / max(evals.max(), 1e-12))

    # The principal axis is a line, not a direction. Longitudinal acceleration and
    # the derivative of speed must agree in sign, which picks the forward end.
    if np.dot(sel @ principal, dv[mask]) < 0:
        principal = -principal

    return float(np.arctan2(principal[1], principal[0])), confidence


def yaw_rotation(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def to_vehicle_frame(trip: Trip, v_ref: np.ndarray | None = None) -> dict:
    """Full alignment for one trip.

    v_ref is the speed signal used only to disambiguate forward from backward. GPS
    speed is the honest choice — alignment happens while GNSS is still available,
    never during the blackout we are being graded on.
    """
    acc_l, gyr_l = level(trip.acc, trip.gyr, trip.hz)
    ref = trip.v_true if v_ref is None else v_ref
    psi_mount, confidence = estimate_mount_yaw(acc_l, gyr_l, ref, trip.hz)

    rot = yaw_rotation(psi_mount)
    return {
        "acc": acc_l @ rot.T,
        "gyr": gyr_l @ rot.T,
        "yaw_rate": gyr_l[:, 2],  # unaffected by mount yaw — available immediately
        "mount_yaw": psi_mount,
        "mount_confidence": confidence,
    }
