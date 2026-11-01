"""Unified trip container + geo helpers.

Every data source (synthetic, Sensor Logger, IO-VNBD) is loaded into a Trip so the
rest of the pipeline never needs to know where the data came from.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

R_EARTH = 6_371_000.0
CANONICAL_HZ = 10.0  # IO-VNBD's native rate; everything is resampled to this


@dataclass
class Trip:
    """One drive, resampled onto a uniform time grid.

    All arrays share length N. Device-frame arrays are raw sensor axes; the
    alignment stage converts them to the vehicle frame.
    """

    name: str
    t: np.ndarray  # (N,)   seconds, uniform spacing
    acc: np.ndarray  # (N,3) m/s^2, device frame, gravity INCLUDED
    gyr: np.ndarray  # (N,3) rad/s, device frame
    lat: np.ndarray  # (N,)   degrees
    lon: np.ndarray  # (N,)   degrees
    v_true: np.ndarray  # (N,) m/s forward speed — the training teacher
    course: np.ndarray  # (N,) rad, GPS course over ground, 0=North, CW positive
    hz: float = CANONICAL_HZ

    def __post_init__(self) -> None:
        n = len(self.t)
        for field in ("acc", "gyr"):
            arr = getattr(self, field)
            if arr.shape != (n, 3):
                raise ValueError(f"{self.name}: {field} is {arr.shape}, expected {(n, 3)}")
        for field in ("lat", "lon", "v_true", "course"):
            if len(getattr(self, field)) != n:
                raise ValueError(f"{self.name}: {field} length != {n}")

    def __len__(self) -> int:
        return len(self.t)

    @property
    def dt(self) -> float:
        return 1.0 / self.hz

    def enu(self) -> tuple[np.ndarray, np.ndarray]:
        """Ground-truth position in local East/North metres, origin at trip start.

        Equirectangular projection — accurate to well under a metre over the few
        kilometres a single trip covers, which is far below our error budget.
        """
        lat0 = np.deg2rad(self.lat[0])
        east = np.deg2rad(self.lon - self.lon[0]) * R_EARTH * np.cos(lat0)
        north = np.deg2rad(self.lat - self.lat[0]) * R_EARTH
        return east, north

    def distance_travelled(self, i0: int, i1: int) -> float:
        """Ground-truth path length between two sample indices, in metres."""
        return float(np.sum(self.v_true[i0:i1]) * self.dt)


def resample_uniform(
    t: np.ndarray, arrays: dict[str, np.ndarray], hz: float = CANONICAL_HZ
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Put irregularly-timestamped sensor data onto a uniform grid.

    Phone sensor timestamps jitter and occasionally drop samples; every downstream
    stage assumes a fixed dt, so this has to happen before anything else.
    """
    t = np.asarray(t, dtype=float)
    order = np.argsort(t)
    t = t[order]
    t_uniform = np.arange(t[0], t[-1], 1.0 / hz)

    out: dict[str, np.ndarray] = {}
    for key, arr in arrays.items():
        arr = np.asarray(arr, dtype=float)[order]
        if arr.ndim == 1:
            out[key] = np.interp(t_uniform, t, arr)
        else:
            out[key] = np.column_stack(
                [np.interp(t_uniform, t, arr[:, c]) for c in range(arr.shape[1])]
            )
    return t_uniform, out


def unwrap_course(course: np.ndarray) -> np.ndarray:
    """Unwrap a heading series so interpolation across the 0/2pi seam behaves."""
    return np.unwrap(np.asarray(course, dtype=float))


def course_to_enu_heading(course: np.ndarray) -> np.ndarray:
    """GPS course (0=North, clockwise) -> ENU yaw (0=East, counter-clockwise)."""
    return np.pi / 2.0 - course


def wrap_pi(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) into [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi
