"""Loader for comma2k19 (HuggingFace parquet form).

Why this dataset is here: it is the only public set shaped for the one experiment we
cannot run on IO-VNBD — does a HIGHER SAMPLE RATE actually help? comma2k19 gives
110 Hz accelerometer with dense CAN speed labels on the same drives, so the rate can
be varied while everything else is held fixed.

What it does NOT give is vehicle diversity: the whole dataset is two cars (a 2016
Honda Civic and a 2017 Toyota RAV4), and the 64-segment demo subset is one of them.
So it answers "does rate matter", never "does it generalise across vehicles".

Measured facts about the data, none of which are in the docs:

  accelerometer   110 Hz
  gyro             20 Hz    <- much slower than the accelerometer, and it is the
                               heading source, so heading resolution is capped here
  CAN speed        83 Hz, metres per second (verified against pose velocity)
  wheel speed      83 Hz, four wheels, m/s
  global pose      20 Hz, ECEF metres — ground truth, better than GPS
  segments         1 minute each; consecutive ones share a route and must be joined
"""
from __future__ import annotations

import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .trip import Trip, unwrap_course

# WGS84
_A = 6_378_137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2 - _F)

SEGMENT_SECONDS = 60.0


def ecef_to_geodetic(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ECEF metres -> (latitude, longitude) in degrees, via Bowring's method."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    b = _A * (1 - _F)
    theta = np.arctan2(z * _A, p * b)
    ep2 = (_A**2 - b**2) / b**2
    lat = np.arctan2(
        z + ep2 * b * np.sin(theta) ** 3,
        p - _E2 * _A * np.cos(theta) ** 3,
    )
    return np.rad2deg(lat), np.rad2deg(lon)


def _get(log: dict, key: str) -> np.ndarray | None:
    v = log.get(key)
    if v is None or len(v) == 0:
        return None
    return np.asarray(v, dtype=float)


def _segment_streams(log: dict) -> dict | None:
    """Pull the streams we need out of one 1-minute segment."""
    t_acc = _get(log, "processed_log__IMU__accelerometer__t")
    v_acc = _get(log, "processed_log__IMU__accelerometer__value")
    t_gyr = _get(log, "processed_log__IMU__gyro__t")
    v_gyr = _get(log, "processed_log__IMU__gyro__value")
    t_spd = _get(log, "processed_log__CAN__speed__t")
    v_spd = _get(log, "processed_log__CAN__speed__value")
    t_pos = _get(log, "global_pose__frame_times")
    v_pos = _get(log, "global_pose__frame_positions")
    v_vel = _get(log, "global_pose__frame_velocities")
    if any(a is None for a in (t_acc, v_acc, t_gyr, v_gyr, t_spd, v_spd, t_pos, v_pos, v_vel)):
        return None
    return {
        "t_acc": t_acc, "acc": v_acc,
        "t_gyr": t_gyr, "gyr": v_gyr,
        "t_spd": t_spd, "spd": v_spd.ravel(),
        "t_pos": t_pos, "pos": v_pos, "vel": v_vel,
    }


def ecef_vector_to_enu(vec: np.ndarray, lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Rotate ECEF vectors into local East/North/Up at each point."""
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    sl, cl = np.sin(lon), np.cos(lon)
    sp, cp = np.sin(lat), np.cos(lat)
    east = -sl * vec[:, 0] + cl * vec[:, 1]
    north = -sp * cl * vec[:, 0] - sp * sl * vec[:, 1] + cp * vec[:, 2]
    up = cp * cl * vec[:, 0] + cp * sl * vec[:, 1] + sp * vec[:, 2]
    return np.column_stack([east, north, up])


def course_from_velocity(
    vel_enu: np.ndarray, speed: np.ndarray, min_speed: float = 1.5
) -> np.ndarray:
    """Heading from the ground-truth velocity vector, held through slow sections.

    Differencing positions instead looks reasonable and is badly wrong: when the
    vehicle is slow or stopped the position deltas are pure noise, so the derived
    heading spins randomly. Measured here, that produced apparent yaw rates of
    75 deg/s — a car would be spinning — and made the real gyro look uncorrelated
    with "truth" when the truth was the broken quantity.
    """
    course = np.arctan2(vel_enu[:, 0], vel_enu[:, 1])  # 0 = North, clockwise
    trusted = speed > min_speed
    if not trusted.any():
        return np.zeros_like(course)
    # Hold the last trustworthy heading across stops rather than letting it wander.
    idx = np.where(trusted, np.arange(len(course)), 0)
    np.maximum.accumulate(idx, out=idx)
    return course[idx]


def _route_to_trip(segments: list[dict], name: str, hz: float) -> Trip | None:
    """Join consecutive 1-minute segments into one continuous Trip.

    Timestamps are absolute within a route, so segments concatenate directly. A
    route with a gap (a missing segment) is still joined — the resampling grid keeps
    real time, so the gap appears as interpolation over a short span rather than as
    a discontinuity in the trajectory.
    """
    cat = {k: np.concatenate([s[k] for s in segments]) for k in segments[0]}
    # Each value stream shares a clock with exactly one time stream; sort together.
    for time_key, value_keys in (
        ("t_acc", ("acc",)), ("t_gyr", ("gyr",)),
        ("t_spd", ("spd",)), ("t_pos", ("pos", "vel")),
    ):
        order = np.argsort(cat[time_key])
        cat[time_key] = cat[time_key][order]
        for vk in value_keys:
            cat[vk] = cat[vk][order]

    t0 = max(cat["t_acc"][0], cat["t_gyr"][0], cat["t_spd"][0], cat["t_pos"][0])
    t1 = min(cat["t_acc"][-1], cat["t_gyr"][-1], cat["t_spd"][-1], cat["t_pos"][-1])
    if t1 - t0 < 30:
        return None
    grid = np.arange(t0, t1, 1.0 / hz)

    def interp(ts, vs):
        if vs.ndim == 1:
            return np.interp(grid, ts, vs)
        return np.column_stack([np.interp(grid, ts, vs[:, c]) for c in range(vs.shape[1])])

    lat, lon = ecef_to_geodetic(interp(cat["t_pos"], cat["pos"]))
    speed = np.clip(interp(cat["t_spd"], cat["spd"]), 0.0, None)  # CAN speed, m/s
    vel_enu = ecef_vector_to_enu(interp(cat["t_pos"], cat["vel"]), lat, lon)
    course = course_from_velocity(vel_enu, speed)

    return Trip(
        name=name,
        t=grid - grid[0],
        acc=interp(cat["t_acc"], cat["acc"]),
        gyr=interp(cat["t_gyr"], cat["gyr"]),
        lat=lat,
        lon=lon,
        v_true=speed,
        course=unwrap_course(course),
        hz=hz,
    )


def load_comma2k19(
    root: str | Path = "data/comma_demo",
    hz: float = 10.0,
    min_seconds: float = 60.0,
    limit: int | None = None,
) -> list[Trip]:
    """Load comma2k19 parquet into Trips, one per route.

    `hz` is the resampling rate and is the knob the whole experiment turns. The
    accelerometer is natively 110 Hz, so anything up to ~50 Hz is real signal rather
    than interpolation; above that you are mostly inventing samples.
    """
    paths = sorted(glob.glob(str(Path(root) / "**" / "*.parquet"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"no parquet under {root}")

    routes: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for path in paths:
        table = pq.read_table(path, columns=["segment_id", "log"])
        for row in table.to_pylist():
            seg_id = row["segment_id"]
            route, _, index = seg_id.rpartition("/")
            streams = _segment_streams(row["log"])
            if streams is not None:
                routes[route].append((int(index), streams))

    trips: list[Trip] = []
    for route, segs in sorted(routes.items()):
        segs.sort(key=lambda pair: pair[0])
        trip = _route_to_trip([s for _, s in segs], name=f"comma-{route.split('|')[-1]}", hz=hz)
        if trip is not None and len(trip) / hz >= min_seconds:
            trips.append(trip)
        if limit and len(trips) >= limit:
            break
    return trips


def vehicles(root: str | Path = "data/comma_demo") -> set[str]:
    """Distinct dongle ids present, i.e. distinct vehicles."""
    ids: set[str] = set()
    for path in sorted(glob.glob(str(Path(root) / "**" / "*.parquet"), recursive=True)):
        for seg in pq.read_table(path, columns=["segment_id"]).column(0).to_pylist():
            ids.add(seg.split("|")[0])
    return ids
