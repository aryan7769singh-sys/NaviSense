"""Loader for our own IDR Logger captures.

Format written by the Android app: one row per sensor event, kind-tagged.

    A/G/M,t,ax,ay,az,gx,gy,gz,mx,my,mz     each row carries the LATEST of every
    P,t,lat,lon,alt,speed,bearing,acc,sats stream, tagged by which one fired
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .trip import CANONICAL_HZ, Trip, unwrap_course


class BadCapture(ValueError):
    """A recording that must not be used for training or evaluation."""


def audit(path: str | Path) -> dict:
    """Assess one capture without loading it, and say plainly whether it is usable.

    Every check here corresponds to a real failure seen in the first three
    recordings, all of which looked fine as files:

      * The accelerometer stopped at 324 s while GNSS ran to 1888 s, because the
        service held no wake lock and the CPU suspended. The loader trims to the
        overlap, so this does not corrupt anything — it silently throws away 83% of
        the drive instead.
      * GNSS speed was 0.000 for 74-100% of fixes, so there were no training labels
        at all, and the positions wandered 754 m while travelling 38 m.
      * Median GNSS accuracy was 28 m, i.e. indoors.

    None of that raises an exception anywhere. It just quietly produces a model
    trained on noise, which is worse than no data because you will trust it.
    """
    df = pd.read_csv(path, comment="#", on_bad_lines="skip")
    df = df[df["kind"].isin(["A", "G", "M", "P"])]
    rep: dict = {"file": Path(path).name, "fatal": [], "warn": []}

    def times(kind: str) -> np.ndarray:
        t = df.loc[df["kind"] == kind, "t"].to_numpy(float)
        t = t[np.isfinite(t)]
        return np.sort(t)

    t_a, t_g, t_p = times("A"), times("G"), times("P")
    if len(t_a) < 100 or len(t_g) < 100 or len(t_p) < 5:
        rep["fatal"].append(f"too few rows (A={len(t_a)} G={len(t_g)} P={len(t_p)})")
        return rep

    span = max(t_a[-1], t_p[-1]) - min(t_a[0], t_p[0])
    overlap = min(t_a[-1], t_g[-1], t_p[-1]) - max(t_a[0], t_g[0], t_p[0])
    rep.update(
        span_s=float(span),
        overlap_s=float(overlap),
        acc_hz=float(len(t_a) / max(t_a[-1] - t_a[0], 1e-6)),
        imu_ended_s=float(t_a[-1]),
        gnss_ended_s=float(t_p[-1]),
    )

    if overlap < 60:
        rep["fatal"].append(f"only {overlap:.0f}s where IMU and GNSS both exist")
    if overlap < 0.5 * span:
        rep["warn"].append(
            f"IMU stopped at {t_a[-1]:.0f}s but GNSS ran to {t_p[-1]:.0f}s "
            f"— {100 * (1 - overlap / span):.0f}% of the recording is unusable "
            f"(the classic missing-wake-lock signature)"
        )

    gaps = np.diff(t_a)
    if len(gaps) and gaps.max() > 2.0:
        rep["warn"].append(f"largest IMU gap {gaps.max():.0f}s — the CPU slept mid-recording")

    p = df[df["kind"] == "P"]
    speed = p["v4"].to_numpy(float)
    speed = speed[np.isfinite(speed)]
    acc_m = p["v6"].to_numpy(float)
    acc_m = acc_m[np.isfinite(acc_m)]
    sats = p["v7"].to_numpy(float)
    sats = sats[np.isfinite(sats)]

    rep.update(
        max_speed_kmh=float(speed.max() * 3.6) if len(speed) else 0.0,
        moving_frac=float((speed > 2.0).mean()) if len(speed) else 0.0,
        gnss_accuracy_m=float(np.median(acc_m)) if len(acc_m) else float("nan"),
        satellites=float(np.median(sats)) if len(sats) else 0.0,
    )

    if rep["max_speed_kmh"] < 15:
        rep["fatal"].append(
            f"peak speed {rep['max_speed_kmh']:.0f} km/h — this is not a drive, and "
            f"GNSS speed is the training label, so there is nothing to learn from"
        )
    elif rep["moving_frac"] < 0.5:
        rep["warn"].append(f"only {100 * rep['moving_frac']:.0f}% of fixes are above 2 m/s")
    if rep["gnss_accuracy_m"] > 15:
        rep["warn"].append(
            f"median GNSS accuracy {rep['gnss_accuracy_m']:.0f} m — drive in open sky"
        )

    rep["usable"] = not rep["fatal"]
    return rep


def audit_dir(root: str | Path) -> list[dict]:
    """Audit every capture in a folder and print a verdict per file."""
    reports = [audit(p) for p in sorted(Path(root).glob("*.csv"))]
    for r in reports:
        mark = "OK  " if r.get("usable") else "BAD "
        extra = ""
        if "max_speed_kmh" in r:
            extra = (
                f"{r['overlap_s']:5.0f}s usable  {r['acc_hz']:5.0f}Hz  "
                f"peak {r['max_speed_kmh']:5.1f}km/h  gnss±{r['gnss_accuracy_m']:.0f}m"
            )
        print(f"{mark}{r['file']:<34} {extra}")
        for m in r["fatal"]:
            print(f"      FATAL: {m}")
        for m in r["warn"]:
            print(f"      warn : {m}")
    return reports


def load_own(
    path: str | Path,
    hz: float = CANONICAL_HZ,
    name: str | None = None,
    strict: bool = True,
) -> Trip:
    """Load one capture, resampling every stream onto a common uniform grid.

    Streams arrive at wildly different rates here (402 Hz accel, 201 Hz gyro,
    100 Hz mag, ~1 Hz GNSS), so each is interpolated independently rather than
    assuming rows line up.

    With strict=True (the default) an unusable recording raises instead of quietly
    returning a short or unlabelled Trip. See audit() for what counts as unusable.
    """
    if strict:
        rep = audit(path)
        if not rep.get("usable", False):
            raise BadCapture(f"{Path(path).name}: " + "; ".join(rep["fatal"]))
        for msg in rep["warn"]:
            print(f"  warn {Path(path).name}: {msg}")

    df = pd.read_csv(path, comment="#", on_bad_lines="skip")
    df = df[df["kind"].isin(["A", "G", "M", "P"])]

    def stream(kind: str, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
        d = df[df["kind"] == kind]
        t = d["t"].to_numpy(float)
        ok = np.isfinite(t)
        for c in cols:
            ok &= np.isfinite(d[c].to_numpy(float))
        t = t[ok]
        v = np.column_stack([d[c].to_numpy(float)[ok] for c in cols])
        order = np.argsort(t)
        return t[order], v[order]

    t_a, acc = stream("A", ["v1", "v2", "v3"])
    t_g, gyr = stream("G", ["v4", "v5", "v6"])
    t_p, gps = stream("P", ["v1", "v2", "v4", "v5"])  # lat, lon, speed, bearing

    if len(t_a) < 100:
        raise ValueError(f"{Path(path).name}: only {len(t_a)} accelerometer rows")
    if len(t_g) < 100:
        raise ValueError(f"{Path(path).name}: only {len(t_g)} gyro rows — sensor did not start")
    if len(t_p) < 5:
        raise ValueError(f"{Path(path).name}: only {len(t_p)} GNSS fixes")

    t0 = max(t_a[0], t_g[0], t_p[0])
    t1 = min(t_a[-1], t_g[-1], t_p[-1])
    grid = np.arange(t0, t1, 1.0 / hz)

    def interp(ts, vs):
        return np.column_stack([np.interp(grid, ts, vs[:, c]) for c in range(vs.shape[1])])

    acc_i = interp(t_a, acc)
    gyr_i = interp(t_g, gyr)
    gps_i = interp(t_p, np.column_stack([gps[:, 0], gps[:, 1], gps[:, 2],
                                         unwrap_course(np.deg2rad(gps[:, 3]))]))

    return Trip(
        name=name or Path(path).stem,
        t=grid - grid[0],
        acc=acc_i,
        gyr=gyr_i,
        lat=gps_i[:, 0],
        lon=gps_i[:, 1],
        v_true=np.clip(gps_i[:, 2], 0.0, None),   # GNSS Doppler speed
        course=gps_i[:, 3],
        hz=hz,
    )
