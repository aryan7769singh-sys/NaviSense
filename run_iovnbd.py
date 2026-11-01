"""Train and benchmark on real IO-VNBD data.

    python run_iovnbd.py

Caps each recording so trip-level diversity beats raw hours: 24 trips of 20 minutes
generalises better than 6 trips of 80, and trips are the unit we split on.
"""
from __future__ import annotations

import pickle
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from proto.dr import benchmark, format_summary, prepare
from proto.figures import drift_figure, pick_representative, speed_figure, trajectory_figure
from proto.loaders import load_iovnbd_dir
from proto.velnet import split_by_trip, train

ARTIFACTS = Path("artifacts")
MAX_MINUTES = 20.0   # 72 unique recordings x 20 min = ~24 h of driving
N_TRIPS = None       # every recording in the dataset
STRIDE = 5


def truncate(trip, minutes: float):
    n = int(minutes * 60 * trip.hz)
    if len(trip) <= n:
        return trip
    return replace(
        trip,
        t=trip.t[:n],
        acc=trip.acc[:n],
        gyr=trip.gyr[:n],
        lat=trip.lat[:n],
        lon=trip.lon[:n],
        v_true=trip.v_true[:n],
        course=trip.course[:n],
    )


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    t0 = time.time()

    print(f"Loading ALL IO-VNBD recordings, {MAX_MINUTES:.0f} min each...", flush=True)
    trips = [truncate(t, MAX_MINUTES) for t in load_iovnbd_dir("data", limit=N_TRIPS)]
    trips = [t for t in trips if t.v_true.max() > 5.0]  # drop recordings that never move
    total_min = sum(len(t) for t in trips) / trips[0].hz / 60
    print(f"  {len(trips)} trips, {total_min:.0f} min ({total_min/60:.1f} h) in {time.time()-t0:.0f}s")
    speeds = np.concatenate([t.v_true for t in trips])
    print(f"    speed range {speeds.min()*3.6:.0f}-{speeds.max()*3.6:.0f} km/h, mean {speeds.mean()*3.6:.0f}")

    tr, va = split_by_trip(trips, val_fraction=0.3, seed=3)
    print(f"\nSplit BY TRIP: {len(tr)} train / {len(va)} val")

    print("Training VelNet on real smartphone IMU, taught by CAN wheel speed...")
    bundle, hist = train(tr, va, epochs=30, stride=STRIDE)

    y_mean = np.concatenate([t.v_true for t in tr]).mean()
    y_val = np.concatenate([t.v_true for t in va])
    print(f"\n  predict-the-mean : {np.sqrt(((y_val - y_mean) ** 2).mean()):5.2f} m/s")
    print(f"  VelNet           : {hist['best_val_rmse']:5.2f} m/s")
    print(f"  CarSpeedNet SOTA : 1.80 m/s (20 Hz data, 4 s window)")

    print("\nSimulating GNSS blackouts on held-out trips...")
    summary, rows = benchmark(va, bundle)
    print("\nREAL IO-VNBD — final position error as % of distance travelled")
    print(format_summary(summary))

    ctx = prepare(va[0], bundle)
    trajectory_figure(pick_representative(rows, 60), ARTIFACTS / "full_trajectory.png")
    drift_figure(summary, ARTIFACTS / "full_drift.png")
    speed_figure(va[0], ctx, ARTIFACTS / "full_speed.png")
    with open(ARTIFACTS / "full_run.pkl", "wb") as fh:
        pickle.dump({"summary": summary, "rows": rows, "hist": hist, "bundle": bundle,
                     "val": va}, fh)

    print(f"\nFigures: artifacts/iovnbd_*.png")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
