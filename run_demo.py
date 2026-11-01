"""NaviSense research demo — one command, end to end.

    python run_demo.py                          # synthetic data, no downloads needed
    python run_demo.py --data sensorlogger --path data/drives
    python run_demo.py --data iovnbd --path data/IO-VNBD

Writes a benchmark table to stdout and three figures to artifacts/.
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from proto.dr import benchmark, format_summary, prepare
from proto.figures import drift_figure, pick_representative, speed_figure, trajectory_figure
from proto.velnet import split_by_trip, train

ARTIFACTS = Path("artifacts")


def load_trips(source: str, path: str | None, n_synth: int):
    if source == "synth":
        from proto.synth import make_dataset

        print(f"Generating {n_synth} synthetic drives (no data download required)...")
        return make_dataset(n_trips=n_synth, seed=1)
    if source == "sensorlogger":
        from proto.loaders import load_sensor_logger_dir

        print(f"Loading Sensor Logger recordings from {path} ...")
        return load_sensor_logger_dir(path)
    if source == "iovnbd":
        from proto.loaders import load_iovnbd_dir

        print(f"Loading IO-VNBD from {path} ...")
        return load_iovnbd_dir(path)
    raise ValueError(source)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="synth", choices=("synth", "sensorlogger", "iovnbd"))
    ap.add_argument("--path", default=None, help="folder for real data sources")
    ap.add_argument("--n-synth", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=22)
    ap.add_argument("--val-fraction", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)
    t_start = time.time()

    trips = load_trips(args.data, args.path, args.n_synth)
    if len(trips) < 3:
        raise SystemExit(f"Need at least 3 trips to split by trip; got {len(trips)}.")

    minutes = sum(len(t) for t in trips) / trips[0].hz / 60
    print(f"  {len(trips)} trips, {minutes:.0f} min of driving\n")

    train_trips, val_trips = split_by_trip(trips, args.val_fraction, seed=args.seed)
    print(f"Split BY TRIP (never by window): {len(train_trips)} train / {len(val_trips)} val")
    print("Training VelNet — IMU in, forward speed out...")
    bundle, hist = train(train_trips, val_trips, epochs=args.epochs)

    y_train_mean = np.concatenate([t.v_true for t in train_trips]).mean()
    y_val = np.concatenate([t.v_true for t in val_trips])
    naive_rmse = float(np.sqrt(((y_val - y_train_mean) ** 2).mean()))
    print(f"\n  predict-the-mean baseline : {naive_rmse:5.2f} m/s")
    print(f"  VelNet                    : {hist['best_val_rmse']:5.2f} m/s")
    print(f"  (CarSpeedNet on real data : 1.80 m/s RMSE — published state of the art)")

    print("\nSimulating GNSS blackouts on held-out trips...")
    summary, rows = benchmark(val_trips, bundle)
    print("\nFinal position error as % of distance travelled")
    print(format_summary(summary))

    ctx = prepare(val_trips[0], bundle)
    paths = [
        trajectory_figure(pick_representative(rows, 60), ARTIFACTS / "trajectory.png"),
        drift_figure(summary, ARTIFACTS / "drift.png"),
        speed_figure(val_trips[0], ctx, ARTIFACTS / "speed.png"),
    ]
    with open(ARTIFACTS / "run.pkl", "wb") as fh:
        pickle.dump({"summary": summary, "hist": hist, "source": args.data}, fh)

    print("\nFigures written:")
    for p in paths:
        print(f"  {p}")

    best = min(summary.values(), key=lambda s: s["median_fused"])
    print(f"\nBest median drift: {best['median_fused']:.1f}%   (spec is <10%)")
    if best["median_fused"] >= 10.0:
        print(
            "  Above spec, as expected for IMU-only dead reckoning — see README:\n"
            "  map matching is what closes this gap, not a better speed model."
        )
    print(f"\nDone in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
