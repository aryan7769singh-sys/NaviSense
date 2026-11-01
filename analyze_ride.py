"""Analyse a ride recorded by the IDR Logger app.

    python analyze_ride.py data/own/drive_XXXXXXXX_XXXXXX.csv

Answers, in order, the questions that actually matter:

  1. Is the capture usable at all?      rates, gaps, missing streams
  2. Did the vehicle actually move?     GNSS speed distribution
  3. Did alignment converge?            mount-yaw confidence
  4. Does VelNet work on a real vehicle? predicted speed vs GNSS Doppler
  5. How far does it drift blind?       simulated GNSS blackouts

Step 4 is the one nobody has ever answered.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from proto.align import to_vehicle_frame
from proto.dr import benchmark, format_summary
from proto.ownlog import load_own
from proto.velnet import trip_features


def section(title: str) -> None:
    print(f"\n{'=' * 66}\n{title}\n{'=' * 66}")


def main(path: str) -> None:
    import pandas as pd

    section("1. CAPTURE QUALITY")
    df = pd.read_csv(path, comment="#", on_bad_lines="skip")
    dur = df["t"].max() - df["t"].min()
    print(f"  duration: {dur / 60:.1f} min")
    ok = True
    for kind, name, want in (("A", "accel", 100), ("G", "gyro", 100),
                             ("M", "mag", 20), ("P", "GNSS", 0.2)):
        d = df[df["kind"] == kind]
        if len(d) < 10:
            print(f"  {name:6}: MISSING — this ride cannot be used" if kind in "AGP"
                  else f"  {name:6}: missing (not fatal)")
            if kind in "AGP":
                ok = False
            continue
        t = d["t"].to_numpy(float)
        rate = len(d) / max(t.max() - t.min(), 1e-9)
        flag = "" if rate >= want else "   <-- LOW"
        print(f"  {name:6}: {len(d):8,} rows  {rate:7.1f} Hz{flag}")
    if not ok:
        print("\n  Stopping: a required stream is missing.")
        return

    trip = load_own(path)

    section("2. DID THE VEHICLE MOVE?")
    v = trip.v_true
    print(f"  GNSS speed: median {np.median(v):.1f}  max {v.max():.1f} m/s "
          f"({v.max() * 3.6:.0f} km/h)")
    print(f"  moving (>2 m/s): {100 * (v > 2).mean():.0f}% of samples")
    if v.max() < 5:
        print("  WARNING: never exceeded 18 km/h — the model was trained on road speeds.")

    section("3. DID ALIGNMENT CONVERGE?")
    al = to_vehicle_frame(trip, v_ref=trip.v_true)
    conf = al["mount_confidence"]
    print(f"  mount-yaw confidence: {conf:.3f}   (>0.35 = trustworthy)")
    print(f"  mount yaw:            {np.rad2deg(al['mount_yaw']):+.0f} deg")
    print(f"  |horizontal accel|:   {np.median(np.hypot(al['acc'][:, 0], al['acc'][:, 1])):.2f} m/s^2")
    if conf < 0.35:
        print("  -> Alignment did NOT converge. Everything below is unreliable.")

    section("4. DOES VelNet WORK ON A REAL VEHICLE?")
    import pickle
    bundle = pickle.load(open("artifacts/full_run.pkl", "rb"))["bundle"]
    v_hat, _ = bundle.predict(trip_features(trip, v_ref=trip.v_true, window=bundle.window))
    m = v > 2
    if m.sum() < 100:
        print("  too few moving samples to judge")
    else:
        rmse = float(np.sqrt(np.mean((v_hat[m] - v[m]) ** 2)))
        bias = float(np.mean(v_hat[m] - v[m]))
        corr = float(np.corrcoef(v_hat[m], v[m])[0, 1])
        mean_rmse = float(np.sqrt(np.mean((v[m] - v[m].mean()) ** 2)))
        print(f"  predicted: median {np.median(v_hat[m]):.1f} m/s   actual: {np.median(v[m]):.1f} m/s")
        print(f"  RMSE  {rmse:5.2f} m/s      (predict-the-mean would give {mean_rmse:.2f})")
        print(f"  bias  {bias:+5.2f} m/s      (this is what integrates into distance error)")
        print(f"  corr  {corr:+5.2f}")
        verdict = ("WORKS — beats predicting the mean" if rmse < mean_rmse
                   else "DOES NOT WORK on this vehicle — worse than guessing the average")
        print(f"  -> {verdict}")

    section("5. DRIFT THROUGH SIMULATED BLACKOUTS")
    try:
        summary, _ = benchmark([trip], bundle, durations_s=(10, 30, 60))
        print(format_summary(summary))
    except Exception as exc:
        print(f"  could not run: {exc}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        files = sorted(Path("data/own").glob("*.csv"))
        if not files:
            raise SystemExit("no rides in data/own/")
        target = str(max(files, key=lambda p: p.stat().st_mtime))
        print(f"(no file given; using newest: {target})")
    else:
        target = sys.argv[1]
    main(target)
