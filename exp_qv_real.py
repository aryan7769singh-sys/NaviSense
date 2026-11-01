"""Is the filter's process noise mistuned for real 400 Hz data?

    .venv/Scripts/python.exe exp_qv_real.py

WHY THIS EXPERIMENT EXISTS

On the first real vehicle data (realme RMX3371, 5 rides, 37.1 km) plain strapdown
integration beat our fused filter at 10 s on every road ride - 9.8-16.7% against
26-55%. That should not happen if the filter is weighting its inputs correctly.

q_v is the process noise on the speed state. It says "how much do I distrust my own
integration between updates". Set it too high and the filter's covariance grows fast,
the Kalman gain on VelNet's observation goes up, and the filter throws away good
short-term integration in favour of a model we now know is biased by -11 to +4 m/s on
real vehicles.

So this sweeps q_v. If a lower value recovers the strapdown-early behaviour WITHOUT
losing the long-blackout advantage, the fix is a constant, not an architecture.

Trained bundle comes from artifacts/full_run.pkl - the IO-VNBD model, unchanged. This
tests the FILTER, not the network.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from proto.dr import benchmark
from proto.ownlog import load_own

RIDES = Path("data/collected/realme-RMX3371")
Q_VALUES = (0.6,)
# How much to inflate VelNet's own sigma. 1 = trust it as it reports;
# large = ignore it and fall back to integration.
SIGMA_SCALES = (1.0, 3.0, 10.0, 30.0, 100.0, 1000.0)
DURATIONS = (10, 30, 60)


def main() -> None:
    bundle = pickle.load(open("artifacts/full_run.pkl", "rb"))["bundle"]

    trips = []
    for f in sorted(RIDES.glob("*.csv")):
        try:
            t = load_own(str(f))
        except Exception as e:
            print(f"  skip {f.name}: {e}")
            continue
        # the 1.9-min bench capture has no motion; triage already rejected it
        if len(t) < 3000 or float(np.max(t.v_true)) < 5.0:
            print(f"  skip {f.name}: too short or never moved")
            continue
        t.name = f.stem[-6:]
        trips.append(t)
    print(f"\n{len(trips)} rides loaded: {', '.join(t.name for t in trips)}\n")

    print(f"{'sigma x':>8s} | " + " | ".join(
        f"{d}s fused  pass" for d in DURATIONS) + " | vs strapdown")
    print("-" * 76)

    results = {}
    for sc in SIGMA_SCALES:
        summary, _ = benchmark(trips, bundle, durations_s=DURATIONS,
                               q_v=0.6, sigma_scale=sc)
        cells = []
        for d in DURATIONS:
            e = summary[d]
            cells.append(f"{e['median_fused']:6.1f} {100*e['pass_fused']:4.0f}%")
        strap = " ".join(f"{summary[d]['median_naive']:5.1f}" for d in DURATIONS)
        print(f"{sc:8.0f} | " + " | ".join(cells) + f" | {strap}")
        results[sc] = summary

    # the strapdown column does not depend on q_v; print it once as the target
    print("\nstrapdown is unaffected by q_v - it is the same column each row, and is")
    print("the number the filter has to beat to be earning its place.\n")

    best = {}
    for d in DURATIONS:
        s_best = min(SIGMA_SCALES, key=lambda s: results[s][d]["median_fused"])
        b, cur = results[s_best][d], results[1.0][d]
        best[d] = (s_best, b["median_fused"], cur["median_fused"])
        print(f"  {d:3d}s  best sigma x{s_best:<6.0f} -> {b['median_fused']:5.1f}%  "
              f"(as-reported {cur['median_fused']:5.1f}%)  "
              f"pass {100*b['pass_fused']:.0f}% vs {100*cur['pass_fused']:.0f}%  "
              f"n={b['n']}")

    print("\nHONEST READING")
    if all(best[d][1] < best[d][2] - 1.0 for d in DURATIONS):
        print("  Inflating sigma helps at EVERY duration. VelNet is overconfident on")
        print("  real vehicles: the filter cannot down-weight it because it does not")
        print("  know it is wrong. Fix the uncertainty head, not the filter.")
    elif any(best[d][1] < best[d][2] - 1.0 for d in DURATIONS):
        print("  Mixed: inflating sigma helps at some durations only - which is the")
        print("  signature of a crossover. Weight integration early and the model")
        print("  late, varying with elapsed blackout time.")
    else:
        print("  Model uncertainty is not the lever either. Stop tuning and go")
        print("  measure what else differs between these rides and IO-VNBD.")
    print("\n  n is small (5 rides, one device, one day). Directional only.")


if __name__ == "__main__":
    main()
