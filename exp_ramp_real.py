"""Does distrusting VelNet early and trusting it late recover the spec?

    .venv/Scripts/python.exe exp_ramp_real.py

WHERE THIS CAME FROM (all measured on the 5 real realme rides, 37.1 km)

  exp_qv_real.py step 1: process noise q_v is irrelevant. Sweeping it over 300x
    moves drift under 1.5 points. The filter ignores q_v because the observation
    noise R dominates the gain.

  exp_qv_real.py step 2: inflating VelNet's own sigma by ~100x cuts 10 s drift
    from 45.6% to 27.5% and quadruples the pass rate, 6% -> 24%. But at 60 s the
    same inflation HURTS, 52.5% -> 54.7%.

That pair is a crossover: integration is better early, the model is better late,
and the filter cannot find this itself because VelNet reports ~100x too much
confidence on vehicles it was not trained on.

So: scale sigma by k early and decay to 1 with time constant tau.

    sigma_eff(t) = sigma * max(1, k * exp(-t / tau))

!! METHODOLOGICAL WARNING !!
k and tau are chosen by looking at these same 5 rides. That is fitting on the
evaluation set, and the numbers below are therefore an UPPER BOUND on what this
would deliver on unseen data - they show the headroom, not a validated result.
Report them that way or not at all. A held-out ride is needed to confirm.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from proto.dr import benchmark
from proto.ownlog import load_own

RIDES = Path("data/collected/realme-RMX3371")
DURATIONS = (10, 30, 60)

# (k, tau_seconds). k from the sigma sweep; tau spans the measured crossover.
RAMPS = [(100.0, 10.0), (100.0, 20.0), (100.0, 40.0), (300.0, 20.0), (30.0, 20.0)]


def load() -> list:
    trips = []
    for f in sorted(RIDES.glob("*.csv")):
        try:
            t = load_own(str(f))
        except Exception:
            continue
        if len(t) < 3000 or float(np.max(t.v_true)) < 5.0:
            continue
        t.name = f.stem[-6:]
        trips.append(t)
    return trips


def row(label: str, summary: dict) -> None:
    cells = []
    for d in DURATIONS:
        e = summary[d]
        cells.append(f"{e['median_fused']:6.1f} {100*e['pass_fused']:4.0f}%")
    print(f"{label:22s} | " + " | ".join(cells))


def main() -> None:
    bundle = pickle.load(open("artifacts/full_run.pkl", "rb"))["bundle"]
    trips = load()
    print(f"\n{len(trips)} rides: {', '.join(t.name for t in trips)}\n")

    print(f"{'configuration':22s} | " + " | ".join(f"{d}s drift  pass" for d in DURATIONS))
    print("-" * 74)

    base, _ = benchmark(trips, bundle, durations_s=DURATIONS)
    row("current (sigma as-is)", base)

    flat, _ = benchmark(trips, bundle, durations_s=DURATIONS, sigma_scale=100.0)
    row("flat sigma x100", flat)

    best_label, best_sum, best_score = None, None, 1e9
    for k, tau in RAMPS:
        s, _ = benchmark(trips, bundle, durations_s=DURATIONS, sigma_ramp=(k, tau))
        label = f"ramp k={k:.0f} tau={tau:.0f}s"
        row(label, s)
        score = float(np.mean([s[d]["median_fused"] for d in DURATIONS]))
        if score < best_score:
            best_label, best_sum, best_score = label, s, score

    print("\nstrapdown (no model at all):")
    row("  strapdown", {d: {"median_fused": base[d]["median_naive"],
                            "pass_fused": base[d]["pass_naive"]} for d in DURATIONS})

    print(f"\nBEST: {best_label}")
    for d in DURATIONS:
        b, c = best_sum[d], base[d]
        print(f"  {d:3d}s  {c['median_fused']:5.1f}% -> {b['median_fused']:5.1f}%   "
              f"pass {100*c['pass_fused']:.0f}% -> {100*b['pass_fused']:.0f}%   n={b['n']}")

    print("\n  REMINDER: k and tau were chosen on these same rides. Upper bound,")
    print("  not a validated result. Confirm on a held-out ride before quoting.")


if __name__ == "__main__":
    main()
