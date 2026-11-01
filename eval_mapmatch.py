"""Does map matching actually close the gap? Measured, not assumed.

Reuses the model trained by run_iovnbd.py, so this only evaluates — no retraining.
Adds a fourth method to the comparison: dead reckoning whose heading comes from the
road network rather than from an integrating gyro.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from proto.dr import dead_reckon_mapmatched, fuse_speed, integrate, prepare
from proto.osm import network_for_trip

ARTIFACTS = Path("artifacts")
DURATIONS = (30, 60, 120, 180)
STRIDE_S = 20.0


def main() -> None:
    with open(ARTIFACTS / "full_run.pkl", "rb") as fh:
        state = pickle.load(fh)
    bundle, val = state["bundle"], state["val"]
    print(f"Loaded model + {len(val)} held-out trips", flush=True)

    rows = []
    for trip in val:
        try:
            net = network_for_trip(trip)
        except Exception as exc:  # network fetch or empty area
            print(f"  {trip.name}: no road network ({type(exc).__name__}) - skipped", flush=True)
            continue
        ctx = prepare(trip, bundle)
        print(f"  {trip.name}: {net.n_segments} road segments", flush=True)

        east, north, heading = ctx["east"], ctx["north"], ctx["heading"]
        for dur in DURATIONS:
            span = int(dur * trip.hz)
            step = max(1, int(STRIDE_S * trip.hz))
            for i0 in range(0, len(trip) - span, step):
                i1 = i0 + span
                sl = slice(i0, i1)
                dist = trip.distance_travelled(i0, i1)
                if dist < 50:
                    continue

                v_fused, _ = fuse_speed(
                    ctx["a_fwd"][sl], ctx["v_hat"][sl], ctx["sigma"][sl],
                    ctx["stationary"][sl], trip.dt, v0=trip.v_true[i0],
                )
                p0, psi0 = (east[i0], north[i0]), heading[i0]

                plain = integrate(v_fused, ctx["yaw_rate"][sl], trip.dt, p0, psi0)
                mapped = dead_reckon_mapmatched(
                    v_fused, ctx["yaw_rate"][sl], trip.dt, p0, psi0, net
                )

                te, tn = east[i1 - 1], north[i1 - 1]
                rows.append({
                    "trip": trip.name, "duration_s": dur, "distance_m": dist,
                    "drift_fused_pct": 100 * np.hypot(plain.east[-1] - te, plain.north[-1] - tn) / dist,
                    "drift_mapped_pct": 100 * np.hypot(mapped.east[-1] - te, mapped.north[-1] - tn) / dist,
                })

    if not rows:
        print("No windows evaluated.")
        return

    print("\nMAP MATCHING vs PLAIN DEAD RECKONING  (median drift, % of distance)")
    print("  outage    n  |  no map   with map  |  change |  pass<10%")
    print("  " + "-" * 62)
    for dur in DURATIONS:
        sel = [r for r in rows if r["duration_s"] == dur]
        if not sel:
            continue
        a = np.array([r["drift_fused_pct"] for r in sel])
        b = np.array([r["drift_mapped_pct"] for r in sel])
        print(f"  {dur:4d}s {len(sel):5d}  |  {np.median(a):6.2f}%  {np.median(b):7.2f}%  |"
              f"  {100*(np.median(a)-np.median(b))/max(np.median(a),1e-9):+6.0f}% |"
              f"  {100*(b<10).mean():5.1f}%")

    with open(ARTIFACTS / "mapmatch_eval.pkl", "wb") as fh:
        pickle.dump(rows, fh)
    print(f"\nSaved artifacts/mapmatch_eval.pkl ({len(rows)} windows)")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Done in {time.time() - t0:.0f}s")
