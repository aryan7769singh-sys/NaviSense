"""Export a real drive + real engine output as JSON for the interactive demo.

Nothing here is staged. The trajectories come from the trained model running over a
held-out IO-VNBD recording, with GNSS genuinely withheld from the moment of the
blackout. The road network is real OpenStreetMap geometry for the same area.

Several blackout START POINTS are exported so the demo's "KILL GPS" button can
genuinely begin a blackout wherever playback has reached, rather than replaying one
pre-baked moment and pretending the button did something.

Only one DURATION is exported per start point, deliberately. Dead reckoning is
causal — the estimate at t+30s never depends on anything after it — so the drift
30 seconds into a 90-second blackout *is* the 30-second drift. The demo's duration
selector just stops the animation earlier. Exporting separate 30/60/90 windows would
duplicate identical numbers.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from proto.dr import fuse_speed, integrate, prepare
from proto.osm import network_for_trip

ARTIFACTS = Path("artifacts")
BLACKOUT_S = 90.0        # longest offered; shorter durations are prefixes of this
LEAD_IN_S = 20.0         # GNSS-good driving shown before the first blackout can start
N_STARTS = 6             # how many places the judge can kill GNSS
SPACING_S = 45.0         # gap between consecutive start points


def run_blackout(trip, ctx, i0: int, n: int) -> dict:
    """Run both methods blind from sample i0 for n samples."""
    sl = slice(i0, i0 + n)
    v_fused, _ = fuse_speed(
        ctx["a_fwd"][sl], ctx["v_hat"][sl], ctx["sigma"][sl], ctx["stationary"][sl],
        trip.dt, v0=trip.v_true[i0],
    )
    p0 = (ctx["east"][i0], ctx["north"][i0])
    psi0 = ctx["heading"][i0]

    ours = integrate(v_fused, ctx["yaw_rate"][sl], trip.dt, p0, psi0)
    naive_v = trip.v_true[i0] + np.cumsum(ctx["a_fwd"][sl]) * trip.dt
    naive = integrate(np.clip(naive_v, 0, None), ctx["yaw_rate"][sl], trip.dt, p0, psi0)

    truth_e, truth_n = ctx["east"][sl], ctx["north"][sl]
    dist = np.cumsum(trip.v_true[sl]) * trip.dt
    return {
        "startFrame": int(i0),
        "ours": [[round(float(e), 1), round(float(n_), 1)] for e, n_ in zip(ours.east, ours.north)],
        "naive": [[round(float(e), 1), round(float(n_), 1)] for e, n_ in zip(naive.east, naive.north)],
        "speedEst": [round(float(v * 3.6), 1) for v in v_fused],
        "distance": [round(float(d), 1) for d in dist],
        "driftOurs": [round(float(d), 1) for d in
                      np.hypot(ours.east - truth_e, ours.north - truth_n)],
        "driftNaive": [round(float(d), 1) for d in
                       np.hypot(naive.east - truth_e, naive.north - truth_n)],
    }


def candidate_starts(trip, ctx, n: int, lead_n: int) -> list[int]:
    """Start points where the vehicle is genuinely driving for the whole window.

    Filters, not selection: every candidate that passes is kept, and the ones used
    are spread evenly across them. Nothing is ranked by how flattering it looks —
    picking the best-drift window is exactly the thing that would make this demo
    worthless the moment a judge asked how it was chosen.
    """
    out = []
    step = int(5 * trip.hz)
    for i0 in range(lead_n, len(trip) - n, step):
        seg = trip.v_true[i0 : i0 + n]
        if seg.mean() < 5.0 or (seg < 0.5).mean() > 0.4:
            continue
        # The vehicle must be genuinely driving at EVERY duration the demo offers,
        # not just averaged over the full window. Without this a window that starts
        # at a red light exports 0 m travelled at 30 s, and drift-as-a-percentage
        # divides by zero.
        if any(trip.distance_travelled(i0, i0 + int(d * trip.hz)) < 60
               for d in (10, 30, 60)):
            continue
        if trip.distance_travelled(i0, i0 + n) < 300:
            continue
        out.append(i0)
    return out


def main() -> None:
    with open(ARTIFACTS / "full_run.pkl", "rb") as fh:
        state = pickle.load(fh)
    bundle, val = state["bundle"], state["val"]

    hz = val[0].hz
    n = int(BLACKOUT_S * hz)
    lead_n = int(LEAD_IN_S * hz)
    spacing = int(SPACING_S * hz)

    # Per-trip performance varies enormously (measured: median drift at 30 s ranges
    # from 21% to 103% across held-out trips). Taking whichever trip happens to be
    # first would show whatever that trip happens to be. Instead, score every usable
    # trip and take the one whose median is CLOSEST TO THE OVERALL MEDIAN — the most
    # representative drive available, chosen without reference to which looks best.
    usable = []
    for trip in val:
        ctx = prepare(trip, bundle)
        cands = candidate_starts(trip, ctx, n, lead_n)
        if len(cands) < N_STARTS:
            continue
        drifts = []
        for i0 in cands:
            b = run_blackout(trip, ctx, i0, n)
            drifts.append(100 * b["driftOurs"][-1] / max(b["distance"][-1], 1e-6))
        usable.append((trip, ctx, cands, float(np.median(drifts))))
    if not usable:
        raise SystemExit("no trip with enough usable blackout windows")

    overall = float(np.median([u[3] for u in usable]))
    trip, ctx, cands, trip_median = min(usable, key=lambda u: abs(u[3] - overall))
    print(f"trips considered: {len(usable)}, overall median drift {overall:.1f}%")
    print(f"selected {trip.name} (median {trip_median:.1f}%) as most representative")
    net = network_for_trip(trip)

    # Spread the start points across the drive rather than clustering them.
    starts: list[int] = []
    for i0 in cands:
        if not starts or i0 - starts[-1] >= spacing:
            starts.append(i0)
        if len(starts) == N_STARTS:
            break

    clip_start = max(0, starts[0] - lead_n)
    clip_end = min(len(trip), starts[-1] + n)
    print(f"{trip.name}: clip {(clip_end-clip_start)/hz:.0f}s, "
          f"{len(starts)} blackout start points, {BLACKOUT_S:.0f}s each")

    blackouts = []
    for i0 in starts:
        b = run_blackout(trip, ctx, i0, n)
        b["startFrame"] = int(i0 - clip_start)  # relative to the exported clip
        d = b["distance"]
        b["pct"] = {}
        for secs in (10, 30, 60, 90):
            k = min(int(secs * hz) - 1, len(d) - 1)
            b["pct"][str(secs)] = [
                round(100 * b["driftOurs"][k] / max(d[k], 1e-6), 1),
                round(100 * b["driftNaive"][k] / max(d[k], 1e-6), 1),
            ]
        blackouts.append(b)
        o, nv = b["pct"]["30"]
        print(f"  t={i0/hz:6.0f}s  @30s: ours {o:5.1f}%   naive {nv:5.1f}%"
              f"   {'ours wins' if o < nv else 'BASELINE WINS'}")

    # Which window is typical? Default to the one closest to this trip's median, so
    # the first thing a judge sees is representative rather than whichever start
    # point happened to come first.
    at30 = [b["pct"]["30"][0] for b in blackouts]
    median30 = float(np.median(at30))
    default_index = int(np.argmin([abs(v - median30) for v in at30]))
    print(f"  default window: #{default_index} ({at30[default_index]:.1f}% @30s, "
          f"trip median {median30:.1f}%)")

    # A single trajectory is a poor summary of a system this variable — these six
    # windows alone span 21% to 101%. Ship the aggregate alongside the animation so
    # the demo can state what TYPICAL is, over every held-out window, not just the
    # one on screen.
    print("  computing aggregate over all held-out windows...", flush=True)
    from proto.dr import benchmark
    agg_summary, agg_rows = benchmark(val, bundle, durations_s=(10, 30, 60, 90))
    stats = {
        str(k): {
            "n": agg_summary[k]["n"],
            "medianOurs": round(agg_summary[k]["median_fused"], 1),
            "medianNaive": round(agg_summary[k]["median_naive"], 1),
            "p90Ours": round(agg_summary[k]["p90_fused"], 1),
            "medianDistM": round(agg_summary[k]["median_dist_m"]),
        }
        for k in sorted(agg_summary)
    }
    for k, v in stats.items():
        print(f"    {k:>3}s  n={v['n']:4d}  ours {v['medianOurs']:5.1f}%  "
              f"naive {v['medianNaive']:6.1f}%  (p90 ours {v['p90Ours']:.0f}%)")
    # Every window's 30 s drift, for the distribution strip on the page.
    dist30 = sorted(round(100 * r["err_fused_m"] / max(r["distance_m"], 1e-6), 1)
                    for r in agg_rows if r["duration_s"] == 30)

    te = ctx["east"][clip_start:clip_end]
    tn = ctx["north"][clip_start:clip_end]

    margin = 400.0
    mid = 0.5 * (net.a + net.b)
    keep = ((mid[:, 0] > te.min() - margin) & (mid[:, 0] < te.max() + margin)
            & (mid[:, 1] > tn.min() - margin) & (mid[:, 1] < tn.max() + margin))
    roads = np.concatenate([net.a[keep], net.b[keep]], axis=1)

    out = {
        "trip": trip.name,
        "hz": hz,
        "blackoutSamples": n,
        "truth": [[round(float(e), 1), round(float(x), 1)] for e, x in zip(te, tn)],
        "speedTrue": [round(float(v * 3.6), 1) for v in trip.v_true[clip_start:clip_end]],
        "roads": [[round(float(x), 1) for x in seg] for seg in roads],
        "blackouts": blackouts,
        "defaultIndex": default_index,
        "stats": stats,
        "dist30": dist30,
    }

    path = ARTIFACTS / "demo_data.json"
    path.write_text(json.dumps(out), encoding="utf-8")
    print(f"  {len(roads)} road segments, {len(out['truth'])} clip frames")
    print(f"  wrote {path} ({path.stat().st_size/1e6:.1f} MB)")

    # Inline the data into the page so the demo is a single self-contained file —
    # no fetch(), so it works from file:// with the venue wifi down.
    tmpl = Path("demo_template.html").read_text(encoding="utf-8")
    if "__DATA__" not in tmpl:
        raise SystemExit("demo_template.html has no __DATA__ placeholder")
    page = ARTIFACTS / "demo.html"
    page.write_text(tmpl.replace("__DATA__", json.dumps(out)), encoding="utf-8")
    print(f"  wrote {page} ({page.stat().st_size/1e6:.1f} MB, self-contained)")


if __name__ == "__main__":
    main()
