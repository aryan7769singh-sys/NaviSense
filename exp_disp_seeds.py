"""Is the 4-5% drift improvement from the displacement loss real, or noise?

One run of each arm showed weight=2 beating the control by 4-5% at 60-120s. But
the mechanism it was supposed to work through — reducing speed BIAS — did not move
(1.493 -> 1.507). An effect with no mechanism, on a metric whose 90th percentile
exceeds 100%, is exactly the shape of noise.

So: run both arms across several seeds and compare the spread of the control
against the spread of the treatment. If the distributions overlap, it is noise.
"""
import pickle
import time

import numpy as np

from exp_displacement import train_displacement
from proto.dr import benchmark, prepare
from proto.velnet import split_by_trip

SEEDS = (0, 1, 2)
trips = pickle.load(open("artifacts/trips_cache.pkl", "rb"))
tr, va = split_by_trip(trips, val_fraction=0.3, seed=3)

out = {}
for w in (0.0, 2.0):
    rows = []
    for seed in SEEDS:
        t0 = time.time()
        bundle, _ = train_displacement(tr, va, disp_weight=w, seed=seed)
        biases = []
        for t in va:
            biases.append(float((prepare(t, bundle)["v_hat"] - t.v_true).mean()))
        summary, _ = benchmark(va, bundle, durations_s=(60, 120))
        rows.append({
            "bias": float(np.mean(np.abs(biases))),
            60: summary[60]["median_fused"],
            120: summary[120]["median_fused"],
        })
        print(f"  w={w:g} seed={seed}  bias {rows[-1]['bias']:.3f}  "
              f"60s {rows[-1][60]:.1f}%  120s {rows[-1][120]:.1f}%  "
              f"[{(time.time()-t0)/60:.1f}m]", flush=True)
    out[w] = rows

pickle.dump(out, open("artifacts/disp_seeds.pkl", "wb"))
print("\n" + "=" * 60, flush=True)
print(f"{'':>10}{'bias':>18}{'drift 60s':>18}{'drift 120s':>18}", flush=True)
for w, rows in out.items():
    label = "control" if w == 0 else f"weight {w:g}"
    cells = []
    for key in ("bias", 60, 120):
        vals = np.array([r[key] for r in rows])
        cells.append(f"{vals.mean():7.2f} +/-{vals.std():5.2f}")
    print(f"{label:>10}{cells[0]:>18}{cells[1]:>18}{cells[2]:>18}", flush=True)

print("", flush=True)
for key, name in ((60, "drift 60s"), (120, "drift 120s")):
    a = np.array([r[key] for r in out[0.0]])
    b = np.array([r[key] for r in out[2.0]])
    sep = abs(a.mean() - b.mean()) / max(np.sqrt((a.std() ** 2 + b.std() ** 2) / 2), 1e-9)
    verdict = "REAL (gap exceeds the spread)" if sep > 2 else "NOISE (spreads overlap)"
    print(f"  {name}: control {a.mean():.1f}% vs treated {b.mean():.1f}%  ->  {verdict}", flush=True)
