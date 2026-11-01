"""Does a longer input window help? Measured, not assumed.

arXiv:2505.18490 uses 10 s windows and CarSpeedNet ablates 0.25/1/4 s with longer
always winning. Ours has been 4 s. Everything else is held fixed.
"""
import time
import numpy as np
from dataclasses import replace
from proto.loaders import load_iovnbd_dir
from proto.velnet import split_by_trip, train

MAX_MIN = 20.0

def truncate(t, m):
    n = int(m*60*t.hz)
    if len(t) <= n: return t
    return replace(t, t=t.t[:n], acc=t.acc[:n], gyr=t.gyr[:n], lat=t.lat[:n],
                   lon=t.lon[:n], v_true=t.v_true[:n], course=t.course[:n])

trips = [truncate(t, MAX_MIN) for t in load_iovnbd_dir("data")]
trips = [t for t in trips if t.v_true.max() > 5.0]
tr, va = split_by_trip(trips, val_fraction=0.3, seed=3)
ymean = np.concatenate([t.v_true for t in tr]).mean()
yval = np.concatenate([t.v_true for t in va])
print(f"{len(trips)} trips ({len(tr)} train / {len(va)} val)")
print(f"predict-the-mean baseline: {np.sqrt(((yval-ymean)**2).mean()):.2f} m/s\n")

print(f"{'window':>10}{'params':>10}{'val RMSE':>11}{'time':>8}")
print("-"*41)
results = {}
for w in (40, 70, 100, 150):
    t0 = time.time()
    bundle, hist = train(tr, va, epochs=20, stride=5, window=w, verbose=False)
    n = sum(p.numel() for p in bundle.model.parameters())
    results[w] = hist['best_val_rmse']
    print(f"{w/10:8.0f}s{n:10,}{hist['best_val_rmse']:11.3f}{time.time()-t0:7.0f}s", flush=True)

best = min(results, key=results.get)
print(f"\nbest: {best/10:.0f}s window at {results[best]:.3f} m/s "
      f"({100*(results[40]-results[best])/results[40]:+.1f}% vs the 4 s baseline)")
