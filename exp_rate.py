"""Does a higher sample rate actually help? The controlled test.

comma2k19 is the only public dataset shaped for this: 110 Hz accelerometer with
dense CAN speed labels on the SAME drives, so the rate can be varied with everything
else held fixed. IO-VNBD cannot answer it — it is 10 Hz and nothing else.

Fairness matters here, so both arms get:
  * the same drives and the same train/val split (BY TRIP)
  * the same window DURATION in seconds — 10 s either way, which means more
    samples per window at higher rates, not a shorter view of the road
  * the same stride in SECONDS, so each arm trains on the same number of windows

If higher rates win, collecting our own data above 10 Hz is justified. If they do
not, that is weeks of driving saved.
"""
import pickle
import time

import numpy as np

from proto.comma import load_comma2k19
from proto.velnet import split_by_trip, train

WINDOW_SECONDS = 10.0
STRIDE_SECONDS = 0.5
EPOCHS = 14
RATES = (10.0, 25.0, 50.0)

results = {}
print(f"{'rate':>6} {'window':>8} {'trips':>6} {'windows':>9} {'val RMSE':>9} {'mins':>6}", flush=True)
print("-" * 52, flush=True)

for hz in RATES:
    trips = load_comma2k19(hz=hz)
    tr, va = split_by_trip(trips, val_fraction=0.3, seed=3)
    window = int(WINDOW_SECONDS * hz)
    stride = max(1, int(STRIDE_SECONDS * hz))
    n_win = sum(len(t) for t in tr) // stride

    t0 = time.time()
    bundle, hist = train(
        tr, va, epochs=EPOCHS, stride=stride, window=window, width=48, verbose=False
    )
    rmse = hist["best_val_rmse"]
    results[hz] = {"val": rmse, "window": window, "n_windows": n_win, "trips": len(trips)}
    print(
        f"{hz:5.0f}H {window:8d} {len(trips):6d} {n_win:9d} {rmse:9.3f} "
        f"{(time.time()-t0)/60:6.1f}",
        flush=True,
    )

pickle.dump(results, open("artifacts/rate.pkl", "wb"))
base = results[RATES[0]]["val"]
print("-" * 52, flush=True)
for hz in RATES:
    delta = 100 * (results[hz]["val"] - base) / base
    print(f"  {hz:5.0f} Hz: {results[hz]['val']:.3f} m/s  ({delta:+.1f}% vs 10 Hz)", flush=True)

best = min(results, key=lambda h: results[h]["val"])
print(f"\nbest: {best:.0f} Hz", flush=True)
if best == RATES[0]:
    print("Higher rates did NOT help on this data — logging above 10 Hz is not", flush=True)
    print("justified by this evidence alone.", flush=True)
else:
    print(f"Higher rate wins by {100*(base-results[best]['val'])/base:.1f}% — collecting", flush=True)
    print("our own data above 10 Hz is justified.", flush=True)
