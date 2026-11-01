"""Does the higher sample rate reduce DRIFT, or only speed RMSE?

The rate sweep showed 10 Hz -> 50 Hz cuts speed RMSE by 14.0%. That is not yet a
reason to celebrate. This project has already been caught by exactly this: widening
the window from 4 s to 10 s cut RMSE by 14.6% and moved drift by nothing, because it
removed random scatter rather than systematic offset.

Distance is the integral of speed, so zero-mean error largely cancels while a
persistent offset accumulates linearly. RMSE mixes the two together and cannot tell
you which one you improved.

So this measures all three, per rate:
  speed RMSE   what the sweep reported
  speed BIAS   mean signed error per trip — the part that integrates into distance
  DRIFT        end-of-blackout position error, % of distance — the graded metric
"""
import pickle
import time

import numpy as np

from proto.comma import load_comma2k19
from proto.dr import benchmark, format_summary, prepare
from proto.velnet import split_by_trip, train

WINDOW_SECONDS = 10.0
STRIDE_SECONDS = 0.5
EPOCHS = 14
RATES = (10.0, 50.0)

out = {}
for hz in RATES:
    print(f"\n{'='*62}\n{hz:.0f} Hz\n{'='*62}", flush=True)
    trips = load_comma2k19(hz=hz)
    tr, va = split_by_trip(trips, val_fraction=0.3, seed=3)
    window = int(WINDOW_SECONDS * hz)
    stride = max(1, int(STRIDE_SECONDS * hz))

    t0 = time.time()
    bundle, hist = train(tr, va, epochs=EPOCHS, stride=stride, window=window,
                         width=48, verbose=False)
    print(f"  trained in {(time.time()-t0)/60:.1f} min   val RMSE {hist['best_val_rmse']:.3f} m/s",
          flush=True)

    # Per-trip signed bias — the component that survives integration.
    biases, rmses = [], []
    for t in va:
        ctx = prepare(t, bundle)
        err = ctx["v_hat"] - t.v_true
        biases.append(float(err.mean()))
        rmses.append(float(np.sqrt((err**2).mean())))
    abs_bias = float(np.mean(np.abs(biases)))
    print(f"  speed RMSE  {np.mean(rmses):.3f} m/s", flush=True)
    print(f"  speed BIAS  {abs_bias:.3f} m/s   <- the part that integrates", flush=True)

    summary, rows = benchmark(va, bundle, durations_s=(10, 30, 60, 120))
    print("\n" + format_summary(summary), flush=True)

    out[hz] = {
        "val_rmse": hist["best_val_rmse"],
        "rmse": float(np.mean(rmses)),
        "bias": abs_bias,
        "summary": summary,
    }

pickle.dump(out, open("artifacts/rate_drift.pkl", "wb"))

lo, hi = RATES[0], RATES[-1]
print(f"\n{'='*62}\nVERDICT: {lo:.0f} Hz vs {hi:.0f} Hz\n{'='*62}", flush=True)
def pct(a, b):
    return 100 * (b - a) / a if a else float("nan")
print(f"  speed RMSE  {out[lo]['rmse']:6.3f} -> {out[hi]['rmse']:6.3f}  "
      f"({pct(out[lo]['rmse'], out[hi]['rmse']):+.1f}%)", flush=True)
print(f"  speed BIAS  {out[lo]['bias']:6.3f} -> {out[hi]['bias']:6.3f}  "
      f"({pct(out[lo]['bias'], out[hi]['bias']):+.1f}%)", flush=True)
for d in sorted(out[lo]["summary"]):
    if d in out[hi]["summary"]:
        a = out[lo]["summary"][d]["median_fused"]
        b = out[hi]["summary"][d]["median_fused"]
        print(f"  drift @{d:4d}s {a:6.1f}% -> {b:6.1f}%  ({pct(a, b):+.1f}%)", flush=True)
print("\nIf BIAS and DRIFT barely move while RMSE improves, the higher rate removed", flush=True)
print("random scatter only — and logging above 10 Hz is NOT justified by this.", flush=True)
