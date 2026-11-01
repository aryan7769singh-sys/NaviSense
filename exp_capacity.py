"""Does adding parameters help, or just memorise harder?

Trains the same data at four capacities and reports the TRAIN-VAL GAP, which is the
number that answers the question. A model that is too small shows a high train error
too; a model that is overfitting shows train error falling while val error does not.
"""
import pickle, time, numpy as np
from proto.velnet import train, split_by_trip, build_dataset

trips = pickle.load(open("artifacts/trips_cache.pkl", "rb"))
tr, va = split_by_trip(trips, val_fraction=0.3, seed=3)
print(f"Real IO-VNBD: {len(tr)} train / {len(va)} val trips, split BY TRIP", flush=True)
print(f"\n{'width':>6} {'params':>10} {'train':>8} {'val':>8} {'gap':>7} {'mins':>6}", flush=True)
print("-" * 50, flush=True)

res = {}
for w in (24, 48, 96, 192):
    t0 = time.time()
    b, h = train(tr, va, epochs=14, stride=5, width=w, verbose=False)
    xtr, ytr, _ = build_dataset(tr, window=b.window, stride=25)
    mu, _ = b.predict(xtr)
    trr = float(np.sqrt(np.mean((mu - ytr) ** 2)))
    v = h["best_val_rmse"]
    n = sum(p.numel() for p in b.model.parameters())
    res[w] = dict(params=n, train=trr, val=v, gap=v - trr)
    print(f"{w:6d} {n:10,} {trr:8.2f} {v:8.2f} {v-trr:7.2f} {(time.time()-t0)/60:6.1f}", flush=True)

pickle.dump(res, open("artifacts/capacity.pkl", "wb"))
best = min(res, key=lambda w: res[w]["val"])
print("-" * 50, flush=True)
print(f"\nbest val RMSE: width {best} ({res[best]['params']:,} params) -> {res[best]['val']:.2f} m/s", flush=True)
print(f"4x the parameters changes val RMSE by "
      f"{100*(res[192]['val']-res[48]['val'])/res[48]['val']:+.1f}%", flush=True)
