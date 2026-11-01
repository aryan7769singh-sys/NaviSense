"""Train on DISPLACEMENT error instead of per-sample speed error.

The problem this fixes, measured twice in this project:

  window 4s -> 10s   speed RMSE -14.6%,  drift ~unchanged
  rate  10 -> 50 Hz  speed RMSE  -5.9%,  speed BIAS +36%

Both "improvements" removed random scatter and left (or worsened) systematic
offset. Distance is the integral of speed, so scatter cancels and offset
accumulates. Per-sample MSE cannot tell them apart — it weights both equally.

Fix: add a term that penalises the MEAN error over a contiguous chunk of
predictions. Within a chunk, random error averages toward zero and contributes
almost no gradient, while a persistent offset survives the average and is punished
in full. That is displacement error, divided by chunk duration to keep the units
in m/s.

Batching therefore has to change too: chunks must be CONTIGUOUS IN TIME from a
single trip, not the usual shuffled independent windows.
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from proto.velnet import VelNet, Bundle, build_dataset, split_by_trip, CHANNELS


def chunk_starts(ids: np.ndarray, chunk: int) -> np.ndarray:
    """Indices where `chunk` consecutive rows all belong to the same trip."""
    n = len(ids)
    if n <= chunk:
        return np.empty(0, dtype=int)
    starts = np.arange(n - chunk)
    return starts[ids[starts] == ids[starts + chunk - 1]]


def train_displacement(train_trips, val_trips, epochs=14, chunks_per_batch=24,
                       chunk=30, lr=2e-3, seed=0, stride=5, window=100,
                       width=48, disp_weight=0.0, verbose=False):
    """Same architecture, loss augmented with a chunk-mean (bias) term."""
    torch.manual_seed(seed)
    xtr, ytr, itr = build_dataset(train_trips, window=window, stride=stride)
    xva, yva, _ = build_dataset(val_trips, window=window, stride=stride)

    mean = xtr.reshape(-1, CHANNELS).mean(0)
    std = xtr.reshape(-1, CHANNELS).std(0) + 1e-6
    xt = torch.tensor(((xtr - mean) / std).transpose(0, 2, 1), dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    xv = torch.tensor(((xva - mean) / std).transpose(0, 2, 1), dtype=torch.float32)
    yv = torch.tensor(yva, dtype=torch.float32)

    all_starts = chunk_starts(itr, chunk)
    # Chunks overlap by chunk-1, so using every start would push each window through
    # the network `chunk` times per epoch — 30x the intended compute. Sample a
    # non-overlapping-sized subset each epoch instead: same cost as normal training,
    # but a different set of chunk boundaries every epoch.
    per_epoch = max(1, len(all_starts) // chunk)
    offsets = torch.arange(chunk)
    model = VelNet(window=window, width=width)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    t_start = time.time()
    best, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        starts = all_starts[rng.permutation(len(all_starts))[:per_epoch]]
        for i in range(0, len(starts), chunks_per_batch):
            sel = starts[i : i + chunks_per_batch]
            if len(sel) == 0:
                continue
            idx = (torch.tensor(sel)[:, None] + offsets).reshape(-1)
            mu, _ = model(xt[idx])
            y = yt[idx]
            per_sample = ((y - mu) ** 2).mean()

            # Mean signed error within each contiguous chunk == displacement error
            # over the chunk, divided by its duration. Scatter cancels here; a
            # systematic offset does not.
            bias = (mu - y).view(len(sel), chunk).mean(dim=1)
            loss = per_sample + disp_weight * (bias ** 2).mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            mv, _ = model(xv)
            rmse = float(torch.sqrt(((mv - yv) ** 2).mean()))
        if rmse < best:
            best, best_state = rmse, {k: v.clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"    epoch {epoch:3d}  val RMSE {rmse:.3f}  [{time.time()-t_start:.0f}s]", flush=True)

    if best_state:
        model.load_state_dict(best_state)
    return Bundle(model, mean, std, window), best


if __name__ == "__main__":
    from proto.dr import benchmark, format_summary, prepare
    import pickle

    source = sys.argv[1] if len(sys.argv) > 1 else "iovnbd"
    if source == "comma":
        from proto.comma import load_comma2k19
        trips = load_comma2k19(hz=10.0)
    else:
        trips = pickle.load(open("artifacts/trips_cache.pkl", "rb"))

    tr, va = split_by_trip(trips, val_fraction=0.3, seed=3)
    print(f"{source}: {len(tr)} train / {len(va)} val trips (split BY TRIP)\n", flush=True)

    results = {}
    for w in (0.0, 2.0, 10.0):
        label = "off (per-sample MSE only)" if w == 0 else f"displacement weight {w:g}"
        print(f"--- {label} ---", flush=True)
        t0 = time.time()
        bundle, rmse = train_displacement(tr, va, disp_weight=w, verbose=True)

        biases, rmses = [], []
        for t in va:
            ctx = prepare(t, bundle)
            err = ctx["v_hat"] - t.v_true
            biases.append(float(err.mean()))
            rmses.append(float(np.sqrt((err ** 2).mean())))
        summary, _ = benchmark(va, bundle, durations_s=(30, 60, 120))
        drift = {d: summary[d]["median_fused"] for d in summary}
        results[w] = {"rmse": np.mean(rmses), "bias": np.mean(np.abs(biases)), "drift": drift}
        print(f"  RMSE {np.mean(rmses):.3f}   BIAS {np.mean(np.abs(biases)):.3f}   "
              f"drift {' '.join(f'{d}s={v:.1f}%' for d, v in drift.items())}   "
              f"[{(time.time()-t0)/60:.1f} min]\n", flush=True)

    pickle.dump(results, open(f"artifacts/displacement_{source}.pkl", "wb"))
    base = results[0.0]
    print("=" * 64, flush=True)
    print(f"{'weight':>8}{'RMSE':>9}{'BIAS':>9}   drift vs baseline", flush=True)
    print("-" * 64, flush=True)
    for w, r in results.items():
        dd = " ".join(f"{d}s {100*(r['drift'][d]-base['drift'][d])/base['drift'][d]:+.0f}%"
                      for d in sorted(r["drift"]))
        print(f"{w:8g}{r['rmse']:9.3f}{r['bias']:9.3f}   {dd}", flush=True)
