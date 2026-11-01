"""Triage every captured ride: what is actually in it, before any modelling.

    .venv/Scripts/python.exe triage_rides.py [dir]

Answers only the questions that decide whether a file is worth analysing:
  - are all three streams alive, and at what rate
  - did it move, how far, how fast
  - how much of it lost GNSS (the interesting part for us)
  - how much vibration energy above 5 Hz (the cue VelNet actually reads)

The vibration column is the one that should separate a metro from an auto. Do not
assume which is which from the filename - nothing in the file says.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

R_EARTH = 6371000.0


def haversine_m(lat, lon):
    """Cumulative ground distance in metres from consecutive lat/lon."""
    if len(lat) < 2:
        return 0.0
    la, lo = np.radians(lat), np.radians(lon)
    dla, dlo = np.diff(la), np.diff(lo)
    a = np.sin(dla / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(dlo / 2) ** 2
    return float(np.sum(2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))))


def hf_energy(t, acc):
    """RMS of accelerometer magnitude above ~5 Hz, the band 10 Hz data cannot see.

    Uses a difference filter rather than an FFT so an irregular sample clock does
    not silently invalidate the number.
    """
    if len(acc) < 100:
        return float("nan")
    mag = np.linalg.norm(acc, axis=1)
    return float(np.std(np.diff(mag)))


def triage(path: Path) -> dict | None:
    try:
        df = pd.read_csv(path, comment="#", on_bad_lines="skip",
                         dtype={"kind": str}, low_memory=False)
    except Exception as e:
        return {"file": path.name, "error": str(e)[:40]}
    if "kind" not in df or "t" not in df:
        return {"file": path.name, "error": "no kind/t column"}

    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df = df.dropna(subset=["t"])
    dur = float(df["t"].max() - df["t"].min()) if len(df) else 0.0

    row: dict = {"file": path.name, "min": dur / 60.0}

    for kind, name in (("A", "acc"), ("G", "gyr"), ("M", "mag"), ("P", "gps")):
        d = df[df["kind"] == kind]
        span = float(d["t"].max() - d["t"].min()) if len(d) > 1 else 0.0
        row[name] = len(d) / span if span > 1 else 0.0
        row[name + "_span"] = span

    p = df[df["kind"] == "P"].copy()
    if len(p) > 2:
        for c in ("v1", "v2", "v4", "v6", "v7"):
            p[c] = pd.to_numeric(p[c], errors="coerce")
        p = p.dropna(subset=["v1", "v2"])
        p = p[(p["v1"].abs() > 1e-6) & (p["v1"].abs() <= 90)]
    if len(p) > 2:
        spd = p["v4"].to_numeric_values if False else p["v4"].to_numpy(float)
        spd = spd[np.isfinite(spd)]
        row["km"] = haversine_m(p["v1"].to_numpy(float), p["v2"].to_numpy(float)) / 1000.0
        row["vmax"] = float(np.nanpercentile(spd, 99)) * 3.6 if len(spd) else 0.0
        row["vmed"] = float(np.nanmedian(spd[spd > 0.5])) * 3.6 if (spd > 0.5).any() else 0.0
        row["moving"] = float(np.mean(spd > 1.0)) if len(spd) else 0.0
        # v6 = accuracy metres, v7 = satellites used in fix.
        # v8 is a literal trailing zero, NOT a data field.
        row["acc_m"] = float(np.nanmedian(p["v6"].to_numpy(float)))
        row["sats"] = float(np.nanmedian(p["v7"].to_numpy(float)))
        # longest stretch with no fix: the blackout we actually care about
        tp = np.sort(p["t"].to_numpy(float))
        row["gap"] = float(np.max(np.diff(tp))) if len(tp) > 1 else 0.0
    else:
        row.update(km=0.0, vmax=0.0, vmed=0.0, moving=0.0, acc_m=np.nan,
                   sats=np.nan, gap=dur)

    a = df[df["kind"] == "A"]
    if len(a) > 100:
        cols = a[["v1", "v2", "v3"]].apply(pd.to_numeric, errors="coerce")
        cols = cols.dropna()
        row["hf"] = hf_energy(a["t"].to_numpy(float), cols.to_numpy(float))
    else:
        row["hf"] = float("nan")
    return row


def verdict(r: dict) -> str:
    if r.get("error"):
        return "UNREADABLE"
    if r["acc"] < 50 or r["gyr"] < 50:
        return "DEAD STREAM"
    if r["min"] < 3:
        return "TOO SHORT"
    if r["km"] < 0.3:
        return "DID NOT MOVE"
    # a stream that stops long before the file ends means the OS killed capture
    if r["acc_span"] < 0.7 * r["min"] * 60:
        return "CAPTURE DIED"
    if r["moving"] < 0.3:
        return "MOSTLY PARKED"
    return "USABLE"


def main(d: str) -> None:
    files = sorted(Path(d).glob("*.csv"))
    parts = sorted(Path(d).glob("*.csv.part"))
    if parts:
        print(f"note: {len(parts)} incomplete upload(s) skipped: "
              f"{', '.join(p.name for p in parts)}\n")

    rows = []
    for f in files:
        r = triage(f)
        if r:
            r["verdict"] = verdict(r)
            rows.append(r)
            print(".", end="", flush=True)
    print("\n")

    hdr = (f"{'file':32s} {'min':>5s} {'km':>6s} {'vmed':>5s} {'vmax':>5s} "
           f"{'mov%':>5s} {'accHz':>6s} {'gyrHz':>6s} {'gap_s':>6s} {'hf':>6s}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: -x.get("km", 0)):
        if r.get("error"):
            print(f"{r['file']:32s} {'':>5s} {'':>6s} {'':>5s} {'':>5s} {'':>5s} "
                  f"{'':>6s} {'':>6s} {'':>6s} {'':>6s}  {r['error']}")
            continue
        print(f"{r['file']:32s} {r['min']:5.1f} {r['km']:6.2f} {r['vmed']:5.1f} "
              f"{r['vmax']:5.1f} {100*r['moving']:5.0f} {r['acc']:6.0f} {r['gyr']:6.0f} "
              f"{r['gap']:6.1f} {r['hf']:6.3f}  {r['verdict']}")

    use = [r for r in rows if r.get("verdict") == "USABLE"]
    print(f"\n{len(use)} usable of {len(rows)}   "
          f"total {sum(r['km'] for r in use):.1f} km, "
          f"{sum(r['min'] for r in use):.0f} min")
    if use:
        print("\nlongest usable rides:")
        for r in sorted(use, key=lambda x: -x["km"])[:5]:
            print(f"  {r['file']}  {r['km']:.2f} km  {r['min']:.1f} min  "
                  f"median {r['vmed']:.0f} km/h  hf {r['hf']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/collected/Xiaomi-2201117TI")
