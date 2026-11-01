"""The figures that go in the deck."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#141A1C"
ACCENT = "#0D6E78"
CRIT = "#A93A2C"
WARN = "#A96A05"
MUTED = "#9AA6A8"


def _style(ax) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(alpha=0.18, linewidth=0.6)


def trajectory_figure(row: dict, out: str | Path, title: str = "") -> Path:
    """The money shot: one blackout, truth vs ours vs a naive strapdown baseline."""
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=160)
    _style(ax)

    te, tn = row["truth"]
    ax.plot(te, tn, color=MUTED, linewidth=7, alpha=0.55, solid_capstyle="round", label="Actual route")

    fused, naive = row["tracks"]["fused"], row["tracks"]["naive"]
    ax.plot(naive.east, naive.north, color=CRIT, linewidth=1.9, label="Naive strapdown INS")
    ax.plot(fused.east, fused.north, color=ACCENT, linewidth=2.4, label="Ours (VelNet + filter)")

    ax.scatter([te[0]], [tn[0]], s=80, color=WARN, zorder=5, edgecolor="white", linewidth=1.5)
    ax.annotate(
        "GNSS lost",
        (te[0], tn[0]),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=9,
        color=WARN,
        weight="bold",
    )
    for track, colour in ((fused, ACCENT), (naive, CRIT)):
        ax.scatter([track.east[-1]], [track.north[-1]], s=42, color=colour, zorder=5,
                   edgecolor="white", linewidth=1.2)
    ax.scatter([te[-1]], [tn[-1]], s=42, color=INK, zorder=5, edgecolor="white", linewidth=1.2)

    ax.set_xlabel("East (m)", fontsize=9, color=INK)
    ax.set_ylabel("North (m)", fontsize=9, color=INK)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(frameon=False, fontsize=9, loc="best")

    head = title or f"{row['duration_s']}s GNSS blackout, {row['distance_m']:.0f} m travelled"
    ax.set_title(head, fontsize=11, color=INK, weight="bold", loc="left", pad=14)
    ax.text(
        0.0, 1.005,
        f"ours {row['err_fused_m']:.0f} m ({row['drift_fused_pct']:.1f}%)     "
        f"naive {row['err_naive_m']:.0f} m ({row['drift_naive_pct']:.1f}%)",
        transform=ax.transAxes, fontsize=9, color=MUTED,
    )

    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def drift_figure(summary: dict, out: str | Path) -> Path:
    """Drift vs outage duration for each method, against the 10% spec line."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    _style(ax)

    durations = sorted(summary)
    series = [
        ("median_naive", CRIT, "Naive strapdown INS", "o-"),
        ("median_velnet", WARN, "VelNet only (no filter)", "s-"),
        ("median_fused", ACCENT, "VelNet + fusion filter", "D-"),
    ]
    for key, colour, label, marker in series:
        ax.plot(durations, [summary[d][key] for d in durations], marker,
                color=colour, linewidth=2.0, markersize=5, label=label)

    ax.axhline(10.0, color=INK, linestyle="--", linewidth=1.2)
    ax.text(durations[-1], 10.6, "ISRO spec: 10%", ha="right", fontsize=9, color=INK, weight="bold")

    ax.set_xlabel("GNSS blackout duration (s)", fontsize=9, color=INK)
    ax.set_ylabel("Position drift (% of distance)", fontsize=9, color=INK)
    ax.set_title("Drift vs blackout duration", fontsize=11, color=INK, weight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def speed_figure(trip, ctx, out: str | Path, seconds: float = 120.0) -> Path:
    """VelNet's speed estimate against truth — the core claim, shown directly."""
    n = min(len(trip), int(seconds * trip.hz))
    t = np.arange(n) / trip.hz

    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=160)
    _style(ax)
    ax.plot(t, trip.v_true[:n] * 3.6, color=MUTED, linewidth=4, alpha=0.7, label="True speed")
    ax.plot(t, ctx["v_hat"][:n] * 3.6, color=ACCENT, linewidth=1.6, label="VelNet (IMU only)")
    ax.fill_between(
        t,
        (ctx["v_hat"][:n] - ctx["sigma"][:n]) * 3.6,
        (ctx["v_hat"][:n] + ctx["sigma"][:n]) * 3.6,
        color=ACCENT, alpha=0.18, linewidth=0, label="±1σ (model's own uncertainty)",
    )
    ax.set_xlabel("time (s)", fontsize=9, color=INK)
    ax.set_ylabel("speed (km/h)", fontsize=9, color=INK)
    ax.set_title("Speed from IMU alone — no GPS, no OBD-II", fontsize=11, color=INK,
                 weight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=3)

    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def pick_representative(rows: list[dict], duration_s: int = 60) -> dict:
    """A window whose result is typical, not cherry-picked — median drift at this duration."""
    sel = [r for r in rows if r["duration_s"] == duration_s]
    if not sel:
        sel = rows
    drifts = np.array([r["drift_fused_pct"] for r in sel])
    return sel[int(np.argsort(drifts)[len(drifts) // 2])]
