"""Hidden Markov map matching (Newson & Krumm), with the constraint that matters.

The point of map matching here is NOT to make the trace look tidy. It is that a
tunnel is topologically one road with no branches, so knowing which road you are on
collapses the error from two dimensions to one:

  cross-track   pinned to the centreline -> effectively zero
  along-track   still unknown, still accumulating
  heading       read off the road bearing, which is what replaces a drifting gyro
                and a magnetometer that is useless inside a rebar tunnel

That is why this component, not a better speed model, is what reaches the 10% spec.

Newson & Krumm's transition term normally uses shortest-path route distance between
candidates, which needs a routing engine. On a dense urban network straight-line
distance is a good proxy and costs nothing, so that is what is used here — the
approximation shows up only where roads are sparse and winding.
"""
from __future__ import annotations

import numpy as np

from .osm import RoadNetwork
from .trip import wrap_pi


def match(
    east: np.ndarray,
    north: np.ndarray,
    heading: np.ndarray | None,
    network: RoadNetwork,
    sigma_z: float = 20.0,
    beta: float = 12.0,
    radius: float = 90.0,
    max_candidates: int = 8,
    heading_weight: float = 1.5,
    step: int = 5,
):
    """Viterbi-decode the most likely sequence of road segments for a trajectory.

    `step` decimates the input: matching at 2 Hz rather than 10 Hz costs a fifth as
    much and loses nothing, because a vehicle cannot change road in half a second.

    Returns (snapped_east, snapped_north, road_heading, matched) at full input length.
    """
    n = len(east)
    idx = np.arange(0, n, step)
    pts = np.column_stack([east[idx], north[idx]])

    cands: list[np.ndarray] = []
    projs: list[np.ndarray] = []
    emissions: list[np.ndarray] = []

    for i, p in enumerate(pts):
        c = network.candidates(p, radius, max_candidates)
        if len(c) == 0:
            cands.append(np.empty(0, int))
            projs.append(np.empty((0, 2)))
            emissions.append(np.empty(0))
            continue
        closest, dist = network.project(p, c)
        # Emission: how plausible is this observation given we are on that segment.
        logp = -0.5 * (dist / sigma_z) ** 2
        if heading is not None and heading_weight > 0:
            # A road matching our direction of travel is far more likely than one
            # crossing it. Road bearing is undirected, so fold to [0, pi/2].
            dh = np.abs(wrap_pi(network.heading[c] - heading[idx[i]]))
            dh = np.minimum(dh, np.pi - dh)
            logp = logp - heading_weight * (dh / (np.pi / 2)) ** 2
        cands.append(c)
        projs.append(closest)
        emissions.append(logp)

    # --- Viterbi ---------------------------------------------------------------
    first = next((i for i, c in enumerate(cands) if len(c)), None)
    if first is None:
        return east.copy(), north.copy(), np.zeros(n), np.zeros(n, bool)

    score = emissions[first].copy()
    back: list[np.ndarray] = [np.full(len(score), -1)]
    active = [first]

    for i in range(first + 1, len(pts)):
        if len(cands[i]) == 0:
            continue
        prev = active[-1]
        gps_step = float(np.linalg.norm(pts[i] - pts[prev]))
        # Transition: penalise candidate pairs whose separation disagrees with how
        # far the vehicle actually moved between the two observations.
        route = np.linalg.norm(projs[i][:, None, :] - projs[prev][None, :, :], axis=2)
        trans = -np.abs(route - gps_step) / beta

        total = trans + score[None, :]
        best = np.argmax(total, axis=1)
        score = total[np.arange(len(cands[i])), best] + emissions[i]
        score -= score.max()  # keep it numerically bounded
        back.append(best)
        active.append(i)

    # --- backtrace -------------------------------------------------------------
    path = np.empty(len(active), dtype=int)
    path[-1] = int(np.argmax(score))
    for k in range(len(active) - 1, 0, -1):
        path[k - 1] = back[k][path[k]]

    snap_e = np.full(n, np.nan)
    snap_n = np.full(n, np.nan)
    road_h = np.full(n, np.nan)
    for k, i in enumerate(active):
        j = path[k]
        snap_e[idx[i]] = projs[i][j, 0]
        snap_n[idx[i]] = projs[i][j, 1]
        road_h[idx[i]] = network.heading[cands[i][j]]

    matched = np.isfinite(snap_e)
    if matched.sum() < 2:
        return east.copy(), north.copy(), np.zeros(n), np.zeros(n, bool)

    # Interpolate between matched samples so the output is full-rate.
    t = np.arange(n)
    snap_e = np.interp(t, t[matched], snap_e[matched])
    snap_n = np.interp(t, t[matched], snap_n[matched])
    # Headings interpolate through their unit vectors, never as raw angles.
    hx = np.interp(t, t[matched], np.cos(road_h[matched]))
    hy = np.interp(t, t[matched], np.sin(road_h[matched]))
    road_h = np.arctan2(hy, hx)
    return snap_e, snap_n, road_h, matched


def constrain_dead_reckoning(
    east: np.ndarray,
    north: np.ndarray,
    heading: np.ndarray,
    network: RoadNetwork,
    cross_track_gain: float = 0.85,
    **kw,
):
    """Pull a dead-reckoned track onto the road network, ACROSS-TRACK ONLY.

    This is the part that is easy to get wrong. The matcher knows which road you are
    on, so it knows your cross-track position well — but it does NOT know how far
    along that road you are, because that is exactly the information dead reckoning
    lost. Snapping fully to the matched point would overwrite the along-track
    estimate with the matcher's guess and can make the error worse.

    So the correction is decomposed in the road frame and only the perpendicular
    component is applied.
    """
    snap_e, snap_n, road_h, matched = match(east, north, heading, network, **kw)
    if matched.sum() < 2:
        return east.copy(), north.copy(), road_h, matched

    delta = np.column_stack([snap_e - east, snap_n - north])
    across = np.column_stack([-np.sin(road_h), np.cos(road_h)])
    cross_component = np.einsum("ij,ij->i", delta, across)
    correction = cross_track_gain * cross_component[:, None] * across
    return east + correction[:, 0], north + correction[:, 1], road_h, matched
