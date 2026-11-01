"""OpenStreetMap road network: fetch, parse, and index.

Kept deliberately small. A full routing engine (Valhalla, OSRM) is the right answer
on a server but far too heavy for a phone, and we only need three things:

  * which road segments are near a point
  * where on a segment the nearest point lies
  * which way the road runs there

The third one is the valuable one. In a tunnel the magnetometer is useless and the
gyro is drifting, so the road's own bearing becomes the heading reference.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .trip import R_EARTH

DRIVABLE = (
    "motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street"
    "|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"
)
OVERPASS = "https://overpass-api.de/api/interpreter"


def fetch_osm(bbox: tuple[float, float, float, float], cache: str | Path) -> dict:
    """Download drivable ways in a lat/lon bbox, caching to disk.

    bbox is (lat_min, lon_min, lat_max, lon_max). A trip-sized area (~5x8 km) comes
    back as roughly 3 MB in a few seconds, which is small enough to ship in an APK.
    """
    cache = Path(cache)
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    query = (
        f'[out:json][timeout:90];way["highway"~"^({DRIVABLE})$"]'
        f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});out geom;"
    )
    req = urllib.request.Request(
        OVERPASS,
        data=urllib.parse.urlencode({"data": query}).encode(),
        headers={"User-Agent": "NaviSense/1.0"},
    )
    raw = urllib.request.urlopen(req, timeout=180).read().decode("utf-8")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(raw, encoding="utf-8")
    return json.loads(raw)


@dataclass
class RoadNetwork:
    """Road segments in a local ENU frame, with a spatial index."""

    a: np.ndarray  # (S,2) segment start, metres east/north
    b: np.ndarray  # (S,2) segment end
    heading: np.ndarray  # (S,) segment bearing in ENU radians
    way_id: np.ndarray  # (S,) which OSM way each segment came from
    tree: cKDTree
    lat0: float
    lon0: float

    @property
    def n_segments(self) -> int:
        return len(self.a)

    def candidates(self, point: np.ndarray, radius: float, k: int = 10) -> np.ndarray:
        """Indices of up to k segments whose midpoint is within `radius` metres."""
        idx = self.tree.query_ball_point(point, radius)
        if len(idx) <= k:
            return np.asarray(idx, dtype=int)
        idx = np.asarray(idx, dtype=int)
        mid = 0.5 * (self.a[idx] + self.b[idx])
        return idx[np.argsort(np.linalg.norm(mid - point, axis=1))[:k]]

    def project(self, point: np.ndarray, seg: np.ndarray):
        """Project a point onto segments. Returns (closest_points, distances)."""
        a, b = self.a[seg], self.b[seg]
        ab = b - a
        denom = np.einsum("ij,ij->i", ab, ab)
        denom = np.where(denom < 1e-9, 1e-9, denom)
        t = np.clip(np.einsum("ij,ij->i", point - a, ab) / denom, 0.0, 1.0)
        closest = a + t[:, None] * ab
        return closest, np.linalg.norm(closest - point, axis=1)


def build_network(osm: dict, lat0: float, lon0: float, max_segment_m: float = 40.0) -> RoadNetwork:
    """Turn Overpass geometry into an indexed set of ENU segments.

    Long ways are split so that no segment exceeds `max_segment_m`; otherwise a
    kilometre-long motorway segment has a midpoint nowhere near the vehicle and the
    nearest-midpoint search misses it entirely.
    """
    cos_lat = np.cos(np.deg2rad(lat0))
    starts, ends, ways = [], [], []

    for el in osm.get("elements", []):
        geom = el.get("geometry")
        if el.get("type") != "way" or not geom or len(geom) < 2:
            continue
        pts = np.array(
            [
                [
                    np.deg2rad(g["lon"] - lon0) * R_EARTH * cos_lat,
                    np.deg2rad(g["lat"] - lat0) * R_EARTH,
                ]
                for g in geom
            ]
        )
        for p, q in zip(pts[:-1], pts[1:]):
            length = float(np.linalg.norm(q - p))
            n_split = max(1, int(np.ceil(length / max_segment_m)))
            for s in range(n_split):
                starts.append(p + (q - p) * (s / n_split))
                ends.append(p + (q - p) * ((s + 1) / n_split))
                ways.append(el["id"])

    a = np.asarray(starts, dtype=float)
    b = np.asarray(ends, dtype=float)
    if len(a) == 0:
        raise ValueError("no drivable ways found in this area")
    delta = b - a
    heading = np.arctan2(delta[:, 1], delta[:, 0])
    return RoadNetwork(
        a=a,
        b=b,
        heading=heading,
        way_id=np.asarray(ways),
        tree=cKDTree(0.5 * (a + b)),
        lat0=lat0,
        lon0=lon0,
    )


def network_for_trip(trip, cache_dir: str | Path = "data/osm", pad_deg: float = 0.01):
    """Fetch and build the road network covering one trip's GPS track."""
    lat, lon = trip.lat, trip.lon
    ok = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) > 0.01)
    lat, lon = lat[ok], lon[ok]
    bbox = (lat.min() - pad_deg, lon.min() - pad_deg, lat.max() + pad_deg, lon.max() + pad_deg)
    safe = "".join(ch if ch.isalnum() else "_" for ch in trip.name)
    osm = fetch_osm(bbox, Path(cache_dir) / f"{safe}.json")
    return build_network(osm, float(trip.lat[0]), float(trip.lon[0]))
