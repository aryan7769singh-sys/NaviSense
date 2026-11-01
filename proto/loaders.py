"""Loaders for real data: Sensor Logger exports and IO-VNBD.

Both land in the same Trip object as the synthetic generator, so nothing downstream
changes when you swap synthetic data for real.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .trip import CANONICAL_HZ, Trip, resample_uniform, unwrap_course


def _read(folder: Path, *names: str) -> pd.DataFrame | None:
    """Find one of several possible CSV filenames, case-insensitively."""
    lowered = {p.name.lower(): p for p in folder.glob("*.csv")}
    for name in names:
        hit = lowered.get(name.lower() + ".csv")
        if hit is not None:
            return pd.read_csv(hit)
    return None


def load_sensor_logger(folder: str | Path, name: str | None = None) -> Trip:
    """Load one unzipped Sensor Logger recording.

    Sensor Logger (free, Android + iOS) writes one CSV per sensor, each with a
    `seconds_elapsed` column. Prefer the UNCALIBRATED accelerometer/gyroscope
    exports when you enable them — vendor-calibrated channels are silently filtered
    and the filtering differs per phone.

    Set the app's sampling rate as high as it will go. The speed cue is
    road-roughness vibration, and CarSpeedNet acquires at 500 Hz before decimating.
    """
    folder = Path(folder)
    acc_df = _read(folder, "AccelerometerUncalibrated", "Accelerometer", "TotalAcceleration")
    gyr_df = _read(folder, "GyroscopeUncalibrated", "Gyroscope")
    loc_df = _read(folder, "Location", "LocationGps")
    missing = [n for n, d in [("accelerometer", acc_df), ("gyroscope", gyr_df), ("location", loc_df)] if d is None]
    if missing:
        raise FileNotFoundError(
            f"{folder}: missing {', '.join(missing)} CSV. Found: "
            f"{sorted(p.name for p in folder.glob('*.csv'))}"
        )

    def xyz(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return df["seconds_elapsed"].to_numpy(float), df[["x", "y", "z"]].to_numpy(float)

    t_acc, acc = xyz(acc_df)
    t_gyr, gyr = xyz(gyr_df)
    t_loc = loc_df["seconds_elapsed"].to_numpy(float)

    # GPS arrives at ~1 Hz, IMU at 100+ Hz. Put everything on the IMU clock.
    t0, t1 = max(t_acc[0], t_gyr[0], t_loc[0]), min(t_acc[-1], t_gyr[-1], t_loc[-1])
    grid = np.arange(t0, t1, 1.0 / CANONICAL_HZ)

    def interp(t_src, arr):
        if arr.ndim == 1:
            return np.interp(grid, t_src, arr)
        return np.column_stack([np.interp(grid, t_src, arr[:, c]) for c in range(arr.shape[1])])

    speed = loc_df["speed"].to_numpy(float)
    speed = np.where(speed < 0, np.nan, speed)  # Android reports -1 when unavailable
    speed = pd.Series(speed).ffill().bfill().to_numpy()

    bearing_col = "bearing" if "bearing" in loc_df.columns else "course"
    bearing = np.deg2rad(loc_df[bearing_col].to_numpy(float))

    return Trip(
        name=name or folder.name,
        t=grid - grid[0],
        acc=interp(t_acc, acc),
        gyr=interp(t_gyr, gyr),
        lat=interp(t_loc, loc_df["latitude"].to_numpy(float)),
        lon=interp(t_loc, loc_df["longitude"].to_numpy(float)),
        v_true=np.clip(interp(t_loc, speed), 0.0, None),
        course=interp(t_loc, unwrap_course(bearing)),
        hz=CANONICAL_HZ,
    )


def load_sensor_logger_dir(root: str | Path) -> list[Trip]:
    """Load every subfolder of `root` that looks like a Sensor Logger export."""
    root = Path(root)
    trips = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            trips.append(load_sensor_logger(sub))
        except (FileNotFoundError, KeyError) as exc:
            print(f"  skipped {sub.name}: {exc}")
    return trips


# --------------------------------------------------------------------------------
# IO-VNBD
# --------------------------------------------------------------------------------
# Verified against the real files, which have several traps:
#
#   * Every column name after the first carries a LEADING SPACE.
#   * Encoding is mixed — the degree and squared signs are Latin-1 while the micro
#     sign is UTF-8, so no single codec decodes the header cleanly. Read as latin-1
#     and match columns on their ASCII content only.
#   * Gyro axes are named Yaw/Pitch/Roll, not X/Y/Z.
#   * GPS speed is km/h; wheel speed is rad/s; orientation is degrees.
#   * Timestamps are NOT trustworthy: one file reports a negative duration and
#     another disagrees with its own paired file. Sampling is a documented uniform
#     10 Hz ("Sample period (seconds)" = 0.1), so time is taken from the row index.
#   * The IMU lives in S-*.csv and the wheel-speed teacher labels live in the paired
#     V-*.csv. The two are row-aligned, which is what "Synchronised" means.

IOVNBD_ENCODING = "latin-1"


def _norm(name: str) -> str:
    """Normalise a column name to ASCII lowercase for robust matching."""
    return "".join(ch for ch in name.lower().strip() if ch.isalnum() or ch == " ").strip()


def _find(df: pd.DataFrame, *fragments: str) -> str | None:
    """First column whose normalised name contains all the given fragments."""
    for col in df.columns:
        norm = _norm(col)
        if all(f in norm for f in fragments):
            return col
    return None


def _read_iovnbd_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=IOVNBD_ENCODING, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def describe_columns(csv_path: str | Path, n: int = 3) -> None:
    """Print a file's columns and sample values — for adapting to a new variant."""
    df = _read_iovnbd_csv(csv_path)
    print(f"{Path(csv_path).name}: {len(df.columns)} columns")
    for col in df.columns:
        vals = ", ".join(f"{v:.6g}" if isinstance(v, float) else str(v)[:20] for v in df[col].head(n))
        print(f"  {col!r:<44} {vals}")


def estimate_sv_lag(s_speed: np.ndarray, v_speed: np.ndarray, max_lag: int = 600) -> int:
    """Find the sample offset between a smartphone file and its paired vehicle file.

    The S-/V- pairs have identical ROW COUNTS, which makes it tempting to assume
    they are sample-aligned. They are not. Measured on S-S1, the true offset is 44
    samples (4.4 s): cross-correlation of the two speed signals scores +0.975 at
    lag 44 versus +0.837 at lag 0.

    Pairing row i with row i would therefore hand the network a speed label from 4.4
    seconds in the past or future. It still trains, and the error is invisible in
    the loss — it just quietly caps how good the model can get.
    """
    n = min(len(s_speed), len(v_speed), 20000)
    a, b = s_speed[:n].astype(float), v_speed[:n].astype(float)
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    xc = np.correlate(a, b, mode="full")
    centre = len(b) - 1
    lo, hi = max(0, centre - max_lag), min(len(xc), centre + max_lag + 1)
    return int(np.argmax(xc[lo:hi]) + lo - centre)


def _apply_lag(lag: int, n: int) -> tuple[slice, slice]:
    """Index slices that align S and V given a lag from estimate_sv_lag.

    Positive lag means the vehicle signal leads: S[j+lag] corresponds to V[j].
    """
    if lag > 0:
        return slice(lag, n), slice(0, n - lag)
    if lag < 0:
        return slice(0, n + lag), slice(-lag, n)
    return slice(0, n), slice(0, n)


def calibrate_wheel_radius(vehicle_df: pd.DataFrame) -> float:
    """Recover effective wheel radius from the data instead of guessing a tyre size.

    Wheel speed is rad/s and the CAN bus separately reports indicated vehicle speed
    in km/h, so the radius is just the least-squares ratio between them. Deriving it
    beats assuming 0.31 m, and it doubles as a sanity check: a sensible car returns
    roughly 0.28-0.34 m.
    """
    rear = [_find(vehicle_df, "wheel speed rear left"), _find(vehicle_df, "wheel speed rear right")]
    indicated = _find(vehicle_df, "indicated vehicle speed")
    if not all(rear) or indicated is None:
        return 0.31
    rad_s = vehicle_df[rear].to_numpy(float).mean(axis=1)
    mps = vehicle_df[indicated].to_numpy(float) / 3.6
    moving = rad_s > 5.0
    if moving.sum() < 100:
        return 0.31
    return float(np.dot(rad_s[moving], mps[moving]) / np.dot(rad_s[moving], rad_s[moving]))


def load_iovnbd_pair(
    smartphone_csv: str | Path,
    vehicle_csv: str | Path | None = None,
    name: str | None = None,
) -> Trip:
    """Load one IO-VNBD recording: smartphone IMU + paired vehicle CAN labels.

    v_true comes from the REAR wheel speeds where the vehicle file is available.
    This is the "wheel speed is a free teacher" idea made concrete: dense, accurate
    speed labels that the network is trained against but never sees at inference,
    so the shipped system needs no OBD-II. Rear wheels on a front-wheel-drive car
    are the least slip-prone pair.

    The data is 10 Hz — below the band where road-roughness vibration lives — so
    expect speed estimation to be harder here than on your own high-rate recordings.
    """
    s = _read_iovnbd_csv(smartphone_csv)
    n = len(s)

    acc_cols = [_find(s, "accelerometer", ax) for ax in ("x", "y", "z")]

    # Gyro axis order is (roll, YAW, PITCH) -> device (x, y, z). Not a typo, and not
    # what the headers suggest.
    #
    # Determined empirically, because the obvious reading is wrong. Integrating each
    # candidate vertical axis over 10 s windows and comparing the heading change
    # against GPS course:
    #
    #     roll,pitch,yaw   corr +0.28   slope 1.94   <- the "obvious" mapping
    #     roll,yaw,pitch   corr +0.88   slope 0.91   <- correct (slope should be 1)
    #
    # It wins on 7 of 8 recordings tested, usually by a wide margin. So the column
    # labelled "Pitch" is really the device z axis, which is also why it carries the
    # largest variance: it is the yaw rate during turns.
    gyr_cols = [_find(s, "gyroscope", ax) for ax in ("roll", "yaw", "pitch")]
    if any(c is None for c in gyr_cols):
        # The flat "Uncategorised" copy labels the same arrays X/Y/Z.
        gyr_cols = [_find(s, "gyroscope", ax) for ax in ("x", "y", "z")]
    lat_c, lon_c = _find(s, "latitude"), _find(s, "longitude")
    missing = [n_ for n_, c in zip(["acc", "gyr", "lat", "lon"], [acc_cols, gyr_cols, lat_c, lon_c])
               if c is None or (isinstance(c, list) and any(x is None for x in c))]
    if missing:
        raise KeyError(f"{Path(smartphone_csv).name}: could not map {missing}")

    acc = s[acc_cols].to_numpy(float)
    # Roll/Pitch/Yaw are the device x/y/z rates respectively.
    gyr = s[gyr_cols].to_numpy(float)

    # The header says "GPS SPEED (Kmh)" but the values are metres per second: its
    # median is 7.18 where the CAN bus reports 27.17 km/h for the same drive, and
    # 27.17 km/h is 7.55 m/s. Trust the numbers, not the label.
    speed_c = _find(s, "gps speed")
    gps_speed = s[speed_c].to_numpy(float) if speed_c else np.zeros(n)
    course_c = _find(s, "gps orientation")
    course = np.deg2rad(s[course_c].to_numpy(float)) if course_c else np.zeros(n)

    lat = s[lat_c].to_numpy(float)
    lon = s[lon_c].to_numpy(float)
    v_true = gps_speed
    lag = 0

    if vehicle_csv is not None and Path(vehicle_csv).exists():
        v = _read_iovnbd_csv(vehicle_csv)
        rear = [_find(v, "wheel speed rear left"), _find(v, "wheel speed rear right")]
        indicated = _find(v, "indicated vehicle speed")
        if all(rear) and indicated is not None:
            # Calibrated against the CAN speedometer, so this is correct whether the
            # column really is rad/s or (as the numbers suggest) already km/h — the
            # fitted factor absorbs either interpretation.
            radius = calibrate_wheel_radius(v)
            wheel_mps = np.clip(v[rear].to_numpy(float).mean(axis=1) * radius, 0.0, None)

            lag = estimate_sv_lag(gps_speed, v[indicated].to_numpy(float))
            s_sl, v_sl = _apply_lag(lag, min(n, len(v)))
            acc, gyr = acc[s_sl], gyr[s_sl]
            lat, lon = lat[s_sl], lon[s_sl]
            gps_speed, course = gps_speed[s_sl], course[s_sl]
            v_true = wheel_mps[v_sl]
            n = len(acc)

    # Row index as the clock — see the note at the top of this section.
    return Trip(
        name=(name or Path(smartphone_csv).stem) + (f"+lag{lag}" if lag else ""),
        t=np.arange(n) / CANONICAL_HZ,
        acc=acc,
        gyr=gyr,
        lat=lat,
        lon=lon,
        v_true=np.clip(np.nan_to_num(v_true[:n]), 0.0, None),
        course=unwrap_course(np.nan_to_num(course)),
        hz=CANONICAL_HZ,
    )


def find_iovnbd_pairs(root: str | Path) -> list[tuple[Path, Path | None]]:
    """Locate unique S-/V- recording pairs, DE-DUPLICATED by recording name.

    IO-VNBD ships every recording twice — once under per-driver folders
    ("Categorised") and once in a flat dump ("Uncategorised"). The arrays are
    byte-identical; only the gyro column headers differ.

    Loading both would put the SAME DRIVE in train and validation, since the split
    is by trip name. That is silent data leakage: the score improves and means
    nothing. So each recording is taken once, preferring the copy that has its
    vehicle twin in the same folder, because that is the copy that yields
    wheel-speed teacher labels.
    """
    best: dict[str, tuple[Path, Path | None]] = {}
    for sp in sorted(Path(root).rglob("S-*.csv")):
        key = sp.name.lower()
        vp = sp.parent / ("V-" + sp.name[2:])
        twin = vp if vp.exists() else None
        if key not in best or (twin is not None and best[key][1] is None):
            best[key] = (sp, twin)
    return [best[k] for k in sorted(best)]


def load_iovnbd_dir(root: str | Path, limit: int | None = None, min_minutes: float = 3.0):
    """Load every unique IO-VNBD recording under `root`."""
    trips = []
    for sp, vp in find_iovnbd_pairs(root):
        try:
            trip = load_iovnbd_pair(sp, vp)
        except (KeyError, ValueError, pd.errors.ParserError) as exc:
            print(f"  skipped {sp.name}: {exc}")
            continue
        if len(trip) / trip.hz / 60.0 < min_minutes:
            continue
        trips.append(trip)
        if limit and len(trips) >= limit:
            break
    return trips
