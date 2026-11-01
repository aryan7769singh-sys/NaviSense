"""Synthetic drive generator.

Exists so the whole pipeline is testable with zero real data — which matters when a
dataset download or a data-collection drive is the long pole. It also generates IMU
at 100 Hz with engine vibration, then decimates to 10 Hz, so it exercises the
anti-alias front-end that IO-VNBD (already 10 Hz) structurally cannot.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation

from .trip import R_EARTH, Trip

SYNTH_HZ = 200.0  # generate high-rate so engine/road vibration is representable
GRAVITY = 9.80665

# Speed information in an accelerometer does NOT come from integrating longitudinal
# acceleration — that only gives you dv/dt, and bias makes it diverge. It comes from
# road-roughness excitation, whose intensity scales with how fast you are driving
# (CarSpeedNet, arXiv:2401.07468). The sprung mass resonates around 1.2-1.6 Hz, which
# is the part that survives decimation to 10 Hz and is therefore the part a model
# trained on 10 Hz data can actually use.
SUSPENSION_HZ = 1.4
SUSPENSION_Q = 1.6
GEAR_RATIOS = (0.055, 0.032, 0.022, 0.017, 0.014)  # crude rev/s per m/s, per gear
IDLE_RPS = 13.0  # ~800 rpm

# A plausible mixed drive: (duration_s, target_speed_mps, yaw_rate_rad_s)
DEFAULT_ROUTE = [
    (12, 0.0, 0.0),  # idling at a light — engine on, vehicle stationary
    (18, 14.0, 0.0),  # pull away, accelerate to ~50 km/h
    (25, 14.0, 0.0),  # cruise
    (9, 10.0, -0.28),  # right turn
    (30, 18.0, 0.0),  # accelerate to ~65 km/h
    (14, 18.0, 0.02),  # long gentle left sweep
    (20, 8.0, 0.0),  # decelerate for traffic
    (10, 0.0, 0.0),  # stopped
    (22, 22.0, 0.0),  # highway on-ramp, ~80 km/h
    (45, 24.0, -0.015),  # highway cruise, very gentle curve
    (11, 16.0, 0.34),  # sharp left exit
    (28, 16.0, 0.0),  # cruise
    (16, 6.0, 0.0),  # slow for junction
    (8, 6.0, -0.30),  # right turn
    (30, 15.0, 0.0),  # final cruise
]


def _speed_profile(route, hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Expand the route table into per-sample speed and yaw-rate series."""
    speeds, yaws = [], []
    v_prev = 0.0
    for dur, v_target, omega in route:
        n = max(1, int(round(dur * hz)))
        # Ramp toward the target rather than stepping — a step would inject an
        # impulse into the accelerometer that no real vehicle produces.
        ramp = np.linspace(v_prev, v_target, n)
        speeds.append(ramp)
        yaws.append(np.full(n, omega))
        v_prev = v_target
    v = np.concatenate(speeds)
    omega = np.concatenate(yaws)

    # Smooth both so derivatives stay physical.
    sos = butter(2, 0.4 / (hz / 2), btype="low", output="sos")
    v = np.clip(sosfiltfilt(sos, v), 0.0, None)
    omega = sosfiltfilt(sos, omega)
    # A stationary vehicle cannot be turning.
    omega = np.where(v < 0.3, 0.0, omega)
    return v, omega


def _decimate(x: np.ndarray, factor: int, hz_in: float) -> np.ndarray:
    """Anti-alias low-pass, then downsample.

    Without the low-pass, engine vibration at 20-100 Hz folds straight into the
    signal band and the model learns noise that isn't there at deployment.
    """
    nyq_out = (hz_in / factor) / 2.0
    sos = butter(4, 0.8 * nyq_out / (hz_in / 2), btype="low", output="sos")
    filtered = sosfiltfilt(sos, x, axis=0)
    return filtered[::factor]


def _road_excitation(v: np.ndarray, t: np.ndarray, rng) -> np.ndarray:
    """Road roughness felt through the suspension — the actual speed carrier.

    A vehicle crossing a fixed spatial roughness profile at speed v excites its
    suspension at a rate proportional to v, and with intensity that grows with v.
    Passing that through the sprung-mass resonance puts a speed-dependent signature
    at ~1.4 Hz, low enough to survive 10 Hz sampling.

    Without this term the accelerometer contains no absolute-speed information at
    all on a straight road, and no model can learn one.
    """
    n = len(v)
    # Broadband road input, intensity rising with speed.
    intensity = (v / 20.0) ** 0.75
    white = rng.normal(0.0, 1.0, size=(n, 3)) * intensity[:, None]

    # Sprung-mass resonance: a lightly damped 2nd-order band-pass.
    lo = SUSPENSION_HZ * (1 - 1 / (2 * SUSPENSION_Q))
    hi = SUSPENSION_HZ * (1 + 1 / (2 * SUSPENSION_Q))
    sos = butter(2, [lo / (SYNTH_HZ / 2), hi / (SYNTH_HZ / 2)], btype="band", output="sos")
    body = sosfiltfilt(sos, white, axis=0) * 6.0

    # Higher-frequency tyre/road hiss, also speed-scaled. Mostly removed by the
    # anti-alias filter, which is exactly the point of having it.
    sos_hi = butter(2, [18.0 / (SYNTH_HZ / 2), 70.0 / (SYNTH_HZ / 2)], btype="band", output="sos")
    tyre = sosfiltfilt(sos_hi, white, axis=0) * 2.5

    # Vertical axis takes the most road input, longitudinal the least.
    return (body + tyre) * np.array([0.45, 0.6, 1.0])


def _engine_vibration(v: np.ndarray, t: np.ndarray, rng, amplitude_g: float) -> np.ndarray:
    """Engine order vibration whose frequency tracks RPM, and RPM tracks speed.

    Gear selection makes this piecewise, which is a genuine (if messy) extra cue —
    and a reason a learned model beats a hand-tuned filter here.
    """
    gear = np.clip(np.digitize(v, [4.0, 8.0, 13.0, 19.0]), 0, len(GEAR_RATIOS) - 1)
    rps = np.maximum(IDLE_RPS, v * np.array(GEAR_RATIOS)[gear] * 60.0)
    firing_hz = np.clip(rps * 2.0, 0.0, 0.42 * SYNTH_HZ)  # 4-cyl 4-stroke, keep below Nyquist

    phase = 2 * np.pi * np.cumsum(firing_hz) / SYNTH_HZ
    amp = amplitude_g * GRAVITY * (0.6 + 0.4 * (v / 25.0))
    return amp[:, None] * np.column_stack(
        [np.sin(phase + p) for p in rng.uniform(0, 2 * np.pi, 3)]
    )


def make_trip(
    name: str = "synth-00",
    seed: int = 0,
    route=None,
    mount_rotation: Rotation | None = None,
    gyro_bias_dps: float = 0.6,
    accel_bias: float = 0.08,
    vibration_g: float = 0.05,
    gps_noise_m: float = 2.5,
) -> Trip:
    """Generate one synthetic drive as a Trip at the canonical 10 Hz.

    mount_rotation maps DEVICE-frame vectors into the VEHICLE frame. Leave it None
    for a random mount, which is what makes this useful for testing alignment.
    """
    rng = np.random.default_rng(seed)
    route = route or DEFAULT_ROUTE

    v, omega = _speed_profile(route, SYNTH_HZ)
    n = len(v)
    dt = 1.0 / SYNTH_HZ
    t = np.arange(n) * dt

    # --- true trajectory in ENU -------------------------------------------------
    psi = np.cumsum(omega) * dt  # ENU yaw, 0 = East, CCW positive
    east = np.cumsum(v * np.cos(psi)) * dt
    north = np.cumsum(v * np.sin(psi)) * dt

    # --- specific force in the VEHICLE frame (x fwd, y left, z up) --------------
    a_fwd = np.gradient(v, dt)
    a_lat = v * omega  # centripetal, positive to the left in a left turn
    f_vehicle = np.column_stack([a_fwd, a_lat, np.full(n, GRAVITY)])
    gyr_vehicle = np.column_stack([np.zeros(n), np.zeros(n), omega])

    # Road-induced pitch/roll disturbance — small, but it is what makes gravity
    # estimation non-trivial and stops the model seeing an unrealistically clean signal.
    road = rng.normal(0.0, 0.012, size=(n, 3))
    sos_road = butter(2, 3.0 / (SYNTH_HZ / 2), btype="low", output="sos")
    gyr_vehicle += sosfiltfilt(sos_road, road, axis=0)

    # --- rotate into the DEVICE frame -------------------------------------------
    if mount_rotation is None:
        mount_rotation = Rotation.random(random_state=int(rng.integers(1 << 31)))
    r_dv = mount_rotation.as_matrix()  # device -> vehicle
    acc = f_vehicle @ r_dv  # equivalent to (r_dv.T @ v_vehicle) per row
    gyr = gyr_vehicle @ r_dv

    # --- corrupt it like a real phone -------------------------------------------
    acc = acc + _road_excitation(v, t, rng) + _engine_vibration(v, t, rng, vibration_g)

    acc += rng.normal(0.0, 0.09, size=acc.shape) + accel_bias * rng.normal(0, 1, size=(1, 3))
    gyr += rng.normal(0.0, 0.004, size=gyr.shape) + np.deg2rad(gyro_bias_dps) * rng.normal(
        0, 1, size=(1, 3)
    )

    # --- decimate 100 Hz -> 10 Hz ------------------------------------------------
    factor = int(SYNTH_HZ / 10.0)
    acc_d = _decimate(acc, factor, SYNTH_HZ)
    gyr_d = _decimate(gyr, factor, SYNTH_HZ)
    v_d = v[::factor]
    psi_d = psi[::factor]
    east_d, north_d = east[::factor], north[::factor]
    m = min(len(acc_d), len(v_d))
    acc_d, gyr_d, v_d, psi_d = acc_d[:m], gyr_d[:m], v_d[:m], psi_d[:m]
    east_d, north_d = east_d[:m], north_d[:m]

    # --- GPS ---------------------------------------------------------------------
    lat0, lon0 = 28.6139, 77.2090  # New Delhi, so plotted output looks plausible
    gps_e = east_d + rng.normal(0, gps_noise_m, m)
    gps_n = north_d + rng.normal(0, gps_noise_m, m)
    lat = lat0 + np.rad2deg(gps_n / R_EARTH)
    lon = lon0 + np.rad2deg(gps_e / (R_EARTH * np.cos(np.deg2rad(lat0))))
    course = np.pi / 2.0 - psi_d  # ENU yaw -> GPS course

    return Trip(
        name=name,
        t=np.arange(m) / 10.0,
        acc=acc_d,
        gyr=gyr_d,
        lat=lat,
        lon=lon,
        v_true=v_d,
        course=course,
        hz=10.0,
    )


def make_dataset(n_trips: int = 12, seed: int = 0) -> list[Trip]:
    """A set of trips, each with a different random mount pose and route jitter.

    Different mounts per trip is the point: a model trained on this cannot learn a
    single fixed phone orientation.
    """
    rng = np.random.default_rng(seed)
    trips = []
    for i in range(n_trips):
        route = [
            (
                max(5, dur + int(rng.integers(-4, 5))),
                max(0.0, spd * float(rng.uniform(0.82, 1.18))),
                yaw * float(rng.uniform(0.8, 1.2)),
            )
            for dur, spd, yaw in DEFAULT_ROUTE
        ]
        rng.shuffle(route)
        trips.append(make_trip(name=f"synth-{i:02d}", seed=seed * 1000 + i, route=route))
    return trips
