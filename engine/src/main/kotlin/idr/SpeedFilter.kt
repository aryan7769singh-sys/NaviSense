package idr

import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin

/**
 * Two-state Kalman filter on [forward speed, accelerometer bias].
 *
 * This is where the drift budget is won. None of the three inputs is sufficient
 * alone:
 *
 *   accelerometer  good at short-term CHANGE, diverges over minutes through bias
 *   VelNet         a bounded absolute estimate, but even published state of the art
 *                  is only ~1.8 m/s RMSE, roughly 9% error on its own
 *   ZUPT           near-exact, but only when genuinely stopped
 *
 * Fusing them lets the accelerometer carry the fast dynamics, the learned estimate
 * stop the drift, and every stop reset the bias that caused the drift.
 *
 * Ported from proto/dr.py `fuse_speed`, tuned against real IO-VNBD data.
 */
class SpeedFilter(
    private val qSpeed: Double = 0.6,
    private val qBias: Double = 1e-4,
    private val rZupt: Double = 0.25,
) {
    var speed: Double = 0.0
        private set
    var accelBias: Double = 0.0
        private set

    private var p00 = 1.0
    private var p01 = 0.0
    private var p10 = 0.0
    private var p11 = 0.05

    fun reset(v0: Double) {
        speed = max(v0, 0.0); accelBias = 0.0
        p00 = 1.0; p01 = 0.0; p10 = 0.0; p11 = 0.05
    }

    /** Propagate with measured forward acceleration over [dt] seconds. */
    fun predict(aForward: Double, dt: Double) {
        speed += (aForward - accelBias) * dt

        // F = [[1, -dt], [0, 1]];  P = F P F' + Q
        val n00 = p00 - dt * (p10 + p01) + dt * dt * p11 + qSpeed * dt
        val n01 = p01 - dt * p11
        val n10 = p10 - dt * p11
        val n11 = p11 + qBias * dt
        p00 = n00; p01 = n01; p10 = n10; p11 = n11

        if (speed < 0.0) speed = 0.0
    }

    /** Scalar update on speed with measurement [z] and variance [r]. */
    fun update(z: Double, r: Double) {
        val s = p00 + max(r, 1e-6)
        val k0 = p00 / s
        val k1 = p10 / s
        val y = z - speed

        speed += k0 * y
        accelBias += k1 * y

        val n00 = (1 - k0) * p00
        val n01 = (1 - k0) * p01
        val n10 = p10 - k1 * p00
        val n11 = p11 - k1 * p01
        p00 = n00; p01 = n01; p10 = n10; p11 = n11

        if (speed < 0.0) speed = 0.0
    }

    /** Learned speed observation, weighted by the model's own reported sigma. */
    fun updateLearned(vHat: Double, sigma: Double) = update(vHat, sigma * sigma)

    /**
     * Zero-velocity update, applied as a SOFT low-speed constraint.
     *
     * The stationary detector runs around 40% precision at 10 Hz, so a hard zero on
     * a false positive would stall the estimate mid-motorway. Measured on real data,
     * a hard ZUPT with a naive detector tripled speed error.
     */
    fun updateZupt() = update(0.0, rZupt)

    val speedVariance: Double get() = p00
}

/** A dead-reckoned position, integrated from speed and heading. */
class DeadReckoner(startEast: Double = 0.0, startNorth: Double = 0.0, startHeading: Double = 0.0) {
    var east: Double = startEast; private set
    var north: Double = startNorth; private set
    var heading: Double = startHeading; private set
    var distance: Double = 0.0; private set

    /** Full reset: position, heading AND trip distance. Use at the start of a leg. */
    fun reset(east: Double, north: Double, heading: Double) {
        this.east = east; this.north = north; this.heading = heading; distance = 0.0
    }

    /**
     * Correct position only, preserving accumulated trip distance.
     *
     * Snapping to a GNSS fix is a position correction, not a new journey. Using
     * reset() for it — as the app did on every fix, about once a second — meant trip
     * distance could never exceed one second of travel, so the DISTANCE readout sat
     * near zero and "drift as % of distance" divided by almost nothing.
     */
    fun snapTo(east: Double, north: Double) {
        this.east = east; this.north = north
    }

    /** Advance by [dt] using forward [speed] and vertical [yawRate]. */
    fun step(speed: Double, yawRate: Double, dt: Double) {
        heading = wrapPi(heading + yawRate * dt)
        val d = speed * dt
        east += d * cos(heading)
        north += d * sin(heading)
        distance += d
    }

    /**
     * Blend an external heading reference in, rather than replacing.
     *
     * Used for the road bearing from map matching: measured at 2-3 degrees error
     * against a gyro drifting ~18 degrees per minute. Blending stops a single wrong
     * match from yanking the track onto a parallel side street.
     */
    fun nudgeHeading(reference: Double, gain: Double = 0.7) {
        heading = wrapPi(heading + gain * wrapPi(reference - heading))
    }
}
