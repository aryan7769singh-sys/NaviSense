package idr

import kotlin.math.abs

/**
 * Tracks and removes the static component of forward specific force.
 *
 * Why this exists: alignment is never perfect, and any residual tilt leaves a slice
 * of gravity in the forward axis. One degree of tilt is 0.17 m/s^2 — which happens
 * to be roughly the MEDIAN magnitude of real vehicle acceleration (measured at
 * 0.174 m/s^2 on IO-VNBD). Integrated, that invents about 10 m/s of speed per
 * minute, and it is why the position crept while the phone sat still.
 *
 * The time constant matters. Real accelerations last a few seconds and must pass
 * through untouched; the tilt offset is effectively constant. A 30 s reference
 * separates them cleanly.
 */
class ForwardBias(private val tauSeconds: Double = 30.0) {

    var value: Double = 0.0
        private set
    private var stationarySeconds = 0.0

    /** True once enough stationary time has been observed to trust the estimate. */
    val primed: Boolean get() = stationarySeconds >= 1.5

    /** Feed one sample, get back the acceleration with the static component removed. */
    fun correct(aForward: Double, dt: Double): Double {
        val alpha = dt / (tauSeconds + dt)
        value += alpha * (aForward - value)
        return aForward - value
    }

    /**
     * Learn fast from a moment the vehicle is KNOWN to be stopped.
     *
     * Forward acceleration is then certainly zero, so whatever the accelerometer
     * reports is the offset — far more information than the slow filter gets while
     * driving.
     *
     * This must be gated on observed stillness, not on elapsed time. An earlier
     * version primed over the first 2 seconds unconditionally, which cancelled out
     * real acceleration whenever recording started with the vehicle already moving.
     *
     * Fixes the reported symptom directly: without it the slow filter needs ~30 s to
     * converge and integrates raw tilt in the meantime, reaching 4.41 m/s of invented
     * speed after 60 s on a device that never moved.
     */
    fun observeStationary(aForward: Double, dt: Double = 0.1) {
        stationarySeconds += dt
        // Running mean over stationary samples: converges in about a second.
        val w = (dt / stationarySeconds.coerceAtLeast(dt)).coerceIn(0.02, 1.0)
        value += w * (aForward - value)
    }

    fun reset() { value = 0.0; stationarySeconds = 0.0 }
}

/**
 * Decides when the vehicle is genuinely stopped, so a zero-velocity update can fire.
 *
 * THE TRAP THIS AVOIDS: an accelerometer cannot distinguish "stopped" from "moving
 * at constant velocity" — that is Galilean relativity, not a sensor defect. The
 * usual tie-breaker is idle engine vibration, which lives at 20-100 Hz.
 *
 * At 10 Hz that band is above Nyquist and simply does not exist, so an earlier
 * version gated the decision on the LEARNED speed instead. That was circular: the
 * correction which fixes a wrong speed estimate only fired when the estimate was
 * already right, so it was disabled exactly when it was needed. Observed result:
 * the tracker showed motion while the phone lay still.
 *
 * Three independent sources now, any of which is sufficient. At 400 Hz the physics
 * route is available again, which is what breaks the circularity.
 */
class StationaryDetector(
    private val hfThreshold: Double = 0.06,
    private val yawRateMax: Double = 0.06,
    private val gnssSpeedMax: Double = 0.5,
    private val learnedSpeedMax: Double = 1.5,
) {
    enum class Source { NONE, GNSS, PHYSICS, MODEL }

    /**
     * @param gnssSpeed metres/second, or null when GNSS is unavailable or withheld
     * @param hfEnergy  short-window variability of raw |acc|, from the FULL-RATE stream
     * @param yawRate   rad/s about vertical
     * @param learnedSpeed the network's estimate, m/s
     */
    fun assess(
        gnssSpeed: Double?,
        hfEnergy: Double,
        yawRate: Double,
        learnedSpeed: Double,
    ): Source {
        // A turning vehicle is not stopped, whatever anything else claims.
        if (abs(yawRate) >= yawRateMax) return Source.NONE

        // Ground truth beats every inference, so it is checked first.
        if (gnssSpeed != null && gnssSpeed < gnssSpeedMax) return Source.GNSS
        if (hfEnergy < hfThreshold) return Source.PHYSICS
        if (learnedSpeed < learnedSpeedMax) return Source.MODEL
        return Source.NONE
    }
}

/**
 * Running estimate of high-frequency vibration energy.
 *
 * Must be fed the RAW stream, before decimation: the band that separates a stopped
 * vehicle from a slowly rolling one sits above 5 Hz, so at 10 Hz it is already gone.
 */
class VibrationEnergy(private val alpha: Double = 0.02, initialMean: Double = 9.81) {

    private var mean = initialMean
    var value: Double = 0.0
        private set

    fun update(accelMagnitude: Double): Double {
        mean += alpha * (accelMagnitude - mean)
        value += alpha * (abs(accelMagnitude - mean) - value)
        return value
    }

    fun reset(initialMean: Double = 9.81) {
        mean = initialMean
        value = 0.0
    }
}
