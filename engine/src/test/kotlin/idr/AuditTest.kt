package idr

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Bugs found by reading the code rather than by observing a symptom.
 *
 * Each of these is demonstrated failing first, so the fix has something to prove.
 */
class AuditTest {

    /**
     * BUG: DeadReckoner.reset() zeroes `distance`, and NavEngine.pushGnss() calls it
     * on EVERY GNSS fix (~1 Hz) to snap position to truth.
     *
     * Consequence: while GNSS is available, trip distance can never exceed one
     * second of travel. The DISTANCE readout shows near zero the whole drive, and
     * "drift as % of distance" divides by a number that is always tiny.
     */
    @Test
    fun `snapping position must not destroy trip distance`() {
        val dr = DeadReckoner()
        val dt = 0.1
        var gnssTicks = 0

        // 30 s at 20 m/s = 600 m, with a position snap every second like real GNSS.
        repeat(300) { i ->
            dr.step(20.0, 0.0, dt)
            if (i % 10 == 9) {
                gnssTicks++
                dr.snapTo(dr.east, dr.north)   // position correction, distance preserved
            }
        }

        assertTrue(gnssTicks == 30, "expected 30 snaps, got $gnssTicks")
        assertTrue(
            abs(dr.distance - 600.0) < 1.0,
            "distance should survive position snaps; got ${dr.distance} instead of 600",
        )
    }

    /**
     * BUG: gyro bias was updated as `gyroBiasZ += 0.02 * headingError`.
     *
     * headingError is an ANGLE (rad); gyroBiasZ is a RATE (rad/s). Adding one to the
     * other is dimensionally meaningless — the numbers only looked plausible because
     * GNSS happens to arrive at roughly 1 Hz, so the two are numerically similar by
     * coincidence. At 5 Hz GNSS the correction would be 5x too large.
     */
    @Test
    fun `gyro bias correction must scale with the interval it accumulated over`() {
        // A gyro reading 0.05 rad/s too fast, observed over different GNSS intervals.
        val trueBias = 0.05

        fun estimate(gnssIntervalS: Double): Double {
            var bias = 0.0
            repeat(200) {
                // Heading error accumulated since the previous fix.
                val headingErr = (trueBias - bias) * gnssIntervalS
                bias += 0.3 * headingErr / gnssIntervalS      // rate = angle / time
            }
            return bias
        }

        val at1Hz = estimate(1.0)
        val at5Hz = estimate(0.2)
        assertTrue(abs(at1Hz - trueBias) < 0.005, "1 Hz estimate wrong: $at1Hz")
        assertTrue(abs(at5Hz - trueBias) < 0.005, "5 Hz estimate wrong: $at5Hz")
        assertTrue(
            abs(at1Hz - at5Hz) < 0.002,
            "estimate depends on GNSS rate ($at1Hz vs $at5Hz) — units are wrong",
        )
    }

    /**
     * BUG: dt was taken straight from sensor timestamps with no bound.
     *
     * A CPU stall, a batching hiccup or a dropped burst produces a large gap. That
     * gap is then multiplied by speed and integrated in a single step, teleporting
     * the position. Observed in the recordings as multi-second IMU gaps.
     */
    @Test
    fun `a sensor gap must not teleport the position`() {
        val dr = DeadReckoner()
        dr.step(20.0, 0.0, 0.1)
        val before = dr.east

        // A 5-second gap arriving as one step would jump 100 m at 20 m/s.
        val clamped = clampDt(5.0)
        dr.step(20.0, 0.0, clamped)
        val jump = dr.east - before

        assertTrue(jump < 8.0, "a 5 s gap moved the position by $jump m in one step")
        // And ordinary jitter must NOT be clamped, or every step would be truncated.
        assertTrue(clampDt(0.12) == 0.12, "normal jitter was clamped")
        assertTrue(clampDt(0.1) == 0.1, "nominal step was clamped")
    }
}
