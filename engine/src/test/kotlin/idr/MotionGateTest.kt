package idr

import kotlin.math.abs
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests for the logic that fixes the "position creeps while the phone is still" bug.
 *
 * Each test corresponds to a failure that actually happened, not a hypothetical.
 */
class ForwardBiasTest {

    @Test
    fun `removes a constant tilt offset`() {
        // One degree of residual tilt leaves this much gravity in the forward axis.
        val tiltOffset = 0.17
        val bias = ForwardBias(tauSeconds = 5.0)
        var corrected = 0.0
        repeat(2000) { corrected = bias.correct(tiltOffset, 0.01) }   // 20 s
        assertTrue(abs(corrected) < 0.01, "offset not removed: $corrected")
        assertEquals(tiltOffset, bias.value, 0.01, "bias not identified")
    }

    @Test
    fun `does NOT eat a real acceleration`() {
        // A vehicle pulling away accelerates for a few seconds. With a 30 s
        // reference that must pass through nearly untouched, or we would filter out
        // the very signal we integrate.
        val bias = ForwardBias(tauSeconds = 30.0)
        var last = 0.0
        repeat(300) { last = bias.correct(1.5, 0.01) }   // 3 s at 1.5 m/s^2
        assertTrue(last > 1.35, "real acceleration was attenuated to $last")
    }

    @Test
    fun `a constant offset would otherwise invent speed`() {
        // The bug, demonstrated: same input, with and without correction.
        val dt = 0.01
        val seconds = 60
        val tiltOffset = 0.17

        val uncorrected = SpeedFilter()
        uncorrected.reset(0.0)
        repeat(seconds * 100) { uncorrected.predict(tiltOffset, dt) }

        val bias = ForwardBias(tauSeconds = 5.0)
        val corrected = SpeedFilter()
        corrected.reset(0.0)
        repeat(seconds * 100) { corrected.predict(bias.correct(tiltOffset, dt), dt) }

        assertTrue(uncorrected.speed > 3.0,
            "expected phantom speed without correction, got ${uncorrected.speed}")
        assertTrue(corrected.speed < 1.0,
            "correction failed; phantom speed still ${corrected.speed}")
    }

    @Test
    fun `stationary observation identifies bias faster than the slow filter`() {
        val slow = ForwardBias(tauSeconds = 30.0)
        val fast = ForwardBias(tauSeconds = 30.0)
        repeat(100) {
            slow.correct(0.17, 0.01)
            fast.correct(0.17, 0.01)
            fast.observeStationary(0.17)
        }
        assertTrue(fast.value > slow.value * 2,
            "stationary hint gave no advantage: ${fast.value} vs ${slow.value}")
    }
}

class StationaryDetectorTest {

    private val d = StationaryDetector()

    @Test
    fun `gnss overrides a wrong model — the circularity fix`() {
        // This is the exact failure: the model wrongly reports 25 m/s while the
        // vehicle is parked. The old gate required the model to agree, so the
        // correction never fired. GNSS must be able to override it.
        val s = d.assess(gnssSpeed = 0.0, hfEnergy = 5.0, yawRate = 0.0, learnedSpeed = 25.0)
        assertEquals(StationaryDetector.Source.GNSS, s)
    }

    @Test
    fun `physics overrides a wrong model when gnss is withheld`() {
        // During a blackout there is no GNSS, so low vibration must be enough.
        val s = d.assess(gnssSpeed = null, hfEnergy = 0.01, yawRate = 0.0, learnedSpeed = 25.0)
        assertEquals(StationaryDetector.Source.PHYSICS, s)
    }

    @Test
    fun `does not fire while genuinely driving`() {
        val s = d.assess(gnssSpeed = 14.0, hfEnergy = 0.8, yawRate = 0.01, learnedSpeed = 13.0)
        assertEquals(StationaryDetector.Source.NONE, s)
    }

    @Test
    fun `does not fire while turning even if everything else looks still`() {
        // A vehicle rotating is moving. Firing a zero-velocity update mid-corner
        // would be actively harmful.
        val s = d.assess(gnssSpeed = 0.0, hfEnergy = 0.0, yawRate = 0.3, learnedSpeed = 0.0)
        assertEquals(StationaryDetector.Source.NONE, s)
    }

    @Test
    fun `model alone still works as a last resort`() {
        val s = d.assess(gnssSpeed = null, hfEnergy = 0.5, yawRate = 0.0, learnedSpeed = 0.4)
        assertEquals(StationaryDetector.Source.MODEL, s)
    }
}

class VibrationEnergyTest {

    @Test
    fun `still phone reads far lower than a shaken one`() {
        val still = VibrationEnergy()
        repeat(2000) { still.update(9.81) }

        val driving = VibrationEnergy()
        var t = 0.0
        repeat(2000) {
            t += 0.0025                                   // 400 Hz
            driving.update(9.81 + 0.9 * sin(2 * Math.PI * 30 * t))   // 30 Hz road buzz
        }

        assertTrue(still.value < 0.02, "still phone read ${still.value}")
        assertTrue(driving.value > still.value * 10,
            "driving ${driving.value} not separable from still ${still.value}")
    }

    @Test
    fun `threshold used in production separates the two cases`() {
        val threshold = 0.06   // StationaryDetector default
        val still = VibrationEnergy()
        repeat(2000) { still.update(9.81 + (if (it % 2 == 0) 0.004 else -0.004)) }
        assertTrue(still.value < threshold,
            "a still phone would be misread as moving: ${still.value}")
    }
}
