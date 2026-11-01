package idr

import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Reproduces a bug reported from the phone: after pressing START, with the device
 * completely still, the displayed speed climbs steadily for roughly 50 seconds.
 *
 * This simulates exactly what NavEngine.step() does, so the cause can be isolated
 * without a device in the loop.
 */
class StationaryStartupTest {

    /** One second of a phone lying still, at the 10 Hz decimated rate. */
    private val dt = 0.1
    private val tiltOffset = 0.17     // ~1 degree of residual tilt, in m/s^2
    private val steps = 600           // 60 seconds

    private fun run(
        useBias: Boolean,
        feedModel: Boolean,
        fireZupt: Boolean,
        modelSpeed: Double = 7.1,     // what VelNet actually reported on stationary/walking data
        primeFromStillness: Boolean = false,
        modelFightsZupt: Boolean = true,
    ): DoubleArray {
        val filter = SpeedFilter()
        filter.reset(0.0)
        val bias = ForwardBias()
        val trace = DoubleArray(steps)

        repeat(steps) { i ->
            val aFwd = if (useBias) bias.correct(tiltOffset, dt) else tiltOffset
            filter.predict(aFwd, dt)

            // Shipped behaviour: the model is only trusted when nothing says we
            // are stopped, so it cannot drag the estimate away from zero.
            val modelReady = feedModel && i >= 100
            if (modelReady && (modelFightsZupt || !fireZupt)) {
                filter.updateLearned(modelSpeed, 1.5)
            }
            if (fireZupt) {
                filter.updateZupt()
                if (primeFromStillness) bias.observeStationary(tiltOffset, dt)
            }
            trace[i] = filter.speed
        }
        return trace
    }

    @Test
    fun `diagnose which input drives the phantom speed`() {
        val cases = listOf(
            "raw tilt only, no correction, no zupt" to run(false, false, false),
            "bias-corrected, no model, no zupt" to run(true, false, false),
            "bias-corrected + zupt, no model" to run(true, false, true),
            "bias-corrected + zupt + MODEL" to run(true, true, true),
            "everything on, model reports 25 m/s" to run(true, true, true, 25.0),
            "SHIPPED: priming + zupt gates model" to
                run(true, true, true, 25.0, primeFromStillness = true, modelFightsZupt = false),
        )
        println("\n%-42s %8s %8s %8s".format("case", "@10s", "@30s", "@60s"))
        println("-".repeat(70))
        for ((name, t) in cases) {
            println("%-42s %8.2f %8.2f %8.2f".format(name, t[99], t[299], t[599]))
        }
        println()
    }

    @Test
    fun `a still phone must not accumulate speed - shipped configuration`() {
        // Everything as shipped: bias primed from observed stillness, ZUPT firing,
        // and the model gated out while stopped. Even with VelNet wrongly reporting
        // 25 m/s, the estimate must stay at zero.
        val trace = run(
            useBias = true, feedModel = true, fireZupt = true, modelSpeed = 25.0,
            primeFromStillness = true, modelFightsZupt = false,
        )
        val worst = trace.max()
        assertTrue(worst < 0.2,
            "stationary phone reached ${"%.2f".format(worst)} m/s " +
                "(${"%.1f".format(worst * 3.6)} km/h) — it should stay near zero")
    }

    @Test
    fun `priming removes the startup climb`() {
        val without = run(true, false, false)                              // no stillness hint
        val with = run(true, false, true, primeFromStillness = true)       // primed
        assertTrue(without[299] > 1.0, "expected a transient without priming: ${without[299]}")
        assertTrue(with[299] < 0.1, "priming failed to remove it: ${with[299]}")
    }

    @Test
    fun `bias correction alone is not enough during its startup transient`() {
        // ForwardBias begins at zero and needs ~30 s to learn the offset. Until it
        // does, the raw tilt is integrated. This documents why ZUPT must carry the
        // first minute rather than the bias filter alone.
        val trace = run(useBias = true, feedModel = false, fireZupt = false)
        assertTrue(trace[299] > 0.5,
            "expected a startup transient to exist so the fix for it is justified; " +
                "got ${trace[299]}")
    }
}
