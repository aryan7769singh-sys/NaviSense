package idr

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MathTest {
    @Test
    fun `wrapPi folds into range`() {
        assertEquals(0.0, wrapPi(2 * PI), 1e-9)
        assertEquals(-PI / 2, wrapPi(3 * PI / 2), 1e-9)
        assertTrue(wrapPi(100.0) in -PI..PI)
    }

    @Test
    fun `align rotates gravity onto vertical`() {
        val rng = Random(7)
        repeat(50) {
            val g = Vec3(rng.nextDouble(-1.0, 1.0), rng.nextDouble(-1.0, 1.0), rng.nextDouble(-1.0, 1.0))
            if (g.norm < 1e-3) return@repeat
            val r = Mat3.align(g.normalized(), Vec3.UP)
            val up = r * g.normalized()
            assertEquals(0.0, up.x, 1e-9, "x should vanish")
            assertEquals(0.0, up.y, 1e-9, "y should vanish")
            assertEquals(1.0, up.z, 1e-9, "z should be +1")
        }
    }

    @Test
    fun `align preserves vector length`() {
        val r = Mat3.align(Vec3(0.3, -0.5, 0.8).normalized(), Vec3.UP)
        val v = Vec3(1.0, 2.0, -3.0)
        assertEquals(v.norm, (r * v).norm, 1e-9)
    }

    @Test
    fun `yaw rotation composes as expected`() {
        val v = Vec3(1.0, 0.0, 0.0)
        val r = Mat3.yaw(PI / 2) * v
        assertEquals(0.0, r.x, 1e-9)
        assertEquals(-1.0, r.y, 1e-9)
    }
}

class SpeedFilterTest {
    @Test
    fun `converges to a constant observation`() {
        val f = SpeedFilter()
        f.reset(0.0)
        repeat(200) {
            f.predict(0.0, 0.1)
            f.updateLearned(12.0, 1.0)
        }
        assertEquals(12.0, f.speed, 0.5, "should track the learned observation")
    }

    @Test
    fun `estimates accelerometer bias and stops the ramp`() {
        // A constant 0.2 m/s^2 bias would integrate to 20 m/s over 100 s if the
        // filter did not identify it. Truth here is a steady 10 m/s.
        val f = SpeedFilter()
        f.reset(10.0)
        repeat(1000) {
            f.predict(0.2, 0.1)
            f.updateLearned(10.0, 1.0)
        }
        assertTrue(abs(f.speed - 10.0) < 1.5, "speed drifted to ${f.speed}")
        assertTrue(f.accelBias > 0.05, "bias not identified: ${f.accelBias}")
    }

    @Test
    fun `zupt pulls speed to zero`() {
        val f = SpeedFilter()
        f.reset(8.0)
        repeat(50) { f.predict(0.0, 0.1); f.updateZupt() }
        assertTrue(f.speed < 1.0, "zupt left speed at ${f.speed}")
    }

    @Test
    fun `speed never goes negative`() {
        val f = SpeedFilter()
        f.reset(1.0)
        repeat(100) { f.predict(-5.0, 0.1) }
        assertTrue(f.speed >= 0.0)
    }
}

class DeadReckonerTest {
    @Test
    fun `straight line integrates to speed times time`() {
        val dr = DeadReckoner(0.0, 0.0, 0.0) // heading 0 = due East
        repeat(100) { dr.step(10.0, 0.0, 0.1) }
        assertEquals(100.0, dr.east, 1e-6)
        assertEquals(0.0, dr.north, 1e-6)
        assertEquals(100.0, dr.distance, 1e-6)
    }

    @Test
    fun `constant yaw rate traces a circle of the right radius`() {
        // r = v / omega = 10 / 0.1 = 100 m; a full lap returns to the start.
        val v = 10.0; val omega = 0.1
        val dr = DeadReckoner(0.0, 0.0, 0.0)
        val steps = (2 * PI / omega / 0.01).toInt()
        repeat(steps) { dr.step(v, omega, 0.01) }
        assertTrue(hypot(dr.east, dr.north) < 1.0, "did not close the loop: ${dr.east}, ${dr.north}")
    }

    @Test
    fun `heading nudge moves toward the reference`() {
        val dr = DeadReckoner(0.0, 0.0, 0.0)
        dr.nudgeHeading(1.0, gain = 0.5)
        assertEquals(0.5, dr.heading, 1e-9)
    }

    @Test
    fun `heading nudge takes the short way around the seam`() {
        val dr = DeadReckoner(0.0, 0.0, 3.0)
        dr.nudgeHeading(-3.0, gain = 1.0)
        assertEquals(-3.0, dr.heading, 1e-6, "should cross the +-pi seam, not travel the long way")
    }
}

class AlignerTest {
    @Test
    fun `recovers a known mount yaw from braking events`() {
        // Phone rotated 40 degrees about vertical relative to the vehicle. Feed
        // pure longitudinal acceleration and check the estimate finds it.
        val trueYaw = Math.toRadians(40.0)
        val aligner = Aligner(hz = 100.0)
        val rng = Random(3)
        var speed = 15.0

        repeat(4000) {
            val aFwd = if (it % 400 < 200) 1.2 else -1.2
            speed = (speed + aFwd * 0.01).coerceIn(5.0, 30.0)
            // Vehicle-frame specific force -> device frame (rotate by -trueYaw).
            val ax = aFwd * cos(-trueYaw) - 0.0 * sin(-trueYaw)
            val ay = aFwd * sin(-trueYaw) + 0.0 * cos(-trueYaw)
            val noise = rng.nextDouble(-0.02, 0.02)
            aligner.update(
                Vec3(ax + noise, ay + noise, 9.81),
                Vec3(0.0, 0.0, 0.0),
                refSpeed = speed,
                refAccel = aFwd,
            )
        }
        // Assert the property that matters rather than the internal sign convention:
        // applying vehicleRotation() to a device-frame forward acceleration must put
        // it back on the vehicle's +x (forward) axis.
        val deviceForward = Vec3(1.2 * cos(-trueYaw), 1.2 * sin(-trueYaw), 9.81)
        val inVehicle = aligner.vehicleRotation() * deviceForward
        val recovered = Math.toDegrees(abs(wrapPi(kotlin.math.atan2(inVehicle.y, inVehicle.x))))
        assertTrue(
            recovered < 5.0,
            "forward acceleration lands $recovered deg off the vehicle x-axis " +
                "(mountYaw=${Math.toDegrees(aligner.mountYaw)})",
        )
        assertTrue(aligner.confidence > 0.5, "confidence too low: ${aligner.confidence}")
    }

    @Test
    fun `levelling removes tilt regardless of mount`() {
        val aligner = Aligner(hz = 100.0)
        // Phone tilted: gravity is not along device z.
        val g = Vec3(2.0, -3.0, 9.0)
        repeat(2000) { aligner.update(g, Vec3.UP * 0.0, refSpeed = 0.0, refAccel = 0.0) }
        val levelled = aligner.levelRotation() * g
        assertTrue(abs(levelled.x) < 1e-6 && abs(levelled.y) < 1e-6, "still tilted: $levelled")
        assertEquals(g.norm, levelled.z, 1e-6)
    }
}
