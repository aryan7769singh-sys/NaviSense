package idr

import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/** One IMU sample in the device frame. Timestamps are seconds. */
data class ImuSample(
    val t: Double,
    val ax: Double, val ay: Double, val az: Double,
    val gx: Double, val gy: Double, val gz: Double,
)

/** One GNSS fix. `speed` is m/s, `course` is radians clockwise from North. */
data class GnssFix(
    val t: Double,
    val lat: Double,
    val lon: Double,
    val speed: Double,
    val course: Double,
    val accuracyM: Double = 10.0,
    val satellites: Int = 0,
)

/** GNSS health. The filter never restarts; only this changes. */
enum class GnssQuality { ACQUIRED, DEGRADED, LOST }

/** What the UI draws, at 10 Hz. */
data class NavState(
    val t: Double,
    val lat: Double,
    val lon: Double,
    val east: Double,
    val north: Double,
    val speed: Double,
    val heading: Double,
    val quality: GnssQuality,
    val distanceM: Double,
    val driftM: Double,
    val calibrated: Boolean,
)

/** Minimal 3-vector. Avoids pulling a linear-algebra dependency into the hot path. */
data class Vec3(val x: Double, val y: Double, val z: Double) {
    operator fun plus(o: Vec3) = Vec3(x + o.x, y + o.y, z + o.z)
    operator fun minus(o: Vec3) = Vec3(x - o.x, y - o.y, z - o.z)
    operator fun times(s: Double) = Vec3(x * s, y * s, z * s)
    infix fun dot(o: Vec3) = x * o.x + y * o.y + z * o.z
    infix fun cross(o: Vec3) = Vec3(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x)
    val norm: Double get() = sqrt(x * x + y * y + z * z)
    fun normalized(): Vec3 {
        val n = norm
        return if (n < 1e-9) Vec3(0.0, 0.0, 1.0) else Vec3(x / n, y / n, z / n)
    }
    companion object { val UP = Vec3(0.0, 0.0, 1.0) }
}

/** Row-major 3x3 matrix. */
class Mat3(private val m: DoubleArray) {
    init { require(m.size == 9) { "Mat3 needs 9 elements" } }

    operator fun get(r: Int, c: Int) = m[r * 3 + c]

    operator fun times(v: Vec3) = Vec3(
        m[0] * v.x + m[1] * v.y + m[2] * v.z,
        m[3] * v.x + m[4] * v.y + m[5] * v.z,
        m[6] * v.x + m[7] * v.y + m[8] * v.z,
    )

    operator fun times(o: Mat3): Mat3 {
        val r = DoubleArray(9)
        for (i in 0..2) for (j in 0..2) {
            var s = 0.0
            for (k in 0..2) s += this[i, k] * o[k, j]
            r[i * 3 + j] = s
        }
        return Mat3(r)
    }

    companion object {
        fun identity() = Mat3(doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

        /** Rotation about the vertical axis by [psi] radians. */
        fun yaw(psi: Double): Mat3 {
            val c = cos(psi); val s = sin(psi)
            return Mat3(doubleArrayOf(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0))
        }

        /**
         * Rotation taking unit vector [from] onto unit vector [to] (Rodrigues).
         *
         * Used to level the device frame: the low-passed accelerometer IS the up
         * axis, so rotating it onto +Z removes roll and pitch in one step.
         */
        fun align(from: Vec3, to: Vec3): Mat3 {
            val a = from.normalized(); val b = to.normalized()
            val axis = a cross b
            val s = axis.norm
            val c = a dot b
            if (s < 1e-8) {
                return if (c > 0) identity()
                else Mat3(doubleArrayOf(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0))
            }
            // Rodrigues with a UNIT axis needs both trig factors:
            //     R = I + sin(theta) [k]x + (1 - cos(theta)) [k]x^2
            // where sin(theta) = |a x b| and cos(theta) = a . b.
            // Dropping the sin(theta) on the first-order term (or folding an s^2
            // into the second) produces a matrix that is not orthonormal — it
            // rotates approximately but does not preserve length.
            val k = axis * (1.0 / s)
            val kx = Mat3(doubleArrayOf(0.0, -k.z, k.y, k.z, 0.0, -k.x, -k.y, k.x, 0.0))
            val kk = kx * kx
            val r = DoubleArray(9)
            for (i in 0..2) for (j in 0..2) {
                r[i * 3 + j] = (if (i == j) 1.0 else 0.0) + s * kx[i, j] + (1.0 - c) * kk[i, j]
            }
            return Mat3(r)
        }
    }
}

const val EARTH_RADIUS_M = 6_371_000.0

/** Wrap an angle into [-pi, pi). */
fun wrapPi(a: Double): Double {
    var x = (a + PI) % (2 * PI)
    if (x < 0) x += 2 * PI
    return x - PI
}

/** GPS course (0 = North, clockwise) to ENU yaw (0 = East, counter-clockwise). */
fun courseToEnu(course: Double) = PI / 2.0 - course

/** Bearing of an ENU displacement. */
fun bearingOf(east: Double, north: Double) = atan2(north, east)


/**
 * Bound a timestep taken from sensor timestamps.
 *
 * Sensor streams gap: CPU stalls, batching hiccups, dropped bursts. Real captures
 * from this project contain multi-second IMU gaps. Feeding such a gap straight into
 * an integrator multiplies it by speed and teleports the position in a single step,
 * so a bound is not optional.
 *
 * The cap is 3x the nominal 10 Hz step: ordinary jitter passes through untouched,
 * while a multi-second gap is truncated. Note this UNDER-estimates travel across a
 * gap rather than over-estimating it, which is the safer direction — an undershoot
 * degrades gracefully, a teleport does not.
 */
fun clampDt(dt: Double, max: Double = 0.3): Double = dt.coerceIn(0.0, max)
