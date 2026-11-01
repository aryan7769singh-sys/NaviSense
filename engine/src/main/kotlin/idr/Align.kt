package idr

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * In-vehicle alignment: work out how the phone is lying, whatever the mount.
 *
 * Roll and pitch are direct — the low-passed accelerometer is the up axis.
 * Yaw is the hard one, and it is recovered from the fact that during straight-line
 * acceleration and braking the horizontal acceleration points along the vehicle's
 * forward axis. Accumulating those samples and taking their principal direction
 * gives the phone-to-vehicle yaw offset.
 *
 * Yaw RATE does not need any of this: rotating about the vertical axis does not
 * change the vertical component of the gyro, so heading stays available even while
 * alignment is still converging.
 *
 * Ported from proto/align.py, where the approach was validated against synthetic
 * drives with random mount poses.
 */
class Aligner(
    private val hz: Double = 100.0,
    private val gravityTau: Double = 6.0,
    private val minAccel: Double = 0.35,
    private val maxYawRate: Double = 0.06,
    private val minSpeed: Double = 2.0,
) {
    private var gravity: Vec3? = null

    // Accumulated second moments of horizontal acceleration, for the PCA.
    private var sxx = 0.0
    private var sxy = 0.0
    private var syy = 0.0
    private var signAccum = 0.0
    private var samples = 0

    var mountYaw: Double = 0.0
        private set
    var confidence: Double = 0.0
        private set
    val calibrated: Boolean get() = confidence > 0.35 && samples > 200

    /** Current levelled-frame rotation, or identity before gravity is known. */
    fun levelRotation(): Mat3 =
        gravity?.let { Mat3.align(it.normalized(), Vec3.UP) } ?: Mat3.identity()

    /** Full device -> vehicle rotation (level, then de-rotate the mount yaw). */
    fun vehicleRotation(): Mat3 = Mat3.yaw(mountYaw) * levelRotation()

    /**
     * Feed one sample. [refSpeed] and [refAccel] come from GNSS while it is
     * available and are used only to disambiguate forward from backward — never
     * during a blackout.
     */
    fun update(acc: Vec3, gyr: Vec3, refSpeed: Double, refAccel: Double) {
        // Gravity: a very slow low-pass. Manoeuvres last seconds; gravity does not.
        val alpha = (1.0 / hz) / gravityTau
        gravity = gravity?.let { it * (1 - alpha) + acc * alpha } ?: acc

        val level = levelRotation()
        val aL = level * acc
        val gL = level * gyr

        val horizontal = sqrt(aL.x * aL.x + aL.y * aL.y)
        if (horizontal < minAccel || abs(gL.z) > maxYawRate || refSpeed < minSpeed) return

        sxx += aL.x * aL.x
        sxy += aL.x * aL.y
        syy += aL.y * aL.y
        samples++

        // The principal axis is a line, not a direction. Longitudinal acceleration
        // and the change in speed must agree in sign; that picks the forward end.
        signAccum += (aL.x * refAccel)

        if (samples % 25 == 0) solve()
    }

    private fun solve() {
        val n = samples.toDouble()
        val cxx = sxx / n
        val cxy = sxy / n
        val cyy = syy / n

        // Closed-form eigen-decomposition of a symmetric 2x2.
        val tr = cxx + cyy
        val det = cxx * cyy - cxy * cxy
        val disc = sqrt(maxOf(tr * tr / 4 - det, 0.0))
        val l1 = tr / 2 + disc
        val l2 = tr / 2 - disc
        if (l1 < 1e-12) return

        var vx: Double
        var vy: Double
        if (abs(cxy) > 1e-12) { vx = l1 - cyy; vy = cxy } else { vx = 1.0; vy = 0.0 }
        val n2 = sqrt(vx * vx + vy * vy)
        if (n2 < 1e-12) return
        vx /= n2; vy /= n2

        if (signAccum < 0) { vx = -vx; vy = -vy }

        mountYaw = atan2(vy, vx)
        confidence = 1.0 - (l2 / l1)
    }

    /**
     * Detect the phone being moved, which invalidates the alignment.
     *
     * A sudden change in the gravity direction means a new mount pose, so the yaw
     * estimate has to be rebuilt rather than slowly dragged to the new value.
     */
    fun mountChanged(acc: Vec3, thresholdRad: Double = 0.35): Boolean {
        val g = gravity ?: return false
        val cosang = (g.normalized() dot acc.normalized()).coerceIn(-1.0, 1.0)
        return kotlin.math.acos(cosang) > thresholdRad
    }

    fun reset() {
        sxx = 0.0; sxy = 0.0; syy = 0.0; signAccum = 0.0; samples = 0
        confidence = 0.0; mountYaw = 0.0
    }
}
