package com.navisense.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import idr.Aligner
import idr.ForwardBias
import idr.clampDt
import idr.StationaryDetector
import idr.VibrationEnergy
import idr.DeadReckoner
import idr.Mat3
import idr.SpeedFilter
import idr.Vec3
import idr.courseToEnu
import idr.wrapPi
import java.nio.FloatBuffer
import org.json.JSONObject
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sqrt

/**
 * The live navigator: raw phone sensors in, position out, no GNSS required.
 *
 * Mirrors the Python research pipeline stage for stage, because any divergence
 * between them is a bug nobody would ever see:
 *
 *   phone IMU (400 Hz)
 *     -> decimate to 10 Hz          (boxcar average = crude anti-alias)
 *     -> align to the vehicle frame (gravity for roll/pitch, accel-PCA for yaw)
 *     -> VelNet over a 10 s window  (speed from road vibration alone)
 *     -> speed Kalman filter        (network + accelerometer + zero-velocity)
 *     -> dead reckoning             (speed + gyro heading)
 *
 * The filter runs continuously whether or not GNSS is present. There is no mode
 * switch, so "handoff" is not an event — losing GNSS just stops one measurement
 * from arriving. That is what makes it instant.
 */
class NavEngine(context: Context) {

    companion object {
        const val TARGET_HZ = 10.0
        private const val EARTH_R = 6_371_000.0
    }

    // --- model ----------------------------------------------------------------
    private val env = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private var window = 100
    private var channels = 8
    private lateinit var normMean: FloatArray
    private lateinit var normStd: FloatArray
    var modelLoaded = false
        private set
    var lastError: String? = null
        private set

    // --- pipeline state -------------------------------------------------------
    private val aligner = Aligner(hz = 400.0)
    private val speedFilter = SpeedFilter()
    private val dr = DeadReckoner()

    private var accSum = Vec3(0.0, 0.0, 0.0)
    private var gyrSum = Vec3(0.0, 0.0, 0.0)
    private var decimCount = 0
    private var lastDecimT = 0.0

    /** Ring buffer of decimated vehicle-frame samples: [acc(3), gyr(3)]. */
    private val ring = Array(200) { FloatArray(6) }
    private var ringLen = 0
    private var ringHead = 0

    private var gyroBiasZ = 0.0
    // These live in the engine module, not here, because they are pure numeric logic
    // with no business depending on android.content.Context — and because there they
    // are unit tested (see MotionGateTest).
    private val fwdBias = ForwardBias()
    private val vibration = VibrationEnergy()
    private val stationary = StationaryDetector()
    private var lastGnssHeading = 0.0
    private var lastGnssSpeed = 0.0
    private var prevGnssSpeed = 0.0
    private var prevGnssT = 0.0
    /** d(GNSS speed)/dt — the reference the aligner needs to tell forward from back. */
    private var gnssAccel = 0.0
    private var haveOrigin = false
    private var lat0 = 0.0
    private var lon0 = 0.0

    var gnssEnabled = true          // the GPS-KILL switch flips this
    var speedMps = 0.0; private set
    var headingRad = 0.0; private set
    var east = 0.0; private set
    var north = 0.0; private set
    var truthEast = 0.0; private set
    var truthNorth = 0.0; private set
    var distanceM = 0.0; private set
    var blindSeconds = 0.0; private set
    var velNetSpeed = 0.0; private set
    var calibrated = false; private set
    var stationarySource = "NONE"; private set

    val driftM: Double get() = hypot(east - truthEast, north - truthNorth)

    init {
        runCatching {
            val norm = JSONObject(context.assets.open("velnet_norm.json").bufferedReader().readText())
            window = norm.getInt("window")
            channels = norm.getInt("channels")
            normMean = FloatArray(channels) { norm.getJSONArray("mean").getDouble(it).toFloat() }
            normStd = FloatArray(channels) { norm.getJSONArray("std").getDouble(it).toFloat() }
            val bytes = context.assets.open("velnet.onnx").readBytes()
            session = env.createSession(bytes, OrtSession.SessionOptions())
            modelLoaded = true
        }.onFailure { lastError = "${it::class.simpleName}: ${it.message}" }
    }

    // --- sensor ingest --------------------------------------------------------

    /** Feed one raw IMU sample at the device's native rate. */
    fun pushImu(t: Double, acc: Vec3, gyr: Vec3) {
        // refAccel was hard-coded to 0.0, which silently disabled the aligner's sign
        // disambiguation: it could settle on an axis pointing BACKWARDS and nothing
        // would flag it. It needs the actual rate of change of speed.
        aligner.update(acc, gyr, lastGnssSpeed, gnssAccel)
        calibrated = aligner.calibrated

        // Decimate 400 Hz -> 10 Hz by averaging. Averaging IS a low-pass, which
        // matters: engine vibration at 20-100 Hz would otherwise alias straight
        // into the band the model reads.
        // High-frequency vibration energy, measured on the RAW 400 Hz stream before
        // decimation removes it. At 10 Hz this was impossible — the band that
        // separates "stopped" from "rolling slowly" sits above the Nyquist limit.
        // At 400 Hz it is available, which is what lets a stationary detector work
        // from physics instead of from the model's opinion.
        vibration.update(kotlin.math.sqrt(acc.x * acc.x + acc.y * acc.y + acc.z * acc.z))

        accSum += acc; gyrSum += gyr; decimCount++
        if (t - lastDecimT < 1.0 / TARGET_HZ) return

        val n = decimCount.coerceAtLeast(1).toDouble()
        val rot = aligner.vehicleRotation()
        val a = rot * (accSum * (1.0 / n))
        val g = rot * (gyrSum * (1.0 / n))
        accSum = Vec3(0.0, 0.0, 0.0); gyrSum = Vec3(0.0, 0.0, 0.0); decimCount = 0
        // Bound the step: real captures contain multi-second IMU gaps, and feeding
        // one straight into the integrator teleports the position.
        val dt = if (lastDecimT == 0.0) 1.0 / TARGET_HZ else clampDt(t - lastDecimT)
        lastDecimT = t

        ring[ringHead] = floatArrayOf(
            a.x.toFloat(), a.y.toFloat(), a.z.toFloat(),
            g.x.toFloat(), g.y.toFloat(), g.z.toFloat(),
        )
        ringHead = (ringHead + 1) % ring.size
        if (ringLen < ring.size) ringLen++

        step(a, g, dt)
    }

    private fun step(a: Vec3, g: Vec3, dt: Double) {
        velNetSpeed = runModel() ?: velNetSpeed

        // Remove the static component from forward specific force. This was
        // described in a comment but never actually done, and it is the reason the
        // dot crept along with the phone sitting still: any tilt error leaves a slice
        // of gravity in the forward axis, and one degree is 0.17 m/s^2 — which
        // integrates to 10 m/s of invented speed in a minute.
        //
        // The reference updates slowly (30 s) so real accelerations, which last
        // seconds, pass through untouched while the constant offset is tracked out.
        val aFwd = fwdBias.correct(a.x, dt)

        speedFilter.predict(aFwd, dt)

        // --- zero-velocity updates, from three independent sources -------------
        //
        // Gating this on the model alone was circular: the correction that fixes a
        // wrong speed estimate only fired when the model already agreed, so it was
        // disabled exactly when it was needed. Any one of these is now enough.
        val source = stationary.assess(
            gnssSpeed = if (gnssEnabled && haveOrigin) lastGnssSpeed else null,
            hfEnergy = vibration.value,
            yawRate = g.z,
            learnedSpeed = velNetSpeed,
        )
        if (source != StationaryDetector.Source.NONE) {
            speedFilter.updateZupt()
            // A known stop is the only moment forward acceleration is certainly
            // zero, so every stationary sample is used to pin down the tilt bias —
            // not just GNSS-confirmed ones. This is what removes the startup climb:
            // a phone sitting still before you drive off primes the estimate in
            // about a second instead of thirty.
            fwdBias.observeStationary(a.x, dt)
        } else if (velNetSpeed > 0) {
            // Only trust the model when nothing says we are stopped. Feeding it
            // unconditionally let it fight the zero-velocity update: with VelNet
            // wrongly reporting 25 m/s on a parked vehicle, the estimate settled at
            // 2.5 m/s instead of zero.
            speedFilter.updateLearned(velNetSpeed, 1.5)
        }
        stationarySource = source.name

        speedMps = speedFilter.speed
        dr.step(speedMps, g.z - gyroBiasZ, dt)
        east = dr.east; north = dr.north; headingRad = dr.heading
        distanceM = dr.distance
        if (!gnssEnabled) blindSeconds += dt
    }

    private fun runModel(): Double? {
        val s = session ?: return null
        if (ringLen < window) return null

        // Build the window oldest-to-newest, matching trip_features() exactly.
        val w = window
        val buf = FloatArray(w * channels)
        val tmp = Array(w) { FloatArray(8) }
        for (i in 0 until w) {
            val idx = ((ringHead - w + i) % ring.size + ring.size) % ring.size
            val r = ring[idx]
            val amag = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
            val gmag = sqrt(r[3] * r[3] + r[4] * r[4] + r[5] * r[5])
            tmp[i] = floatArrayOf(r[0], r[1], r[2], r[3], r[4], r[5], amag, gmag)
        }
        // Per-window mean removal on the accelerometer channels and |acc|: their DC
        // is gravity plus sensor bias, i.e. it identifies the device rather than the
        // speed. Leaving it in lets the model memorise recordings.
        val mx = (0 until w).sumOf { tmp[it][0].toDouble() } / w
        val my = (0 until w).sumOf { tmp[it][1].toDouble() } / w
        val mz = (0 until w).sumOf { tmp[it][2].toDouble() } / w
        val mm = (0 until w).sumOf { tmp[it][6].toDouble() } / w
        for (i in 0 until w) {
            tmp[i][0] = (tmp[i][0] - mx).toFloat()
            tmp[i][1] = (tmp[i][1] - my).toFloat()
            tmp[i][2] = (tmp[i][2] - mz).toFloat()
            tmp[i][6] = (tmp[i][6] - mm).toFloat()
        }
        // (1, C, W) with normalisation applied, matching the training transform.
        for (c in 0 until channels) for (i in 0 until w) {
            buf[c * w + i] = (tmp[i][c] - normMean[c]) / normStd[c]
        }

        return runCatching {
            OnnxTensor.createTensor(env, FloatBuffer.wrap(buf), longArrayOf(1, channels.toLong(), w.toLong()))
                .use { t ->
                    s.run(mapOf("imu" to t)).use { r ->
                        // Output is 1-D, shape (batch,) — NOT (batch, 1). Casting to
                        // Array<FloatArray> compiles fine and throws at runtime on
                        // every single inference, which presents as "model silently
                        // does nothing" rather than as an error.
                        when (val v = r[0].value) {
                            is FloatArray -> v[0].toDouble()
                            is Array<*> -> ((v[0] as FloatArray)[0]).toDouble()
                            else -> error("unexpected output type ${'$'}{v?.javaClass}")
                        }
                    }
                }
        }.onFailure { lastError = it.message }.getOrNull()
    }

    // --- gnss -----------------------------------------------------------------

    fun pushGnss(lat: Double, lon: Double, speed: Double, bearingDeg: Double) {
        if (!haveOrigin) {
            lat0 = lat; lon0 = lon; haveOrigin = true
            dr.reset(0.0, 0.0, courseToEnu(Math.toRadians(bearingDeg)))
            speedFilter.reset(speed.toDouble())
        }
        val cosLat = cos(Math.toRadians(lat0))
        truthEast = Math.toRadians(lon - lon0) * EARTH_R * cosLat
        truthNorth = Math.toRadians(lat - lat0) * EARTH_R

        val now = android.os.SystemClock.elapsedRealtimeNanos() / 1e9
        if (prevGnssT > 0 && now > prevGnssT) {
            gnssAccel = (speed - prevGnssSpeed) / (now - prevGnssT)
        }
        prevGnssSpeed = speed; prevGnssT = now
        lastGnssSpeed = speed
        lastGnssHeading = courseToEnu(Math.toRadians(bearingDeg))

        if (!gnssEnabled) return   // GPS-KILL: measurement withheld, filter keeps running

        // Heading, not speed, is the dominant term in the drift budget, so estimate
        // the gyro's vertical bias against GNSS course while the sky is visible and
        // freeze it through the blackout — exactly what an error-state filter does.
        if (speed > 4.0 && prevGnssT > 0) {
            // Heading error is an ANGLE; gyro bias is a RATE. Dividing by the
            // interval it accumulated over is what makes this dimensionally valid —
            // without it the correction silently scaled with GNSS update rate.
            val err = wrapPi(lastGnssHeading - dr.heading)
            val interval = (now - prevGnssT).coerceIn(0.2, 5.0)
            gyroBiasZ += 0.15 * err / interval
            gyroBiasZ = gyroBiasZ.coerceIn(-0.2, 0.2)   // a real gyro bias is small
            dr.nudgeHeading(lastGnssHeading, gain = 0.25)
        }
        speedFilter.updateLearned(speed.toDouble(), 0.4)
        // snapTo, not reset: this is a position correction, not a new journey.
        // reset() zeroes trip distance, and being called on every fix meant DISTANCE
        // never exceeded one second of travel.
        dr.snapTo(truthEast, truthNorth)
        blindSeconds = 0.0
    }

    fun killGnss() { gnssEnabled = false; blindSeconds = 0.0; distanceM = 0.0; dr.reset(east, north, headingRad) }
    fun restoreGnss() { gnssEnabled = true }
}
