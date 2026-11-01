package com.navisense.app

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.GnssStatus
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock
import androidx.core.app.ActivityCompat
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/**
 * Captures raw IMU and GNSS at full rate and writes them to CSV.
 *
 * Deliberate choices, each of which bit us on the research side:
 *
 *  - UNCALIBRATED accelerometer and gyroscope. Vendors silently filter the
 *    calibrated channels, and the filtering differs per phone, so a model trained
 *    on one device's "calibrated" output does not transfer to another.
 *  - LocationManager, never FusedLocationProvider. Fused hides satellite count and
 *    C/N0, and does its own dead reckoning — which would contaminate the very thing
 *    we are trying to measure.
 *  - A foreground service AND a partial wake lock. The foreground service alone is
 *    not enough and we learned that the expensive way: it stops Android KILLING the
 *    process, but it does nothing to stop the CPU suspending. On the first real
 *    recording the accelerometer died silently at 324 s while GNSS limped on to
 *    1888 s with a 662 s gap, producing a file that looked valid and was not.
 *  - The fastest sensor rate the device will give. The speed cue is road vibration
 *    above 5 Hz; sampling at 10 Hz cannot represent it at all.
 */
class CaptureService : Service(), SensorEventListener {

    data class Stats(
        val running: Boolean = false,
        val accHz: Double = 0.0,
        val gyrHz: Double = 0.0,
        val magHz: Double = 0.0,
        val samples: Long = 0,
        val fixes: Int = 0,
        val satellites: Int = 0,
        val speedMps: Float = 0f,
        val lat: Double = 0.0,
        val lon: Double = 0.0,
        val accuracyM: Float = 0f,
        val file: String = "",
        val elapsedS: Double = 0.0,
        val deadStreams: List<String> = emptyList(),
        /** Set when recording was stopped because a required sensor never started. */
        val fatalError: String? = null,
        val visibleSats: Int = 0,
        /** Peak GNSS speed so far — the training label, so this decides usability. */
        val peakSpeedKmh: Double = 0.0,
        /** Fraction of fixes above 2 m/s. A ride spent parked teaches nothing. */
        val movingFraction: Double = 0.0,
        // Latest raw readings, so the numbers can be sanity-checked on the spot.
        val accXyz: FloatArray = FloatArray(3),
        val gyrXyz: FloatArray = FloatArray(3),
        val magXyz: FloatArray = FloatArray(3),
        // live navigation state
        val navReady: Boolean = false,
        val modelLoaded: Boolean = false,
        val gnssEnabled: Boolean = true,
        val navSpeed: Double = 0.0,
        val velNetSpeed: Double = 0.0,
        val navEast: Double = 0.0,
        val navNorth: Double = 0.0,
        val truthEast: Double = 0.0,
        val truthNorth: Double = 0.0,
        val driftM: Double = 0.0,
        val blindS: Double = 0.0,
        val navDistance: Double = 0.0,
        val calibrated: Boolean = false,
        val trail: List<FloatArray> = emptyList(),
        val truthTrail: List<FloatArray> = emptyList(),
    )

    companion object {
        private val _stats = MutableStateFlow(Stats())
        val stats: StateFlow<Stats> = _stats
        const val CHANNEL_ID = "idr_capture"

        /** Live handle so the UI can flip GPS-KILL on the running service. */
        private var instance: CaptureService? = null
        fun setGnss(on: Boolean) { instance?.setGnssEnabled(on) }
    }

    private lateinit var sensors: SensorManager
    private var nav: NavEngine? = null
    private val trail = ArrayList<FloatArray>(4096)
    private val truthTrail = ArrayList<FloatArray>(2048)
    private var writer: java.io.BufferedWriter? = null
    /** Sensor and GNSS callbacks are on different threads and share the writer.
     *  Measured on real captures: 2 torn rows in 538,694 without this lock. Rare,
     *  but it grows with recording length and can corrupt IMU rows too. */
    private val writeLock = Any()
    private var startedAtNanos = 0L
    private var wakeLock: PowerManager.WakeLock? = null

    private val accCount = AtomicLong()
    private val gyrCount = AtomicLong()
    private val magCount = AtomicLong()
    private var fixCount = 0
    private var satCount = 0
    private var peakSpeed = 0.0
    private var movingFixes = 0
    private var visibleSats = 0

    private val probeHandler = android.os.Handler(android.os.Looper.getMainLooper())
    /** Sensor types that produced nothing at any rate — surfaced in the UI. */
    private val deadStreams = java.util.Collections.synchronizedSet(mutableSetOf<Int>())

    // Latest values per stream. Sensors arrive at different rates, so each CSV row
    // records which stream triggered it and carries the most recent of the others.
    private val acc = FloatArray(3)
    private val gyr = FloatArray(3)
    private val mag = FloatArray(3)

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        instance = this
        sensors = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        nav = NavEngine(this)
        createChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground(1, buildNotification("Recording IMU + GNSS"))
        acquireWakeLock()
        peakSpeed = 0.0; movingFixes = 0; fixCount = 0
        openFile()

        // Set the epoch BEFORE registering anything. Sensor events can arrive on
        // another thread the instant registration returns, and if startedAtNanos is
        // still zero those first samples get stamped with nanoseconds-since-boot.
        // The symptom is mild — the first recording opened at t = -0.06 s — but the
        // same race can put a whole burst of samples far outside the timeline.
        startedAtNanos = SystemClock.elapsedRealtimeNanos()

        registerSensors()
        registerGnss()
        _stats.value = Stats(running = true, file = currentFile?.name ?: "")
        return START_STICKY
    }

    /**
     * Keep the CPU alive for the whole recording.
     *
     * Sensor delivery stops when the SoC suspends, and a screen-off phone in a
     * cradle suspends within seconds. Nothing reports an error — the callbacks
     * simply stop arriving — so the failure is invisible until the data is
     * analysed. GNSS survives longer because incoming fixes briefly wake the CPU,
     * which is why a corrupted file still looks superficially plausible: it has
     * position data spanning the whole drive and IMU for only the first few minutes.
     */
    private fun acquireWakeLock() {
        if (wakeLock?.isHeld == true) return
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "idr:capture").apply {
            setReferenceCounted(false)
            acquire(4 * 60 * 60 * 1000L)  // hard cap so a crashed service cannot flatten the battery
        }
    }

    private fun releaseWakeLock() {
        runCatching { if (wakeLock?.isHeld == true) wakeLock?.release() }
        wakeLock = null
    }

    override fun onDestroy() {
        releaseWakeLock()
        probeHandler.removeCallbacksAndMessages(null)
        deadStreams.clear()
        sensors.unregisterListener(this)
        runCatching {
            (getSystemService(Context.LOCATION_SERVICE) as LocationManager)
                .removeUpdates(locationListener)
        }
        writer?.flush(); writer?.close(); writer = null
        // Try to ship the ride as soon as it ends. If there is no Wi-Fi or no
        // endpoint this is a no-op; the file stays and is retried on next launch.
        runCatching { Uploader(this).syncNow() }
        instance = null
        trail.clear(); truthTrail.clear()
        _stats.value = _stats.value.copy(running = false)
        super.onDestroy()
    }

    // ---------------------------------------------------------------- sensors ---

    private fun registerSensors() {
        // Uncalibrated first; fall back only if the device genuinely lacks them.
        fun pick(uncalibrated: Int, calibrated: Int): Sensor? =
            sensors.getDefaultSensor(uncalibrated) ?: sensors.getDefaultSensor(calibrated)

        val a = pick(Sensor.TYPE_ACCELEROMETER_UNCALIBRATED, Sensor.TYPE_ACCELEROMETER)
        val g = pick(Sensor.TYPE_GYROSCOPE_UNCALIBRATED, Sensor.TYPE_GYROSCOPE)
        val m = pick(Sensor.TYPE_MAGNETIC_FIELD_UNCALIBRATED, Sensor.TYPE_MAGNETIC_FIELD)

        // Register fast, then VERIFY DATA ACTUALLY ARRIVES.
        //
        // Checking registerListener()'s return value is not enough. Measured on a
        // Redmi Note 11 (BMI220): the uncalibrated gyro registers successfully at
        // 800 Hz — dumpsys confirms samplingPeriod=1250us — and then never emits a
        // single event. Registration returns true, the app looks healthy, and you
        // end up with a folder of gyro-less drives.
        //
        // So each stream is probed: register, wait, and if nothing arrived, drop to
        // a slower rate and try again. Heading depends on the gyro, and heading is
        // the largest term in the drift budget, so a silently dead gyro is the worst
        // failure this app can have.
        listOfNotNull(a, g, m).forEach { registerWithProbe(it, 0) }
    }

    /** Sampling periods to try, fastest first. 0 = SENSOR_DELAY_FASTEST. */
    private val rateLadder = intArrayOf(0, 2_500, 5_000, 10_000, 20_000)

    private fun registerWithProbe(sensor: Sensor, ladderIndex: Int) {
        if (ladderIndex >= rateLadder.size) {
            deadStreams.add(sensor.type)
            publish()
            abortIfCrippled(sensor)
            return
        }
        val before = countFor(sensor.type)
        runCatching { sensors.registerListener(this, sensor, rateLadder[ladderIndex], 1_000_000) }

        // 4 s, not 2. Observed on a Redmi Note 11: one session produced ZERO gyro
        // rows while the very next produced 47,046 from t=0 — so a stream can be
        // slow to come up. The probe must be generous enough not to demote a
        // healthy sensor that is merely cold, while still catching a dead one fast.
        probeHandler.postDelayed({
            if (countFor(sensor.type) == before) {
                sensors.unregisterListener(this, sensor)
                registerWithProbe(sensor, ladderIndex + 1)
            }
        }, 4_000)
    }

    /**
     * Stop recording outright if a stream we cannot navigate without is dead.
     *
     * The rate ladder exhausts in ~20 s. Before this, a dead gyro just set a flag
     * and the service happily kept writing: one capture ran 329 s and 132,222
     * accelerometer rows with ZERO gyro rows. Heading comes from the gyro, and
     * heading is the largest term in the drift budget, so that file was worthless
     * the moment it started.
     *
     * Failing at the roadside where the drive can be restarted costs a minute.
     * Discovering it during analysis costs the whole drive.
     */
    private fun abortIfCrippled(sensor: Sensor) {
        val essential = sensor.type in setOf(
            Sensor.TYPE_GYROSCOPE_UNCALIBRATED, Sensor.TYPE_GYROSCOPE,
            Sensor.TYPE_ACCELEROMETER_UNCALIBRATED, Sensor.TYPE_ACCELEROMETER,
        )
        if (!essential) return  // a dead magnetometer is survivable; we gate on it anyway

        val label = if (sensor.type == Sensor.TYPE_GYROSCOPE ||
            sensor.type == Sensor.TYPE_GYROSCOPE_UNCALIBRATED
        ) "Gyroscope" else "Accelerometer"

        _stats.value = _stats.value.copy(
            running = false,
            fatalError = "$label never delivered data. Recording stopped — " +
                "restart the app and try again before driving.",
        )
        updateNotification("STOPPED: $label dead")
        stopSelf()
    }

    private fun countFor(type: Int): Long = when (type) {
        Sensor.TYPE_ACCELEROMETER_UNCALIBRATED, Sensor.TYPE_ACCELEROMETER -> accCount.get()
        Sensor.TYPE_GYROSCOPE_UNCALIBRATED, Sensor.TYPE_GYROSCOPE -> gyrCount.get()
        else -> magCount.get()
    }

    override fun onSensorChanged(event: SensorEvent) {
        val kind = when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER_UNCALIBRATED, Sensor.TYPE_ACCELEROMETER -> {
                System.arraycopy(event.values, 0, acc, 0, 3); accCount.incrementAndGet(); "A"
            }
            Sensor.TYPE_GYROSCOPE_UNCALIBRATED, Sensor.TYPE_GYROSCOPE -> {
                System.arraycopy(event.values, 0, gyr, 0, 3); gyrCount.incrementAndGet(); "G"
            }
            Sensor.TYPE_MAGNETIC_FIELD_UNCALIBRATED, Sensor.TYPE_MAGNETIC_FIELD -> {
                System.arraycopy(event.values, 0, mag, 0, 3); magCount.incrementAndGet(); "M"
            }
            else -> return
        }

        // event.timestamp is the sensor hub's own clock in nanoseconds, which is far
        // more accurate than the time this callback happened to run.
        val t = (event.timestamp - startedAtNanos) / 1e9

        // Drive the live navigator from the same samples we are logging.
        if (kind == "A" || kind == "G") {
            nav?.pushImu(
                t,
                idr.Vec3(acc[0].toDouble(), acc[1].toDouble(), acc[2].toDouble()),
                idr.Vec3(gyr[0].toDouble(), gyr[1].toDouble(), gyr[2].toDouble()),
            )
        }

        synchronized(writeLock) { writer?.write(
            "$kind,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f\n".format(
                Locale.US, t,
                acc[0], acc[1], acc[2], gyr[0], gyr[1], gyr[2],
                mag[0], mag[1], mag[2],
            )
        ) }
        publish()
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    // ------------------------------------------------------------------- gnss ---

    private val locationListener = LocationListener { loc -> onFix(loc) }

    private val gnssCallback = object : GnssStatus.Callback() {
        override fun onSatelliteStatusChanged(status: GnssStatus) {
            satCount = (0 until status.satelliteCount).count { status.usedInFix(it) }
            // Satellites VISIBLE, not just used. Indoors this is 0 and the app looks
            // broken; outdoors it climbs well before the first fix arrives, so it
            // distinguishes "no sky view" from "still acquiring".
            visibleSats = status.satelliteCount
            _stats.value = _stats.value.copy(satellites = satCount, visibleSats = visibleSats)
        }
    }

    private fun registerGnss() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED
        ) return
        val lm = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        runCatching {
            lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, 0L, 0f, locationListener)
            lm.registerGnssStatusCallback(gnssCallback, null)
        }
    }

    private fun onFix(loc: Location) {
        fixCount++
        val t = (loc.elapsedRealtimeNanos - startedAtNanos) / 1e9
        synchronized(writeLock) { writer?.write(
            "P,%.6f,%.7f,%.7f,%.2f,%.3f,%.2f,%.2f,%d,0\n".format(
                Locale.US, t, loc.latitude, loc.longitude, loc.altitude,
                loc.speed, loc.bearing, loc.accuracy, satCount,
            )
        ) }
        if (loc.speed > peakSpeed) peakSpeed = loc.speed.toDouble()
        if (loc.speed > 2.0f) movingFixes++
        nav?.pushGnss(loc.latitude, loc.longitude, loc.speed.toDouble(), loc.bearing.toDouble())
        nav?.let { if (it.gnssEnabled) truthTrail.add(floatArrayOf(it.truthEast.toFloat(), it.truthNorth.toFloat())) }
        _stats.value = _stats.value.copy(
            fixes = fixCount, satellites = satCount, speedMps = loc.speed,
            lat = loc.latitude, lon = loc.longitude, accuracyM = loc.accuracy,
            peakSpeedKmh = peakSpeed * 3.6,
            movingFraction = movingFixes.toDouble() / fixCount.coerceAtLeast(1),
        )
    }

    // -------------------------------------------------------------- plumbing ---

    private var currentFile: File? = null

    private fun openFile() {
        val dir = getExternalFilesDir(null) ?: filesDir
        val name = "drive_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) + ".csv"
        val f = File(dir, name)
        currentFile = f
        writer = f.bufferedWriter().apply {
            write("# IDR capture v1  device=${Build.MANUFACTURER} ${Build.MODEL} sdk=${Build.VERSION.SDK_INT}\n")
            write("# rows: A/G/M = t,ax,ay,az,gx,gy,gz,mx,my,mz   P = t,lat,lon,alt,speed,bearing,acc,sats,0\n")
            write("kind,t,v1,v2,v3,v4,v5,v6,v7,v8,v9\n")
        }
    }

    private fun publish() {
        val elapsed = (SystemClock.elapsedRealtimeNanos() - startedAtNanos) / 1e9
        if (elapsed < 0.5) return
        val total = accCount.get() + gyrCount.get() + magCount.get()
        // Throttle UI updates; at 400+ Hz this callback runs far too often to
        // repaint on every sample.
        if (total % 25L != 0L) return
        nav?.let { n ->
            val last = trail.lastOrNull()
            if (last == null || hypotF(n.east - last[0], n.north - last[1]) > 1.0) {
                trail.add(floatArrayOf(n.east.toFloat(), n.north.toFloat()))
                if (trail.size > 4000) trail.removeAt(0)
            }
        }
        _stats.value = _stats.value.copy(
            running = true,
            accHz = accCount.get() / elapsed,
            gyrHz = gyrCount.get() / elapsed,
            magHz = magCount.get() / elapsed,
            samples = total,
            elapsedS = elapsed,
            accXyz = acc.copyOf(),
            gyrXyz = gyr.copyOf(),
            magXyz = mag.copyOf(),
            file = currentFile?.name ?: "",
            navReady = nav != null,
            modelLoaded = nav?.modelLoaded ?: false,
            gnssEnabled = nav?.gnssEnabled ?: true,
            navSpeed = nav?.speedMps ?: 0.0,
            velNetSpeed = nav?.velNetSpeed ?: 0.0,
            navEast = nav?.east ?: 0.0,
            navNorth = nav?.north ?: 0.0,
            truthEast = nav?.truthEast ?: 0.0,
            truthNorth = nav?.truthNorth ?: 0.0,
            driftM = nav?.driftM ?: 0.0,
            blindS = nav?.blindSeconds ?: 0.0,
            navDistance = nav?.distanceM ?: 0.0,
            calibrated = nav?.calibrated ?: false,
            trail = ArrayList(trail),
            truthTrail = ArrayList(truthTrail),
            deadStreams = deadStreams.map {
                when (it) {
                    Sensor.TYPE_ACCELEROMETER_UNCALIBRATED, Sensor.TYPE_ACCELEROMETER -> "accelerometer"
                    Sensor.TYPE_GYROSCOPE_UNCALIBRATED, Sensor.TYPE_GYROSCOPE -> "gyroscope"
                    else -> "magnetometer"
                }
            },
        )
    }

    private fun hypotF(a: Double, b: Float): Double = kotlin.math.hypot(a - b, 0.0)
    private fun hypotF(a: Double, b: Double): Double = kotlin.math.hypot(a, b)

    /** GPS-KILL, driven from the UI. */
    fun setGnssEnabled(on: Boolean) { if (on) nav?.restoreGnss() else nav?.killGnss() }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val ch = NotificationChannel(
                CHANNEL_ID, "IDR capture", NotificationManager.IMPORTANCE_LOW
            )
            (getSystemService(NotificationManager::class.java)).createNotificationChannel(ch)
        }
    }

    private fun updateNotification(text: String) {
        runCatching {
            (getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager)
                .notify(1, buildNotification(text))
        }
    }

    private fun buildNotification(text: String): Notification =
        Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("NaviSense")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setOngoing(true)
            .build()
}
