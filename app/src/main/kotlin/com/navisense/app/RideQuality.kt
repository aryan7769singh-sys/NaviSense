package com.navisense.app

/**
 * Live verdict on whether the ride being recorded will be usable for training.
 *
 * This is the Python `audit()` moved onto the phone, and it exists because of a
 * concrete failure: three recordings were made, all looked fine on the screen, and
 * all three turned out to be worthless — one truncated at 324 s, one with no gyro
 * at all, one that never exceeded 14 km/h. That was only discovered hours later on
 * a laptop, by which point the vehicle was gone.
 *
 * A logger that cannot tell you it is failing is not a logger, it is a
 * random-file generator. The verdict has to be visible while there is still time
 * to fix it.
 */
object RideQuality {

    enum class Level { GOOD, WARN, BAD }

    data class Verdict(val level: Level, val headline: String, val detail: String)

    /** Below this the GNSS speed label is unusable, so the ride teaches nothing. */
    private const val MIN_PEAK_KMH = 15.0
    private const val MIN_IMU_HZ = 90.0
    private const val MAX_GNSS_ACCURACY_M = 15.0

    fun assess(s: Stats): Verdict {
        // --- fatal: nothing recoverable -----------------------------------------
        if (s.deadStreams.isNotEmpty()) {
            return Verdict(
                Level.BAD,
                "${s.deadStreams.joinToString()} not recording",
                "This ride cannot be used. Stop, restart the app, and check the rate cards.",
            )
        }
        if (s.elapsedS > 25 && s.gyrHz < 1.0) {
            return Verdict(
                Level.BAD, "Gyro is silent",
                "Heading is impossible without it. Stop and restart before driving further.",
            )
        }
        if (s.elapsedS > 25 && s.accHz < MIN_IMU_HZ) {
            return Verdict(
                Level.BAD, "IMU only ${s.accHz.toInt()} Hz",
                "Below the ${MIN_IMU_HZ.toInt()} Hz the speed model needs. The vibration cue lives above 5 Hz.",
            )
        }

        // --- waiting -------------------------------------------------------------
        if (s.fixes == 0) {
            return Verdict(
                Level.WARN,
                if (s.visibleSats == 0) "No satellites" else "Acquiring GNSS",
                if (s.visibleSats == 0) "GPS needs sky view. Nothing is being labelled yet."
                else "${s.visibleSats} satellites visible. Wait for a fix before driving off.",
            )
        }
        if (s.elapsedS < 60) {
            return Verdict(Level.WARN, "Warming up", "Keep driving — a verdict needs about a minute.")
        }

        // --- the label problem ---------------------------------------------------
        // GNSS Doppler speed IS the training label. If it stays near zero the ride
        // produces no supervision at all, however clean the IMU looks.
        val peakKmh = s.peakSpeedKmh
        if (peakKmh < MIN_PEAK_KMH) {
            return Verdict(
                Level.BAD, "Peak only ${peakKmh.toInt()} km/h",
                "GNSS speed is the training label and it is near zero. Walking cannot train a car model.",
            )
        }
        if (s.movingFraction < 0.3) {
            return Verdict(
                Level.WARN, "Mostly stationary",
                "Only ${(100 * s.movingFraction).toInt()}% of fixes are above 2 m/s. Keep moving.",
            )
        }
        if (s.accuracyM > MAX_GNSS_ACCURACY_M) {
            return Verdict(
                Level.WARN, "GNSS +/-${s.accuracyM.toInt()} m",
                "Poor sky view weakens the labels. Open road is better than a built-up street.",
            )
        }

        val mins = (s.elapsedS / 60).toInt()
        return Verdict(
            Level.GOOD,
            "Good ride — $mins min, peak ${peakKmh.toInt()} km/h",
            "IMU ${s.accHz.toInt()}/${s.gyrHz.toInt()} Hz, GNSS +/-${s.accuracyM.toInt()} m. Keep going.",
        )
    }
}

/** Alias so the object above reads cleanly against the service's Stats type. */
private typealias Stats = CaptureService.Stats
