package com.navisense.app

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.PowerManager
import android.provider.Settings

/**
 * Keeps the recording alive across manufacturers.
 *
 * Every Android OEM layers its own power management on top of stock Android, and
 * each one will happily freeze a foreground service mid-drive. Observed directly on
 * a Redmi Note 11:
 *
 *     SmartPower.com.navisense.app died->invisible(345642ms)
 *     ProcessMemoryCleaner: Compact memory: com.navisense.app state=hibernation adj=700
 *
 * That is very likely a cause of the "IMU stopped at 324 s while GNSS ran to 1888 s"
 * pattern seen in early recordings — 85% of a drive lost, with nothing on screen to
 * suggest anything was wrong.
 *
 * The wake lock and foreground service are necessary but NOT sufficient: OEM layers
 * override them. The one portable lever is the standard battery-optimisation
 * exemption, which every OEM must implement. Anything beyond that is per-vendor and
 * can only be signposted, not automated.
 */
object PowerSetup {

    /** The standard, portable check. Works on every Android 6+ device. */
    fun isExempt(context: Context): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) return true
        val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        return pm.isIgnoringBatteryOptimizations(context.packageName)
    }

    /**
     * Opens the system dialog asking to ignore battery optimisation.
     *
     * This is the only cross-vendor route. Samsung, OnePlus, Xiaomi, Oppo and Vivo
     * all honour it, even where they additionally impose their own restrictions.
     */
    fun requestExemption(context: Context) {
        if (isExempt(context)) return
        runCatching {
            context.startActivity(
                Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS)
                    .setData(Uri.parse("package:${context.packageName}"))
                    .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            )
        }.onFailure {
            // Some builds hide the direct dialog; fall back to the settings list.
            runCatching {
                context.startActivity(
                    Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS)
                        .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                )
            }
        }
    }

    /**
     * Extra step this specific manufacturer needs, beyond the standard exemption.
     *
     * These are the aggressive vendors. The instructions are deliberately phrased as
     * a path the user can follow, because the underlying settings screens are not
     * reliably reachable by intent and change between OS versions.
     */
    fun vendorAdvice(): String? = when (Build.MANUFACTURER.lowercase()) {
        "xiaomi", "redmi", "poco" ->
            "MIUI also needs: Settings → Apps → IDR Logger → Autostart ON, " +
                "and Battery saver → No restrictions."
        "oneplus", "oppo", "realme" ->
            "OxygenOS/ColorOS also needs: Settings → Battery → Background power " +
                "usage → Allow, and turn OFF 'Optimise battery use' for this app."
        "samsung" ->
            "One UI also needs: Settings → Battery → Background usage limits → " +
                "make sure IDR Logger is NOT in 'Sleeping' or 'Deep sleeping apps'."
        "vivo", "iqoo" ->
            "Funtouch also needs: Settings → Battery → High background power " +
                "consumption → allow IDR Logger."
        "huawei", "honor" ->
            "EMUI also needs: Settings → Apps → IDR Logger → Battery → " +
                "set Launch to Manage manually, all three toggles ON."
        else -> null
    }

    /** True for vendors known to kill foreground services regardless of exemption. */
    val vendorIsAggressive: Boolean
        get() = Build.MANUFACTURER.lowercase() in setOf(
            "xiaomi", "redmi", "poco", "oneplus", "oppo", "realme",
            "vivo", "iqoo", "huawei", "honor", "samsung",
        )
}
