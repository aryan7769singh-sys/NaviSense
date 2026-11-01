package com.navisense.app

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

/**
 * Ships finished recordings to a collection endpoint, so data from many phones and
 * many vehicles ends up in one place.
 *
 * This is the point of the whole logger. The measured bottleneck is not hours of
 * data, it is VEHICLE DIVERSITY — the model plateaued at ~4.5 m/s speed error
 * because IO-VNBD contains about five cars, and suspension response is part of the
 * signal it has to learn. One person cannot fix that. Twenty people driving twenty
 * different vehicles can.
 *
 * Design rules, each chosen to avoid a specific way this goes wrong:
 *
 *  - **Never delete the local file.** Upload marks a sidecar `.uploaded`; the CSV
 *    stays. A failed upload must never cost a drive that cannot be repeated.
 *  - **Wi-Fi only by default.** Rides are 10-30 MB; nobody will keep using a logger
 *    that eats their mobile data.
 *  - **Resume on next launch.** Anything unmarked is retried, so a drive that ended
 *    in a basement car park still arrives eventually.
 *  - **Quality travels with the file.** The server can then reject unusable rides
 *    rather than silently training on them.
 */
class Uploader(private val context: Context) {

    data class Status(
        val endpoint: String = "",
        val pending: Int = 0,
        val uploaded: Int = 0,
        val failed: Int = 0,
        val busy: Boolean = false,
        val lastError: String? = null,
    )

    companion object {
        /**
         * Baked-in collection endpoint. Set this once after `wrangler deploy` and
         * every install uploads automatically with nothing to configure.
         *
         * This matters for the actual goal: the model's limitation is vehicle
         * diversity, which is only fixed by OTHER PEOPLE driving THEIR vehicles.
         * Anyone who has to paste a URL before their data counts will not do it.
         * Ship it pre-wired, or the flywheel never turns.
         *
         * Format: https://idr-collect.<subdomain>.workers.dev/<upload-token>
         */
        const val DEFAULT_ENDPOINT = "http://172.16.37.70:8000/sih26168-zL6v-Zx6"

        private const val PREFS = "idr_upload"
        private const val KEY_ENDPOINT = "endpoint"
        private const val KEY_WIFI_ONLY = "wifi_only"

        private val _status = MutableStateFlow(Status())
        val status: StateFlow<Status> = _status
    }

    private val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    private val scope = CoroutineScope(Dispatchers.IO)

    /** Stored override wins, otherwise the baked-in default. */
    var endpoint: String
        get() = prefs.getString(KEY_ENDPOINT, "")?.ifBlank { DEFAULT_ENDPOINT } ?: DEFAULT_ENDPOINT
        set(v) {
            prefs.edit().putString(KEY_ENDPOINT, v.trim()).apply()
            _status.value = _status.value.copy(endpoint = v.trim())
            refresh()
        }

    var wifiOnly: Boolean
        get() = prefs.getBoolean(KEY_WIFI_ONLY, true)
        set(v) = prefs.edit().putBoolean(KEY_WIFI_ONLY, v).apply()

    /** True when uploads need no user action at all. */
    val preconfigured: Boolean get() = DEFAULT_ENDPOINT.isNotBlank()

    private fun dir(): File = context.getExternalFilesDir(null) ?: context.filesDir

    private fun pendingFiles(): List<File> =
        dir().listFiles { f -> f.name.endsWith(".csv") }
            ?.filter { !File(it.absolutePath + ".uploaded").exists() && it.length() > 4096 }
            ?.sortedBy { it.lastModified() }
            ?: emptyList()

    fun refresh() {
        _status.value = _status.value.copy(
            endpoint = endpoint,
            pending = pendingFiles().size,
        )
    }

    private fun networkOk(): Boolean {
        val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val caps = cm.getNetworkCapabilities(cm.activeNetwork) ?: return false
        if (!caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)) return false
        return !wifiOnly || caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)
    }

    /** Try to send everything not yet marked as uploaded. Safe to call repeatedly. */
    fun syncNow() {
        if (endpoint.isBlank()) {
            _status.value = _status.value.copy(lastError = "No endpoint set")
            return
        }
        if (_status.value.busy) return
        if (!networkOk()) {
            _status.value = _status.value.copy(
                lastError = if (wifiOnly) "Waiting for Wi-Fi" else "No network"
            )
            return
        }

        scope.launch {
            _status.value = _status.value.copy(busy = true, lastError = null)
            var sent = 0
            var failed = 0
            for (f in pendingFiles()) {
                if (!networkOk()) break
                val err = upload(f)
                if (err == null) {
                    File(f.absolutePath + ".uploaded").writeText(System.currentTimeMillis().toString())
                    sent++
                } else {
                    failed++
                    _status.value = _status.value.copy(lastError = err)
                }
                _status.value = _status.value.copy(
                    uploaded = _status.value.uploaded + if (err == null) 1 else 0,
                    failed = failed,
                    pending = pendingFiles().size,
                )
            }
            _status.value = _status.value.copy(busy = false, pending = pendingFiles().size)
        }
    }

    /** Returns null on success, or a human-readable error. */
    private fun upload(file: File): String? = runCatching {
        val url = URL(endpoint.trimEnd('/') + "/" + file.name)
        (url.openConnection() as HttpURLConnection).run {
            requestMethod = "PUT"
            doOutput = true
            connectTimeout = 20_000
            readTimeout = 120_000
            setRequestProperty("Content-Type", "text/csv")
            setFixedLengthStreamingMode(file.length())
            // Metadata the collector needs to sort rides without parsing 20 MB.
            setRequestProperty("X-Device", "${android.os.Build.MANUFACTURER}-${android.os.Build.MODEL}")
            setRequestProperty("X-App-Version", "0.1")

            file.inputStream().use { input -> outputStream.use { out -> input.copyTo(out, 64 * 1024) } }

            val code = responseCode
            if (code in 200..299) null else "HTTP $code"
        }
    }.getOrElse { "${it::class.simpleName}: ${it.message}" }
}
