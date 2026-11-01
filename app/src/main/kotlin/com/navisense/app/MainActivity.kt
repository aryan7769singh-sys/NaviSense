package com.navisense.app

import android.Manifest
import android.content.Intent
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle

/**
 * Stage 1 of the app: prove the phone captures clean, fast sensor data.
 *
 * Before any navigation exists, this answers the question everything else rests on
 * — what rate does this device actually deliver, and are there gaps? The research
 * pipeline assumes >=100 Hz IMU; a phone that quietly delivers 50 Hz would
 * invalidate the model without any visible error.
 */
class MainActivity : ComponentActivity() {

    private val permissions = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        askPermissions()
        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                var tab by remember { mutableIntStateOf(0) }
                val s by CaptureService.stats.collectAsStateWithLifecycle()
                Surface(Modifier.fillMaxSize()) {
                    Column {
                        TabRow(selectedTabIndex = tab) {
                            Tab(tab == 0, { tab = 0 }) {
                                Text("TRACKER", Modifier.padding(14.dp), fontSize = 12.sp,
                                     fontWeight = FontWeight.SemiBold)
                            }
                            Tab(tab == 1, { tab = 1 }) {
                                Text("SENSORS", Modifier.padding(14.dp), fontSize = 12.sp,
                                     fontWeight = FontWeight.SemiBold)
                            }
                        }
                        if (tab == 0) TrackScreen(s, ::toggleCapture) else CaptureScreen()
                    }
                }
            }
        }
    }

    private fun toggleCapture() {
        val i = Intent(this, CaptureService::class.java)
        if (CaptureService.stats.value.running) stopService(i) else startForegroundService(i)
    }

    private fun askPermissions() {
        val wanted = mutableListOf(Manifest.permission.ACCESS_FINE_LOCATION)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            wanted += Manifest.permission.POST_NOTIFICATIONS
        }
        permissions.launch(wanted.toTypedArray())
    }

    @Composable
    private fun CaptureScreen() {
        val s by CaptureService.stats.collectAsStateWithLifecycle()

        Surface(Modifier.fillMaxSize()) {
            Column(
                Modifier.fillMaxSize().padding(20.dp).verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                Text(
                    "NaviSense",
                    fontSize = 26.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = (-0.5).sp,
                )
                Text(
                    "GNSS-denied navigation · sensor capture",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )

                Spacer(Modifier.height(2.dp))

                // The headline numbers: is the phone actually fast enough?
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    RateCard("ACCEL", s.accHz, Modifier.weight(1f))
                    RateCard("GYRO", s.gyrHz, Modifier.weight(1f))
                    RateCard("MAG", s.magHz, Modifier.weight(1f))
                }

                ElevatedCard {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(7.dp)) {
                        Field("Elapsed", "%.1f s".format(s.elapsedS))
                        Field("IMU samples", "%,d".format(s.samples))
                        Field("GNSS fixes", "${s.fixes}   (${s.satellites} used / ${s.visibleSats} seen)")
                        if (s.fixes == 0 && s.running) {
                            Text(
                                if (s.visibleSats == 0)
                                    "No satellites visible — GPS needs sky view. This is normal indoors."
                                else
                                    "${s.visibleSats} satellites visible, acquiring a fix...",
                                fontSize = 11.sp,
                                color = MaterialTheme.colorScheme.tertiary,
                            )
                        }
                        Field("Speed", "%.1f km/h".format(s.speedMps * 3.6f))
                        Field("Position", if (s.fixes == 0) "waiting for fix"
                                          else "%.5f, %.5f  ±%.0fm".format(s.lat, s.lon, s.accuracyM))
                        Field("File", s.file.ifEmpty { "—" })
                    }
                }

                if (s.running) LiveReadings(s)

                if (s.running && s.elapsedS > 3) Verdict(s)

                Spacer(Modifier.height(4.dp))

                Button(
                    onClick = {
                        val i = Intent(this@MainActivity, CaptureService::class.java)
                        if (s.running) stopService(i) else startForegroundService(i)
                    },
                    modifier = Modifier.fillMaxWidth().height(56.dp),
                    colors = if (s.running)
                        ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error)
                    else ButtonDefaults.buttonColors(),
                ) {
                    Text(if (s.running) "STOP" else "START RECORDING", fontSize = 16.sp,
                         fontWeight = FontWeight.SemiBold)
                }

                UploadPanel()

                Text(
                    "Mount the phone, tap START, drive, tap STOP. Nothing to touch in " +
                        "between — never operate it while driving. Files land in " +
                        "Android/data/com.navisense.app/files/",
                    fontSize = 11.sp,
                    lineHeight = 16.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }

    /** Live sensor values, so the data can be sanity-checked without a laptop. */
    @Composable
    private fun LiveReadings(s: CaptureService.Stats) {
        val accMag = kotlin.math.sqrt(
            s.accXyz.fold(0.0) { a, v -> a + v * v.toDouble() }
        )
        val gyrMag = kotlin.math.sqrt(
            s.gyrXyz.fold(0.0) { a, v -> a + v * v.toDouble() }
        )
        // Angle between measured gravity and the phone's own z axis.
        val tiltDeg = if (accMag > 1) {
            Math.toDegrees(kotlin.math.acos((s.accXyz[2] / accMag).coerceIn(-1.0, 1.0)))
        } else 0.0

        ElevatedCard {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("LIVE READINGS", fontSize = 9.sp, letterSpacing = 1.3.sp,
                     color = MaterialTheme.colorScheme.onSurfaceVariant)
                Row(Modifier.fillMaxWidth()) {
                    listOf("", "X", "Y", "Z", "|v|").forEach {
                        Text(it, Modifier.weight(if (it == "") 1.1f else 1f),
                             fontSize = 10.sp, fontFamily = FontFamily.Monospace,
                             color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
                Axis("acc", s.accXyz, accMag, "m/s²")
                Axis("gyro", s.gyrXyz, gyrMag, "rad/s")
                Axis("mag", s.magXyz, kotlin.math.sqrt(
                    s.magXyz.fold(0.0) { a, v -> a + v * v.toDouble() }), "µT")

                HorizontalDivider(Modifier.padding(vertical = 2.dp))
                // |acc| should read ~9.81 at rest; anything else means the
                // accelerometer is miscalibrated or the phone is moving.
                Field("|acc| at rest should be 9.81", "%.2f".format(accMag))
                Field("phone tilt from vertical", "%.0f°".format(tiltDeg))
            }
        }
    }

    @Composable
    private fun Axis(label: String, v: FloatArray, mag: Double, unit: String) {
        Row(Modifier.fillMaxWidth()) {
            Text(label, Modifier.weight(1.1f), fontSize = 11.sp,
                 color = MaterialTheme.colorScheme.onSurfaceVariant)
            v.forEach {
                Text("%+.2f".format(it), Modifier.weight(1f), fontSize = 11.sp,
                     fontFamily = FontFamily.Monospace)
            }
            Text("%.2f".format(mag), Modifier.weight(1f), fontSize = 11.sp,
                 fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
        }
        Text(unit, fontSize = 8.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }

    @Composable
    private fun UploadPanel() {
        val uploader = remember { Uploader(this) }
        val st by Uploader.status.collectAsStateWithLifecycle()
        var url by remember { mutableStateOf(uploader.endpoint) }
        LaunchedEffect(Unit) { uploader.refresh(); uploader.syncNow() }

        ElevatedCard {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("RIDE UPLOAD", fontSize = 9.sp, letterSpacing = 1.3.sp,
                     color = MaterialTheme.colorScheme.onSurfaceVariant)

                if (uploader.preconfigured) {
                    // Nothing to configure: the build already knows where to send.
                    Text("Automatic — ${st.pending} pending, ${st.uploaded} sent",
                         fontSize = 12.sp, fontFamily = FontFamily.Monospace)
                    st.lastError?.let {
                        Text(it, fontSize = 10.sp, color = MaterialTheme.colorScheme.error)
                    }
                    Text("Rides upload over Wi-Fi on their own. Local files are kept.",
                         fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    return@Column
                }

                Text("Not configured in this build — paste the endpoint once, or set " +
                     "Uploader.DEFAULT_ENDPOINT and rebuild so nobody has to.",
                     fontSize = 10.sp, color = MaterialTheme.colorScheme.tertiary)
                OutlinedTextField(
                    value = url,
                    onValueChange = { url = it },
                    label = { Text("Collection endpoint", fontSize = 11.sp) },
                    placeholder = { Text("https://...workers.dev/token", fontSize = 11.sp) },
                    singleLine = true,
                    textStyle = androidx.compose.ui.text.TextStyle(
                        fontSize = 11.sp, fontFamily = FontFamily.Monospace),
                    modifier = Modifier.fillMaxWidth(),
                )
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    Button(onClick = { uploader.endpoint = url; uploader.syncNow() },
                           enabled = url.isNotBlank()) {
                        Text(if (st.busy) "Uploading..." else "Save & sync", fontSize = 12.sp)
                    }
                    Text("${st.pending} pending  ${st.uploaded} sent",
                         fontSize = 11.sp, fontFamily = FontFamily.Monospace,
                         color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                st.lastError?.let {
                    Text(it, fontSize = 10.sp, color = MaterialTheme.colorScheme.error)
                }
                Text("Wi-Fi only. Local files are never deleted — upload only marks them.",
                     fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }

    @Composable
    private fun RateCard(label: String, hz: Double, modifier: Modifier = Modifier) {
        // 100 Hz is the bar: the road-vibration cue that carries speed lives above
        // 5 Hz, and 10 Hz data (as in IO-VNBD) cannot represent it.
        val good = hz >= 90
        val ok = hz >= 45
        ElevatedCard(modifier) {
            Column(Modifier.padding(vertical = 14.dp, horizontal = 10.dp),
                   horizontalAlignment = Alignment.CenterHorizontally) {
                Text(label, fontSize = 9.sp, letterSpacing = 1.4.sp,
                     color = MaterialTheme.colorScheme.onSurfaceVariant)
                Text(
                    if (hz > 0) "%.0f".format(hz) else "—",
                    fontSize = 26.sp, fontWeight = FontWeight.Bold,
                    color = when { good -> MaterialTheme.colorScheme.primary
                                   ok -> MaterialTheme.colorScheme.tertiary
                                   else -> MaterialTheme.colorScheme.error },
                )
                Text("Hz", fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }

    @Composable
    private fun Verdict(s: CaptureService.Stats) {
        val slowest = listOf(s.accHz, s.gyrHz).filter { it > 0 }.minOrNull() ?: 0.0
        val (text, colour) = when {
            slowest >= 90 -> "Capture rate is good (>=90 Hz). This phone can see the " +
                "vibration band the speed model depends on." to MaterialTheme.colorScheme.primary
            slowest >= 45 -> "Usable but slow (%.0f Hz). Above IO-VNBD's 10 Hz, below the " .format(slowest) +
                "100 Hz the model was designed around." to MaterialTheme.colorScheme.tertiary
            else -> "Too slow (%.0f Hz). The speed cue lives above 5 Hz; this will not " .format(slowest) +
                "carry it." to MaterialTheme.colorScheme.error
        }
        Card(colors = CardDefaults.cardColors(containerColor = colour.copy(alpha = 0.12f))) {
            Text(text, Modifier.padding(14.dp), fontSize = 12.sp, lineHeight = 17.sp, color = colour)
        }
    }

    @Composable
    private fun Field(label: String, value: String) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(label, fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text(value, fontSize = 12.sp, fontFamily = FontFamily.Monospace,
                 fontWeight = FontWeight.Medium)
        }
    }
}
