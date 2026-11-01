package com.navisense.app

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.max

private val ACCENT = Color(0xFF48CEDA)
private val CRIT = Color(0xFFE0796A)
private val TRUTH = Color(0xFF5A6A6C)
private val WARN = Color(0xFFDDA43C)
private val OK = Color(0xFF6DC28E)

/**
 * The demo screen: live position, and what happens when GNSS is taken away.
 *
 * Grey is where GNSS says you are. Teal is where the engine thinks you are with
 * GNSS withheld. Before the kill they sit on top of each other; after it, the gap
 * between them IS the drift, growing in real time.
 */
@Composable
fun TrackScreen(s: CaptureService.Stats, onStartStop: () -> Unit) {
    Column(
        Modifier.fillMaxSize().padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically) {
            Column {
                Text("Live Tracker", fontSize = 22.sp, fontWeight = FontWeight.Bold)
                Text(
                    if (s.modelLoaded) "VelNet loaded · IMU-only navigation"
                    else "model not loaded — classical only",
                    fontSize = 11.sp,
                    color = if (s.modelLoaded) OK else WARN,
                )
            }
            GnssPill(s)
        }

        // --- the map -----------------------------------------------------------
        Card(Modifier.fillMaxWidth().weight(1f)) {
            Box(Modifier.fillMaxSize()) {
                TrailCanvas(s)
                if (s.trail.size < 2) {
                    Text(
                        if (!s.running) "Press START to begin"
                        else if (s.fixes == 0) "Waiting for GPS fix — go outside"
                        else "Move to draw a track",
                        Modifier.align(Alignment.Center),
                        fontSize = 13.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        // --- the numbers that make the point ------------------------------------
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Stat("SPEED", "%.0f".format(s.navSpeed * 3.6), "km/h", ACCENT, Modifier.weight(1f))
            Stat("DISTANCE", "%.0f".format(s.navDistance), "m", null, Modifier.weight(1f))
            Stat(
                "DRIFT",
                if (s.gnssEnabled) "—" else "%.0f".format(s.driftM),
                if (s.gnssEnabled) "" else "m",
                if (s.driftM > 0 && s.navDistance > 20 && s.driftM / s.navDistance > 0.10) CRIT else OK,
                Modifier.weight(1f),
            )
            Stat(
                "OF DIST",
                if (!s.gnssEnabled && s.navDistance > 20)
                    "%.0f".format(100 * s.driftM / s.navDistance) else "—",
                if (!s.gnssEnabled && s.navDistance > 20) "%" else "",
                if (s.navDistance > 20 && s.driftM / max(s.navDistance, 1.0) > 0.10) CRIT else OK,
                Modifier.weight(1f),
            )
        }

        if (!s.gnssEnabled) {
            Card(colors = CardDefaults.cardColors(containerColor = WARN.copy(alpha = 0.14f))) {
                Text(
                    "GNSS WITHHELD for %.0f s — position below is from motion sensors alone"
                        .format(s.blindS),
                    Modifier.padding(12.dp), fontSize = 12.sp, color = WARN,
                )
            }
        }

        // --- will this device even let us finish the recording? --------------------
        PowerWarning()

        // --- is this ride actually worth anything? --------------------------------
        if (s.running) {
            val v = RideQuality.assess(s)
            val c = when (v.level) {
                RideQuality.Level.GOOD -> OK
                RideQuality.Level.WARN -> WARN
                RideQuality.Level.BAD -> CRIT
            }
            Card(colors = CardDefaults.cardColors(containerColor = c.copy(alpha = 0.15f))) {
                Column(Modifier.padding(12.dp)) {
                    Text(v.headline, fontSize = 14.sp, fontWeight = FontWeight.Bold, color = c)
                    Text(v.detail, fontSize = 11.sp, lineHeight = 15.sp,
                         color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }

        // --- controls ------------------------------------------------------------
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Button(
                onClick = onStartStop,
                modifier = Modifier.weight(1f).height(52.dp),
                colors = if (s.running)
                    ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error)
                else ButtonDefaults.buttonColors(),
            ) { Text(if (s.running) "STOP" else "START", fontWeight = FontWeight.SemiBold) }

            Button(
                onClick = { CaptureService.setGnss(!s.gnssEnabled) },
                enabled = s.running,
                modifier = Modifier.weight(1.4f).height(52.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (s.gnssEnabled) CRIT else OK,
                ),
            ) {
                Text(
                    if (s.gnssEnabled) "KILL GPS" else "RESTORE GPS",
                    fontWeight = FontWeight.Bold, color = Color.Black,
                )
            }
        }

        // A required sensor never started, so recording was aborted. This has to be
        // impossible to miss: the failure it replaces was a silently gyro-less file
        // that only showed up during analysis, hours after the drive.
        s.fatalError?.let { message ->
            Surface(
                color = CRIT,
                shape = MaterialTheme.shapes.medium,
                modifier = Modifier.fillMaxWidth().padding(top = 12.dp),
            ) {
                Text(
                    message,
                    Modifier.padding(12.dp),
                    color = Color.Black,
                    fontSize = 13.sp,
                    fontWeight = FontWeight.SemiBold,
                )
            }
        }
    }
}

/**
 * Warns when the device will likely kill the recording partway through.
 *
 * Shown before anything else because it is the failure that costs a whole drive and
 * leaves no trace on screen — an early recording lost 85% of its data this way.
 */
@Composable
private fun PowerWarning() {
    val ctx = androidx.compose.ui.platform.LocalContext.current
    var exempt by remember { mutableStateOf(PowerSetup.isExempt(ctx)) }
    // Re-check when returning from the system dialog.
    androidx.compose.runtime.LaunchedEffect(Unit) { exempt = PowerSetup.isExempt(ctx) }

    val advice = PowerSetup.vendorAdvice()
    if (exempt && advice == null) return

    Card(colors = CardDefaults.cardColors(
        containerColor = (if (exempt) WARN else CRIT).copy(alpha = 0.15f))) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(
                if (exempt) "One more step on this device" else "Battery optimisation will stop recording",
                fontSize = 13.sp, fontWeight = FontWeight.Bold,
                color = if (exempt) WARN else CRIT,
            )
            if (!exempt) {
                Text("Android will freeze this app partway through a drive. Tap to allow it to run.",
                     fontSize = 11.sp, lineHeight = 15.sp,
                     color = MaterialTheme.colorScheme.onSurfaceVariant)
                Button(onClick = { PowerSetup.requestExemption(ctx); exempt = PowerSetup.isExempt(ctx) },
                       modifier = Modifier.fillMaxWidth()) {
                    Text("ALLOW BACKGROUND RUNNING", fontSize = 12.sp, fontWeight = FontWeight.Bold)
                }
            }
            advice?.let {
                Text(it, fontSize = 11.sp, lineHeight = 15.sp,
                     color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

@Composable
private fun GnssPill(s: CaptureService.Stats) {
    val (label, colour) = when {
        !s.running -> "IDLE" to TRUTH
        !s.gnssEnabled -> "GNSS LOST" to CRIT
        s.fixes == 0 -> "ACQUIRING" to WARN
        else -> "GNSS OK" to OK
    }
    Surface(color = colour.copy(alpha = 0.18f), shape = MaterialTheme.shapes.small) {
        Text(label, Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
             color = colour, fontSize = 11.sp, fontWeight = FontWeight.Bold,
             fontFamily = FontFamily.Monospace)
    }
}

@Composable
private fun Stat(label: String, value: String, unit: String, colour: Color?, modifier: Modifier) {
    ElevatedCard(modifier) {
        Column(Modifier.padding(vertical = 10.dp, horizontal = 8.dp),
               horizontalAlignment = Alignment.CenterHorizontally) {
            Text(label, fontSize = 8.sp, letterSpacing = 1.2.sp,
                 color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text(value, fontSize = 21.sp, fontWeight = FontWeight.Bold,
                 color = colour ?: MaterialTheme.colorScheme.onSurface)
            Text(unit, fontSize = 9.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun TrailCanvas(s: CaptureService.Stats) {
    Canvas(Modifier.fillMaxSize().padding(10.dp)) {
        val pts = s.trail
        val truth = s.truthTrail
        if (pts.size < 2 && truth.size < 2) return@Canvas

        // Fit both tracks in view with a margin.
        var minE = Float.MAX_VALUE; var maxE = -Float.MAX_VALUE
        var minN = Float.MAX_VALUE; var maxN = -Float.MAX_VALUE
        (pts + truth).forEach {
            minE = minOf(minE, it[0]); maxE = maxOf(maxE, it[0])
            minN = minOf(minN, it[1]); maxN = maxOf(maxN, it[1])
        }
        val spanE = max(maxE - minE, 20f)
        val spanN = max(maxN - minN, 20f)
        val scale = minOf(size.width / spanE, size.height / spanN) * 0.85f
        val cx = size.width / 2f
        val cy = size.height / 2f
        val midE = (minE + maxE) / 2f
        val midN = (minN + maxN) / 2f
        fun px(p: FloatArray) = Offset(cx + (p[0] - midE) * scale, cy - (p[1] - midN) * scale)

        fun stroke(list: List<FloatArray>, colour: Color, width: Float) {
            if (list.size < 2) return
            val path = Path().apply {
                val f = px(list[0]); moveTo(f.x, f.y)
                for (i in 1 until list.size) { val q = px(list[i]); lineTo(q.x, q.y) }
            }
            drawPath(path, colour, style = Stroke(width = width, cap = StrokeCap.Round))
        }

        stroke(truth, TRUTH, 10f)          // where GNSS says we went
        stroke(pts, ACCENT, 4f)            // where the engine thinks we went

        truth.lastOrNull()?.let { drawCircle(TRUTH, 9f, px(it)) }
        pts.lastOrNull()?.let { drawCircle(ACCENT, 11f, px(it)) }

        // The moment GNSS was cut, and the gap it opened.
        if (!s.gnssEnabled && pts.isNotEmpty() && truth.isNotEmpty()) {
            val a = px(pts.last()); val b = px(truth.last())
            drawLine(CRIT, a, b, strokeWidth = 2.5f)
        }
    }
}
