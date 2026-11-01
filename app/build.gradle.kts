plugins {
    id("com.android.application") version "8.9.0"
    kotlin("android") version "2.0.21"
    id("org.jetbrains.kotlin.plugin.compose") version "2.0.21"
}

android {
    namespace = "com.navisense.app"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.navisense.app"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "0.1"

        ndk {
            // ONNX Runtime ships native libraries for four architectures, which took
            // the debug APK to 92 MB. Every phone made since ~2015 is arm64-v8a;
            // x86_64 is kept only so the app can run in an emulator.
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        release { isMinifyEnabled = false }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true }
}

dependencies {
    implementation(project(":engine"))
    // The Android build of ONNX Runtime: same ai.onnxruntime API, ARM native libs only.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
    implementation(platform("androidx.compose:compose-bom:2024.09.03"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.activity:activity-compose:1.9.2")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.6")
    implementation("androidx.core:core-ktx:1.13.1")
}
