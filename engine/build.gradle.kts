plugins {
    kotlin("jvm") version "2.0.21"
}

kotlin { jvmToolchain(21) }

dependencies {
    // compileOnly, NOT implementation. The desktop and Android ONNX Runtime
    // artifacts expose the identical `ai.onnxruntime` API but ship different native
    // libraries. Compiling against the API here and letting each consumer supply
    // its own runtime is what keeps this module platform-neutral.
    //
    // Declaring it as `implementation` leaked the desktop artifact — with native
    // binaries for Windows, Linux and macOS — into the Android APK, taking it to
    // 92 MB.
    compileOnly("com.microsoft.onnxruntime:onnxruntime:1.20.0")
    testImplementation("com.microsoft.onnxruntime:onnxruntime:1.20.0")
    testImplementation(kotlin("test"))
}

tasks.test { useJUnitPlatform() }
