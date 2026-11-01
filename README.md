# NaviSense

### When GPS disappears, navigation shouldn't stop.

NaviSense is a smartphone-based intelligent navigation
system designed to maintain movement tracking during
GNSS outages.

It combines inertial sensor measurements, machine learning,
and dead reckoning to estimate movement when satellite
positioning becomes unavailable.

## Core Capabilities

- GNSS outage detection
- Smartphone IMU data acquisition
- Inertial dead reckoning
- Machine learning-based speed estimation
- Trajectory reconstruction
- GNSS recovery and position comparison

## Project Structure

- `app/` — Android application
- `engine/` — Kotlin navigation engine
- `proto/` — Python research and evaluation pipeline
- `artifacts/` — Demonstrations, visualizations and results
- `cloud/` — Cloud service components

## Live Demo

https://aryan7769singh-sys.github.io/NaviSense/

## Source Code

https://github.com/aryan7769singh-sys/NaviSense