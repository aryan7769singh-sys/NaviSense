# NaviSense — Engineering Guidelines

NaviSense is a smartphone-based intelligent navigation
system designed to maintain movement tracking during
GNSS outages.

The project combines inertial sensor processing, machine
learning, dead reckoning and trajectory reconstruction.

## 1. Engineering Principles

- A successful compilation does not guarantee correct behaviour.
- Verify runtime behaviour before declaring a feature complete.
- Validate comments against actual implementation.
- Keep numerical algorithms in the Kotlin engine module.
- Keep Android-specific functionality inside the app module.
- Prefer testable, modular implementations.

## 2. Testing and Validation

- Write tests for numerical algorithms.
- Test failure conditions, not just successful execution.
- Verify sensor availability and sampling rates.
- Validate ONNX model inputs and outputs.
- Test GNSS outage and recovery scenarios.
- Never report untested functionality as verified.

## 3. Measurement and Evaluation

Distinguish between:
- Measured results.
- Inferred results.
- Assumptions.

Use consistent dataset splits when comparing results.

Re-run evaluations after changes to the navigation engine.

Do not present stale benchmark results as current performance.

## 4. Sensor Processing

Important engineering considerations:

- Accelerometer calibration.
- Gyroscope bias estimation.
- Sensor coordinate alignment.
- Device orientation and mounting angle.
- Sampling-rate consistency.
- Sensor fusion and drift correction.

## 5. Machine Learning

- Validate model input dimensions and normalization.
- Verify ONNX inference outputs.
- Keep model preprocessing consistent between Python and Kotlin.
- Test inference on representative sensor recordings.
- Monitor prediction errors and accumulated trajectory drift.

## 6. Error Handling

- Detect invalid sensor recordings.
- Surface sensor initialization failures.
- Avoid silently accepting corrupted data.
- Validate uploaded recordings.
- Log important runtime failures.

## 7. Documentation

- Document architectural changes.
- Keep installation instructions current.
- Distinguish prototype functionality from production functionality.
- Record evaluation methodology and limitations.
- Preserve reproducibility of experiments.

## 8. Project Structure

- app/ — Android application.
- engine/ — Kotlin navigation engine.
- proto/ — Python research and evaluation pipeline.
- artifacts/ — Demonstrations, visualizations and results.
- cloud/ — Cloud service components.