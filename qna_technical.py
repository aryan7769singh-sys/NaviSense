"""Technical Q&A content — for whoever on the team can defend the engineering.

Separate from Judge_QnA.pdf on purpose: different reader, different depth.

RULE FOR EDITING THIS FILE: every number here must trace to a measurement in
artifacts/, to a source file, or to a cited paper. If a figure is inferred rather
than measured, the answer must say so in words. An ISRO panel will pull one thread;
a number nobody can source loses the whole deck.

Sources for the numbers used below:
  artifacts/win10_log.txt      latest full IO-VNBD run (speed RMSE, drift table)
  artifacts/mapmatch2_log.txt  map-matching gain and pass rates
  artifacts/full_log.txt       pre-gyro-fix drift, used as the "before" figure
  proto/velnet.py              architecture; params confirmed by instantiation
  app/src/main/assets/         velnet.onnx, 206,797 bytes
  engine/src/test/             32 tests
"""

INTRO = (
    "This document is for the person on the team who can defend the engineering. "
    "It assumes the reader knows what a Kalman filter and a convolutional network "
    "are.\n\n"
    "Answers are written the way they should be spoken: the claim first, then the "
    "number, then the limit. Where we do not know something the answer says so - "
    "those are the strongest answers in here, not the weakest.\n\n"
    "Every figure is measured on held-out journeys unless the text says otherwise. "
    "Three figures are explicitly marked NOT MEASURED. Do not quietly upgrade them "
    "under pressure."
)

SECTIONS = [

    # ------------------------------------------------------------------ model
    ("1.  THE MODEL", [

        ("Walk me through the network.",
         "A dilated causal temporal convolutional network. 5 layers, dilations "
         "1-2-4-8-16, kernel width 5, 48 channels wide, 49,874 parameters. Input is "
         "8 channels by 100 samples; output is a speed and a log-variance.",
         "Receptive field is 125 samples, deliberately larger than the 100-sample "
         "window so the last output sees the whole input. Causal padding means no "
         "output ever depends on a future sample, which is what makes it deployable "
         "in real time rather than only in post-processing."),

        ("Why a CNN and not an LSTM, GRU or transformer?",
         "OdoNet tested CNN against RNN on this exact task and found the CNN better "
         "on both precision AND efficiency. TinyOdom uses a convolutional backbone "
         "for on-device inertial navigation. We followed the published comparison "
         "rather than picking by fashion.",
         "There is an engineering reason too. A TCN has fixed compute per step and "
         "no hidden state to corrupt, so latency is deterministic and a dropped "
         "sample cannot poison all future outputs. For a system that must hold a "
         "hard 10 Hz guarantee that matters more than a fraction of a m/s."),

        ("What are the 8 input channels?",
         "Three accelerometer axes, three gyroscope axes, and the magnitude of each "
         "vector. The two magnitudes are rotation-invariant, so the network still "
         "has usable signal even when our frame alignment is wrong.",
         "That last point is not decoration. We know alignment is our weakest "
         "component, and the magnitude channels are what stop a bad alignment from "
         "destroying the speed estimate outright."),

        ("Why remove the mean of each window?",
         "Because the DC level of the accelerometer is gravity plus sensor bias, and "
         "that combination identifies the RECORDING, not the speed. Left in, the "
         "network learns to recognise which drive it is looking at.",
         "We measured this failing. With the DC term present, validation error "
         "climbed to 19.5 m/s while training error kept falling - textbook "
         "memorisation. Per-window mean removal fixed it. It is the single largest "
         "correction we made to the input representation."),

        ("What is the loss?",
         "Mean squared error for the first half of training, then Gaussian negative "
         "log-likelihood with a retained MSE term. The network predicts its own "
         "uncertainty, and the filter uses that as measurement noise directly rather "
         "than a hand-tuned constant.",
         "Pure NLL from step zero collapses, and it collapses quietly. The cheapest "
         "way to reduce the loss early is to inflate sigma rather than fit the mean; "
         "once sigma is large the gradient on the mean nearly vanishes. Training "
         "loss falls the whole time while validation error rises. The MSE warm-up "
         "prevents it."),

        ("How much training data, and how is it split?",
         "35 usable recordings from IO-VNBD, 8.4 hours, speeds 0 to 116 km/h. Split "
         "25 journeys for training and 10 for validation - split by journey, never "
         "by window.",
         "The dataset ships 72 recordings but only 35 survive screening: the rest "
         "are under three minutes or never actually move. It also ships every "
         "recording TWICE, byte-identical, under two folder trees. Loading both puts "
         "the same drive in train and test, and nothing warns you."),

        ("Why is splitting by journey so important?",
         "Consecutive windows overlap by about 97 percent. A random split therefore "
         "puts near-identical rows in both sets and you are largely testing on your "
         "training data.",
         "There is a published measurement of exactly this. The same paper evaluates "
         "DeepTrack at 1.12 m/s on a random window split and 4.15 m/s on a held-out "
         "trajectory split - a factor of 3.7 from the split method alone. We report "
         "the strict number, which is why ours can look worse than papers that are "
         "actually weaker."),

        ("Is 49,874 parameters not too small to learn this?",
         "It is small on purpose and we have no evidence it is the binding "
         "constraint. Validation error stops improving after about 5 epochs and then "
         "slowly worsens - that is a data limit, not a capacity limit.",
         "If capacity were the problem we would see training and validation error "
         "both plateau high. Instead training error keeps falling while validation "
         "rises. Adding parameters against 8.4 hours of data from roughly 5 vehicles "
         "would buy overfitting, not accuracy."),
    ]),

    # ---------------------------------------------------------------- sensors
    ("2.  SENSORS AND SIGNAL", [

        ("What is the actual physical cue the network reads?",
         "Road roughness. A vehicle's suspension is excited by the road surface, and "
         "the amplitude and spectrum of that excitation scale with speed. "
         "CarSpeedNet demonstrated that small road-surface irregularities carry "
         "speed information. Tyre and engine vibration add to it.",
         "This is also the honest weakness: on a very smooth surface the cue "
         "weakens. We have not yet isolated a smooth-road subset to quantify how "
         "much. That is a measurement we owe."),

        ("Why can you not just double-integrate the accelerometer?",
         "Because bias integrates quadratically. A 0.05 m/s2 bias - small for a "
         "phone MEMS part - becomes about 90 metres of position error in 60 seconds, "
         "and nothing in the accelerometer itself can observe it.",
         "Be precise about where this actually bites. Over 10 seconds classical "
         "strapdown BEATS us, 26.5 percent against our 33.0 - it has not had time to "
         "diverge. Our advantage begins at 30 seconds, 36.3 against 41.5, and widens "
         "to 58.3 against 70.8 at 180 seconds. We should be weighting the two by "
         "elapsed blackout time. We are not yet."),

        ("How bad is the tilt sensitivity?",
         "One degree of tilt error leaks 0.17 m/s2 of gravity into the forward axis. "
         "Median vehicle acceleration in this dataset is 0.174 m/s2. So one degree "
         "of error produces a false signal the same size as the real one.",
         "This is the number that reframed the whole project. It is why mounting "
         "angle dominates, and why OdoNet found vehicle type and road condition "
         "matter comparatively little."),

        ("What is your sampling rate, and why does it matter so much?",
         "IO-VNBD is 10 Hz. Our own app captures 402 Hz from the accelerometer, 200 "
         "to 800 Hz from the gyroscope and 100 Hz from the magnetometer, measured on "
         "a mid-range Redmi Note 11.",
         "10 Hz gives a 5 Hz Nyquist limit. Idle engine vibration lives at 20 to 100 "
         "Hz and suspension detail well above 5 Hz, so much of the cue is not "
         "attenuated in the dataset - it is absent. Everything we report is "
         "therefore a lower bound on what the method can do."),

        ("Do you handle aliasing when you downsample?",
         "Yes. Capture runs at the native rate and is decimated to 10 Hz with an "
         "averaging anti-alias filter, not by taking every Nth sample. Naive "
         "decimation would fold that 20 to 100 Hz vibration back down as noise.",
         "The filter and the dead-reckoning integrator run at the native rate. Only "
         "the network input is decimated, because the network was trained at 10 Hz."),

        ("What about magnetometer heading? Or a barometer for altitude?",
         "The magnetometer is gated out entirely inside a vehicle - the steel body, "
         "the speakers and the wiring dominate the Earth's field, and a tunnel's "
         "reinforcement makes it worse. Heading comes from the gyroscope and the "
         "road map.",
         "Altitude we deliberately do not estimate. The requirement is horizontal "
         "positioning; a barometer would add a vertical channel we are not scored on, "
         "plus a pressure-drift problem in exactly the enclosed spaces we care about. "
         "It is a reasonable future addition for multi-storey car parks, where floor "
         "level is genuinely the question."),

        ("Why not just read the speed off the car? OBD-II, or the CAN bus?",
         "We DO - during training. The wheel-speed sensor is our answer key: it is "
         "how the network learns what a given vibration pattern means. Then we throw "
         "it away, so at runtime the phone needs nothing.",
         "Three reasons not to require it in the field. It is a cost and a fitting "
         "job per vehicle. It does not exist in a metro, a bus, or any vehicle the "
         "user does not own - two of our five test rides had no OBD port available "
         "at all. And ISRO wants this engine on their own platforms, which do not "
         "have automotive OBD."),

        ("Why not use the camera? Visual odometry is well established.",
         "That is the strongest version of this question. Visual-inertial odometry "
         "genuinely beats IMU-only where it applies, and if the deployment allowed a "
         "windscreen mount I would want it in the system.",
         "It does not apply here. The camera must see out of a window - and we "
         "already cannot guarantee the phone is rigidly mounted, let alone aimed. "
         "Continuous capture plus vision inference is far more battery and compute "
         "than a 50 KB model on IMU data. And it puts a camera recording inside "
         "private vehicles, which is a consent problem we do not have with an "
         "accelerometer."),

        ("What about the microphone? Tyre and engine noise track speed.",
         "We have not tried it, and I think we should. This is a real gap in what we "
         "have tested rather than a decision we have justified.",
         "The argument for it got stronger today. We measured where the speed cue "
         "actually sits in our accelerometer and it is weaker than we assumed - mean "
         "correlation about 0.3 in the best band. A microphone samples at 44 kHz "
         "where our accelerometer manages 400, and tyre and engine noise sit exactly "
         "in the band we are struggling to resolve. It may carry more speed "
         "information than the sensor we are currently using."),

        ("Why not WiFi, cell towers, or beacons in the tunnel?",
         "Because those are infrastructure. That is a cost per tunnel rather than per "
         "vehicle, and it means the system only works where somebody already paid for "
         "it.",
         "Dead reckoning has to work in the tunnel nobody upgraded, and in the "
         "basement car park with no installation. Where that infrastructure does "
         "exist it is a perfectly good complement and the filter would accept it as "
         "another measurement - but it cannot be the thing we depend on."),

        ("How do you handle sensor bias and calibration across devices?",
         "Accelerometer bias is a filter state. Gyro bias is estimated against GNSS "
         "course rate while satellites are visible, then held through the blackout. "
         "We do not rely on factory calibration.",
         "OdoNet found IMU bias and mounting angle to be the two factors that ruin "
         "the method, and added an explicit data-cleaning step for both. We have "
         "implemented the bias half. The mounting-angle half is our open problem."),
    ]),

    # --------------------------------------------------------------- filtering
    ("3.  ESTIMATION AND FILTERING", [

        ("Describe the filter.",
         "A two-state Kalman filter over forward speed and accelerometer bias. The "
         "network supplies a speed pseudo-measurement with its own variance; "
         "zero-velocity detections supply a hard speed-zero measurement. The output "
         "feeds a dead-reckoning integrator.",
         "Deliberately small. We wanted every state in it to be one we had measured "
         "and understood before adding more. A 15-state error-state filter with an "
         "unvalidated attitude section would hide exactly the errors we were trying "
         "to find."),

        ("Why not a full 15-state error-state EKF, or an invariant EKF like AVNet?",
         "That is the right destination and it is in our plan. AVNet feeds learned "
         "attitude and velocity as pseudo-measurements into an invariant EKF and "
         "reports 0.4 percent of distance.",
         "Two caveats we insist on. First, AVNet's 0.4 percent was measured in a "
         "parking lot - low speed, short duration - not on 180-second open-road "
         "blackouts, so it will not transfer as-is. Second, a 15-state filter is only "
         "as good as its attitude input, and attitude is the component we have "
         "measured failing. Fixing it has to come first."),

        ("How do you detect zero velocity? An accelerometer cannot distinguish "
         "stationary from constant velocity.",
         "Correct, and that is Galilean relativity - no accelerometer will ever solve "
         "it. We use three independent sources: GNSS speed when available, "
         "high-frequency vibration energy, and the learned speed. Any one can fire.",
         "The physics route depends on idle engine vibration at 20 to 100 Hz. At the "
         "dataset's 10 Hz that band is above Nyquist and simply gone, which forced an "
         "earlier version to gate zero-velocity on the model output - which is "
         "circular, and disabled the correction exactly when it was most needed. Our "
         "402 Hz capture restores the physics route."),

        ("Do you use non-holonomic constraints?",
         "Yes. A road vehicle cannot move sideways or vertically relative to its own "
         "body frame, so lateral and vertical velocity are constrained to near zero. "
         "That is a free measurement at every step, and it is what makes a two-state "
         "filter viable at all.",
         "It fails in exactly the cases you would expect: skids, sharp low-speed "
         "manoeuvring, and being carried on another vehicle. We do not currently "
         "detect those conditions."),

        ("How is gyro bias estimated, and how much does it matter?",
         "Estimated against GNSS course rate while GNSS is available, then frozen "
         "through the blackout. It matters enormously.",
         "Measured directly: fixing gyro bias handling took 180-second drift from "
         "86.5 percent to 56.9 percent. An uncorrected 0.5 to 0.6 deg/s bias is 30 to "
         "36 degrees of heading error per minute. Residual after correction is around "
         "0.3 deg/s, or 18 degrees per minute, which is still the largest single term "
         "in our error budget."),

        ("What is your error budget?",
         "Measured by substituting oracle values one term at a time on 60-second "
         "blackouts. Heading contributes 27.2 percent, speed contributes 17.2. "
         "Heading is the larger term.",
         "We got this wrong for a long time and it cost us. We optimised speed on the "
         "assumption it dominated. Cutting speed RMSE by 13.8 percent, 4.56 to 3.93 "
         "m/s, moved 60-second drift from 43.1 to 42.6 percent - half a point. What "
         "propagates into distance error is the systematic per-trip bias, about 1.6 "
         "m/s, not the random error that RMSE measures."),

        ("How do you determine the phone's orientation in the vehicle?",
         "Gravity from the low-passed accelerometer gives roll and pitch. Yaw comes "
         "from principal component analysis of horizontal acceleration during "
         "straight-line braking, where acceleration must lie along the vehicle's "
         "forward axis.",
         "STATE THE FAILURE. We measured this on real captures and got alignment "
         "confidence between 0.05 and 0.48, against a threshold of 0.35 for the "
         "estimate to be trusted. It does not reliably work. Replacing it with a "
         "learned attitude head, supervised by GNSS-derived heading, is our "
         "highest-priority fix and needs no new data."),

        ("Why would a learned attitude head work where the analytic method does not?",
         "Because the analytic method needs a clean straight-line braking event to "
         "exist and be identified, and in ordinary driving there often is not one. A "
         "learned head can use the whole window, including turning dynamics and "
         "vibration anisotropy.",
         "We would represent the rotation in a 6D continuous parameterisation rather "
         "than Euler angles or quaternions. Rotation regression fails on those "
         "because of the discontinuities - the network has to learn a function that "
         "jumps, and it cannot."),
    ]),

    # -------------------------------------------------------------------- map
    ("4.  THE MAP", [

        ("How does the map matching work?",
         "A hidden Markov model over an OpenStreetMap road graph, decoded with "
         "Viterbi - the Newson-Krumm formulation. Emission probability from distance "
         "to each candidate road, transition probability from route feasibility "
         "between consecutive candidates. The graph is stored on the phone.",
         "On-device storage is not an optimisation, it is a requirement. The whole "
         "premise is that there is no signal in a tunnel, so a server-side matcher "
         "would be useless in the one case that matters."),

        ("Why does map matching not simply fix the problem?",
         "Because it only constrains the direction it actually knows. We apply the "
         "correction ACROSS the road, never along it. The matcher can tell you which "
         "road you are on, so lateral position is well determined - but not how far "
         "along it you are, which is exactly the information dead reckoning lost.",
         "Snapping fully along the road would replace a real estimate with a guess, "
         "and would look better on screen while being less correct. The anisotropy is "
         "deliberate."),

        ("How much does it actually buy you?",
         "Measured: 30-second drift 32.3 to 26.5 percent, a gain of 18 percent. "
         "60-second drift 39.5 to 32.9, a gain of 17 percent. It also lifts the "
         "fraction of blackouts meeting the 10 percent spec to 17.1 percent at 30 "
         "seconds and 12.2 percent at 60.",
         "The gain shrinks with duration - 4 percent at 120 seconds, 6 percent at "
         "180. Once along-road uncertainty is large enough, knowing the road stops "
         "helping much. That is consistent with the constraint being across-track "
         "only."),

        ("What is the map worth for heading specifically?",
         "A road bearing is accurate to 2 to 3 degrees. Our gyro drifts about 18 "
         "degrees per minute after bias correction. Since heading is the largest term "
         "in the error budget at 27.2 percent, the map is the strongest heading "
         "reference we have during a blackout.",
         "In a tunnel it is the ONLY heading reference - no GNSS course, no usable "
         "magnetometer. That is the argument for the map being architectural rather "
         "than an add-on."),
    ]),

    # ------------------------------------------------------------- evaluation
    ("5.  EVALUATION", [

        ("Give me your headline numbers.",
         "Speed RMSE 3.93 m/s against 8.23 for predicting the mean. Median blackout "
         "drift 33.0 percent at 10 seconds, 36.3 at 30, 42.6 at 60, 51.7 at 120, 58.3 "
         "at 180. With map matching, 26.5 at 30 seconds and 32.9 at 60. All on "
         "held-out journeys.",
         "Sample sizes are 429 to 489 blackouts per duration, over distances from 130 "
         "metres at 10 seconds to 1,673 metres at 180."),

        ("You are quoting medians. What does the distribution look like?",
         "The tail is bad and we should say so. The 90th percentile is 96 percent at "
         "10 seconds and 106 percent at 60 - meaning one blackout in ten ends up "
         "roughly as far from the truth as it travelled.",
         "Median and p90 diverging that much says the failures are not gradual "
         "degradation but distinct bad cases, most likely alignment failures. It is "
         "another piece of evidence pointing at attitude."),

        ("How do you measure error inside a tunnel, where there is no truth?",
         "You cannot, and we do not claim to. The metric is end-point error: take the "
         "last good fix before the blackout, dead-reckon through, compare against the "
         "first good fix after, divide by distance travelled.",
         "That is the metric the problem statement specifies. For development we "
         "simulate blackouts on open-sky drives where truth exists throughout, which "
         "is where every number in this document comes from."),

        ("Your fused result is barely better than the network alone. Why keep the "
         "filter?",
         "Correct, and worth admitting - 33.05 percent network-only against 32.96 "
         "fused at 10 seconds. The filter is currently contributing almost nothing "
         "beyond the network.",
         "We read that as the filter being under-used rather than useless. It has no "
         "adaptive weighting between strapdown and network, and no attitude state. "
         "Both of those are how a filter earns its place, and neither is implemented "
         "yet. Quoting fusion as a strength today would be dishonest."),

        ("How do you compare against the state of the art?",
         "Carefully, because most published numbers are random-split. Measured on "
         "held-out trajectories the honest state of the art is about 2.26 m/s (DVSE), "
         "and a well-known method, DeepTrack, gets 4.15. We get 3.93.",
         "So we already beat one published method under honest evaluation and are "
         "about 1.7x from the best. The frequently quoted 1.80 m/s from CarSpeedNet "
         "and 0.49 from OdoNet are on different data and, for CarSpeedNet, very "
         "likely a random split. We do not compare against those directly and neither "
         "should anyone else."),

        ("How do you know your numbers are not just a bug?",
         "32 automated tests over the engine mathematics, and after writing a test we "
         "deliberately reintroduce the bug to confirm the test fails for the right "
         "reason. A test nobody has seen fail proves nothing.",
         "It has caught real bugs. One comment claimed forward-bias removal that the "
         "code never performed. One Rodrigues rotation was missing its sine term, "
         "which makes it exactly correct at 90 degrees and catastrophically wrong near "
         "zero - it survived days of synthetic testing because the synthetic case sat "
         "near 90."),

        ("How do you know the phone agrees with your Python research code?",
         "The numeric engine is written once, in Kotlin, with no Android dependencies. "
         "The app and the laptop tool link the same compiled code. The Python and "
         "Kotlin implementations are cross-checked to agree to 1e-6.",
         "This removes an entire class of bug where offline evaluation and the shipped "
         "product quietly disagree, which is the kind of thing that invalidates a "
         "whole results table without anyone noticing."),
    ]),

    # ----------------------------------------------------------------- system
    ("6.  SYSTEM AND DEPLOYMENT", [

        ("What is the inference cost?",
         "The model is 202 KB in ONNX. Measured on a laptop CPU pinned to one thread: "
         "0.20 ms median, 0.50 ms at the 95th percentile, per inference.",
         "NOT MEASURED ON THE PHONE. We have not instrumented on-device timing, so do "
         "not quote a phone figure. What we can say is that the budget is 100 ms at 10 "
         "Hz, so even a 10x penalty on mobile hardware leaves a very large margin. Say "
         "that as an inference, not as a measurement."),

        ("How do you hit a real-time guarantee?",
         "Fixed compute per step. No recurrence, no variable-length search, no dynamic "
         "allocation in the inner loop. The cost of a step does not depend on the "
         "data.",
         "Capture runs at 402 Hz in an Android foreground service with a wake lock. "
         "The integrator runs at that native rate; only the network runs at 10 Hz."),

        ("What happens when GNSS comes back? Does the position jump?",
         "There is no mode switch to recover from. The same filter runs the whole "
         "time, and GNSS is simply one more measurement that stops arriving and later "
         "resumes. When it resumes it is fused with its own variance, not snapped to.",
         "That is the direct answer to the seamless-handover requirement. Because "
         "nothing switches, there is no switching latency to tune and no discontinuity "
         "to smooth over. It is a property of the architecture rather than of "
         "parameter tuning."),

        ("How do you decide GNSS has been lost? Degraded is harder than absent.",
         "Absent is easy. Degraded is harder and more common - urban canyon multipath "
         "gives a confident, wrong fix. We use reported accuracy and satellite count "
         "to down-weight rather than a binary cutoff, since the filter takes a "
         "variance rather than a yes/no.",
         "Be honest: we have not systematically evaluated multipath rejection, because "
         "IO-VNBD is open-sky driving. It is a known untested area, not a solved one."),

        ("Can this run on ISRO's own hardware rather than a phone?",
         "Yes, and that is why the engine has no Android dependency. It already runs "
         "as a laptop command-line tool: give it a sensor file and it produces a "
         "trajectory. Better sensors would improve it directly, since our error budget "
         "is dominated by sensor-driven terms.",
         "What would need work is the input adapter for their file format, and "
         "retraining the speed model, because the vibration signature is "
         "platform-specific. The filter, the constraints and the map matching carry "
         "over unchanged."),

        ("What are the failure modes, and does the system know when it has failed?",
         "Four we know of: the phone moving mid-drive, alignment failing at start, the "
         "OS killing the capture service, and prolonged very smooth driving. The first "
         "three are detected and surfaced.",
         "The app runs a quality check on every recording and reports GOOD, WARN or "
         "BAD rather than looking healthy while producing garbage - we lost three "
         "recordings that way before adding it. The fourth, smooth driving, degrades "
         "quietly and we do not yet detect it. That is a real gap."),

        ("What if the phone is picked up or shifts during the drive?",
         "Alignment is invalidated from that moment and everything downstream is "
         "wrong. We detect sudden changes in the gravity vector and re-run alignment, "
         "and the app instructs users to wedge the phone.",
         "This is a genuine operational weakness. We assume a rigid mount. A handheld "
         "phone in a moving vehicle is outside what we can currently handle, and we "
         "would rather say that than claim robustness we have not demonstrated."),
    ]),

    # -------------------------------------------------------------- the gaps
    ("7.  WHAT WE DO NOT KNOW YET", [

        ("What have you not tested?",
         "Three things, plainly. On-device inference timing is not instrumented. "
         "Multipath and degraded-GNSS handling is not evaluated. And no full vehicle "
         "drive has been recorded and analysed end to end with our own app.",
         "If a real drive has been completed before you present, replace this answer "
         "with the actual numbers. Check with whoever ran it - do not guess."),

        ("What is the single biggest thing between you and the 10 percent spec?",
         "Attitude. Heading is 27.2 percent of the error budget against speed's 17.2, "
         "our analytic alignment measurably does not work, and the gap between median "
         "and 90th percentile points at distinct alignment failures rather than "
         "gradual degradation. Three independent lines of evidence, same conclusion.",
         "We were wrong about this for a long time. We believed speed dominated and "
         "spent effort there, and the measurement said otherwise. That is why we now "
         "re-measure the budget after every change rather than carrying a diagnosis "
         "forward."),

        ("What would you do with three more months?",
         "In order: a learned attitude head with 6D rotation output, supervised by "
         "GNSS-derived heading on data we already have. Then adaptive weighting "
         "between strapdown and network by blackout duration. Then re-tune window "
         "length at 400 Hz - OdoNet found 1 second optimal at 50 Hz, and our "
         "10-second window was chosen at 10 Hz, so the answer may invert.",
         "Notably NOT first on the list: more vehicles. OdoNet's controlled "
         "experiments found vehicle load and road condition have little impact while "
         "mounting angle dominates. More data was our top priority until we read that "
         "and checked our own alignment numbers against it."),

        ("What would make you abandon this approach?",
         "If the oracle-alignment experiment showed that perfect attitude barely "
         "improves drift, the whole diagnosis would be wrong and we would stop and "
         "re-measure rather than build on it.",
         "That experiment is designed and is the gate on our next phase. We do not "
         "intend to rewrite the model on a hypothesis - that is how the last three "
         "wrong diagnoses happened."),
    ]),
]

CLOSING = [
    ("THE THREE NUMBERS NOT TO OVERSTATE",
     "On-device inference latency - measured on a laptop, not a phone. Real-vehicle "
     "performance - not yet recorded. Multipath handling - not evaluated. If pushed "
     "on any of these, say it is not measured. An unmeasured number that turns out "
     "wrong puts every other number in this document in doubt."),
    ("IF YOU ARE ASKED SOMETHING NOT IN HERE",
     "Say what you would measure to find out. A panel asking a research question "
     "usually wants to know whether you think experimentally, not whether you have "
     "memorised an answer. 'We have not measured that - here is the experiment that "
     "would settle it' is a strong answer, not a weak one."),
    ("THE ARGUMENT UNDERNEATH ALL OF THIS",
     "We do not meet the spec yet and we say so. What we have is a measured error "
     "budget, a diagnosis that overturned our own assumption, and a specific next "
     "experiment that could falsify it. That is a working method, and it is worth "
     "more than a team quoting a number they cannot decompose."),
]
