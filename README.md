# SOC Estimation of Li-Ion Batteries using an EKF

**PROJECT OVERVIEW:**
- **Goal:** Dynamic State of Charge (SOC) estimation of a lithium-ion battery using an Extended Kalman Filter (EKF) based on a Second-Order RC Equivalent Circuit Model (2-RC ECM).
- **Cell:** Turnigy Graphene 5000mAh 65C (Li-ion).
- **Dataset:** Public dataset by Dr. Phillip Kollmeyer (McMaster University). Tested on a high-precision Digatron cycler. Includes HPPC pulse tests and dynamic EV drive cycles (HWFET) across temperatures from -10°C to 40°C.
- **Software:** MATLAB (pure scripting, no Simulink). Requires Optimization Toolbox (`lsqnonlin`) and Statistics & Machine Learning Toolbox (`fitrgp`, `TreeBagger`).
- **SOC convention:** SOC = 100% at full charge; coulomb counted on the 5.0 Ah nominal capacity in every phase. (Measured usable capacity is ~4.7 Ah at 25°C, ~4.4 Ah at -10°C; Phase 1 reports it per temperature.)

## Pipeline (run in this order — each phase saves inputs for the next into `matlab output/`)

### Phase 1 — `Phase1_HPPC_Extraction.m`
Parameter identification from HPPC data, adapted to this dataset's actual protocol:
- The protocol has ~12 SOC setpoints, each with **four discharge pulses at different rates** (-5/-10/-25/-50 A, 10 s each), only **30 s of clean rest** after each pulse (then a +0.5 A recharge starts), and long logging gaps during the -1.5 A SOC-adjustment segments.
- Pulses are **grouped by SOC**; parameters come from the **1C (-5 A) pulse** of every group (consistent rate, matches drive-cycle current levels); each row stores the **actual coulomb-counted SOC**, never a target label.
- 2-RC parameters `[R0, R1, tau1, R2, tau2]` are identified by **jointly fitting the pulse and its relaxation** (simulating the ECM ODE against the measured current), giving sub-mV fit residuals at 25°C. No post-fit clamping — questionable fits are flagged instead.
- OCV points come from the long rest before each group's first pulse, **plus a low-SOC anchor** from the rest after the final discharge to cutoff (this captures the steep OCV tail below 10% SOC).

### Phase 1b — `Phase1b_OCV_SOC.m`
Builds the OCV(SOC, T) surface with **PCHIP** (shape-preserving, no overshoot next to the steep tail), linear extension to 0% SOC, and a precomputed **dOCV/dSOC gradient grid** for the EKF Jacobian.

### Phase 2 — `Phase2_ML_LookupTable.m`
Machine-learning parameter surfaces over (SOC, T) using **Gaussian Process Regression** (ARD squared-exponential kernel, log-space targets). GPR suits the ~60-point training set: smooth surfaces, graceful interpolation between the five measured temperatures, and uncertainty bands.

### Phase 3 — `Phase3_EKF_Validation.m`
Adaptive EKF (state `[SOC; V1; V2]`) validated on all five HWFET files:
- **Per-sample dt** (the 25°C file has a 1240 s logging gap holding 11.5% SOC — a fixed dt corrupts both the EKF and the reference).
- **Ground truth from the cycler's Ah counter** (reset-aware), not re-integrated current.
- **Measured cell temperature** drives every lookup (the "0°C" file starts at 24°C).
- **SOC state clamped** to the table range (prevents runaway when the OCV lookup saturates).
- **Innovation-adaptive measurement noise**: when 2-RC model error grows in the cold, R inflates and the filter automatically leans on coulomb counting.
- Joseph-form covariance update; deliberately wrong initial guess (80% vs true 100%) to demonstrate convergence.

## Validated results (Python mirror of this exact pipeline, wrong 80% init, metrics after 300 s settling)

| Temp | SOC RMSE | Max err | Voltage RMSE |
|------|----------|---------|--------------|
| 40°C | 0.98% | 1.1% | 2.1 mV |
| 25°C | 0.62% | 1.4% | 2.7 mV |
| 10°C | 0.86% | 2.0% | 18 mV |
| 0°C  | 1.67% | 2.7% | 8 mV |
| -10°C | 4.20% | 5.5% | 83 mV |

The -10°C case is limited by the linear 2-RC model itself (large voltage residuals in the cold), which the adaptive R turns into graceful degradation rather than divergence.

## Repository layout
- `hppc/` — HPPC test files (one per temperature)
- `driving cycle data/` — HWFET drive-cycle files (one per temperature)
- `matlab output/` — generated `.mat` artifacts (`HPPC_params.mat`, `OCV_SoC_results.mat`, `EKF_LookupTable.mat`, `EKF_results.mat`). Regenerate by running Phases 1 → 1b → 2 → 3; outputs from the old pipeline are not compatible with the new Phase 3.
