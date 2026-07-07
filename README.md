**PROJECT OVERVIEW:**
- **Goal:** Dynamic State of Charge (SOC) estimation of a lithium-ion battery using an Extended Kalman Filter (EKF) based on a Second-Order RC Equivalent Circuit Model (2-RC ECM).
- **Cell:** Turnigy Graphene 5000mAh 65C (Li-ion).
- **Dataset:** Public dataset by Dr. Phillip Kollmeyer (McMaster University). Tested on a high-precision Digatron cycler (0.1% accuracy). Includes HPPC pulse tests, C/20 discharges, and dynamic EV drive cycles (UDDS, HWFET) across temperatures from -10°C to 40°C.
- **Software:** MATLAB (Pure scripting only. NO Simulink).

**WHAT HAS ALREADY BEEN ACCOMPLISHED (The Pipeline):**
- **Phase 1 (Parameter Extraction):** Extracted internal parameters (R0, R1, C1, R2, C2) from HPPC pulse relaxation data across multiple temperatures. 
- **Phase 1b (OCV-SOC Curve):** Used Modified Akima interpolation (`makima`) on HPPC resting voltages to create a mathematically perfectly smooth OCV-SOC lookup table (avoided traditional polynomial/Gaussian overfitting).
- **Phase 2 (Machine Learning Surrogate):** Trained a Random Forest Regressor (`TreeBagger`, 500 trees) on the extracted HPPC parameters. Generated a dense, continuous 2D lookup grid (SOC vs. Temperature) for R0, R1, C1, R2, C2 so the EKF can query parameters instantly without doing real-time ML computation.
- **Phase 3 (EKF Execution):** Built the discrete-time EKF loop. State vector: x = [SOC, V1, V2]^T. The EKF actively queries the Phase 2 Random Forest grids and Phase 1b Makima curve at every time step to recalculate State-Space matrices (A, B, C, D). It calculates the H-matrix Jacobian numerically.
- **Results:** The EKF successfully tracks SOC. Under 25°C HWFET, it achieved ~1.5% RMSE against the lab's Coulomb counting baseline. At -10°C, it successfully outperformed standard Coulomb counting by dynamically detecting voltage sag and capacity fade, tracking SOC to 0% at the 3.0V cutoff.
