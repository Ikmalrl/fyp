%% =========================================================================
%  Phase3_EKF_Validation.m
%  Extended Kalman Filter for 2-RC ECM Parameter Validation
%  Turnigy Graphene 5000mAh 65C Cell
% =========================================================================
clear; clc; close all;

%% -------------------------------------------------------------------------
%  1. LOAD PHASE 1b & PHASE 2 RESULTS (The Brain)
% -------------------------------------------------------------------------
fprintf('Loading Lookup Tables...\n');
try
    ocv_data = load('C:\Users\ekmal\Documents\fyp\laboratory\matlab output\OCV_SoC_results.mat');
    lut_data = load('C:\Users\ekmal\Documents\fyp\laboratory\matlab output\EKF_LookupTable.mat');
catch
    error('Cannot find your .mat files. Check the file paths, bro.');
end

% Extract grids for cleaner code below
SOC_grid_ocv  = ocv_data.SOC_DENSE;
Temp_grid_ocv = ocv_data.valid_temps;
OCV_surf      = ocv_data.OCV_grid;

SOC_grid_rc   = lut_data.LUT.SOC_vec;
Temp_grid_rc  = lut_data.LUT.Temp_vec;
R0_surf = lut_data.LUT.R0_grid;
R1_surf = lut_data.LUT.R1_grid;
C1_surf = lut_data.LUT.C1_grid;
R2_surf = lut_data.LUT.R2_grid;
C2_surf = lut_data.LUT.C2_grid;

%% -------------------------------------------------------------------------
%  2. LOAD REAL DRIVE CYCLE DATA (UDDS)
% -------------------------------------------------------------------------
fprintf('Loading UDDS Drive Cycle...\n');

% Load your specific 25degC UDDS file here!
raw_data = load('C:\Users\ekmal\Documents\fyp\laboratory\driving cycle data\04-09-19_16.42 763_HWFET_40degC_Turnigy_Graphene.mat');

% Assuming the data is stored in a struct called 'meas' like your HPPC files
time    = raw_data.meas.Time;
current = raw_data.meas.Current;
V_meas  = raw_data.meas.Voltage;

% The cycler might not be exactly 1.0 seconds. We calculate the real dt.
dt = mean(diff(time)); 
fprintf('  Calculated time step (dt) = %.3f seconds\n', dt);

% Total steps
N = length(time);

% True SOC (Coulomb Counting from the cycler's exact current)
true_soc = zeros(size(time));
true_soc(1) = 100; % Assuming the drive cycle started fully charged at 100%
C_nom = 5.0; % Ah

for k = 2:N
    true_soc(k) = true_soc(k-1) + (current(k-1) * dt) / (C_nom * 3600) * 100;
end

%% -------------------------------------------------------------------------
%  3. EKF INITIALISATION
% -------------------------------------------------------------------------
fprintf('Initialising EKF Matrix Math...\n');

N = length(time);
Temp_test = 25; % Running the test at 25 degrees C

% State Vector: x = [SOC(%); V1(V); V2(V)]
x_est = zeros(3, N);
x_est(:,1) = [80; 0; 0]; % INTENTIONAL WRONG GUESS! Guessing 80% instead of 90%

% Covariance Matrix P (Trust in our current state)
P = diag([1e-2, 1e-3, 1e-3]); 

% Process Noise Q (Trust in the math model)
% SOC changes slowly, V1/V2 change fast
Q = diag([1e-6, 1e-4, 1e-4]); 

% Measurement Noise R (Trust in the voltage sensor)
R = 1e-2; 

% Storage for plotting
V_pred_log = zeros(1, N);

%% -------------------------------------------------------------------------
%  4. THE MAIN EKF LOOP (Where the magic happens)
% -------------------------------------------------------------------------
fprintf('Running EKF Loop... don''t let the MX250 melt...\n');

for k = 2:N
    
    % --- Step A: Read inputs ---
    I_k1 = current(k-1);
    soc_prev = x_est(1, k-1);
    v1_prev  = x_est(2, k-1);
    v2_prev  = x_est(3, k-1);
    
    % Clamp SOC for lookup tables so interp2 doesn't crash
    soc_query = max(0, min(100, soc_prev)); 
    
    % --- Step B: Dynamically Query Parameters ---
    OCV = interp2(SOC_grid_ocv, Temp_grid_ocv, OCV_surf, soc_query, Temp_test);
    R0  = interp2(SOC_grid_rc, Temp_grid_rc, R0_surf, soc_query, Temp_test);
    R1  = interp2(SOC_grid_rc, Temp_grid_rc, R1_surf, soc_query, Temp_test);
    C1  = interp2(SOC_grid_rc, Temp_grid_rc, C1_surf, soc_query, Temp_test);
    R2  = interp2(SOC_grid_rc, Temp_grid_rc, R2_surf, soc_query, Temp_test);
    C2  = interp2(SOC_grid_rc, Temp_grid_rc, C2_surf, soc_query, Temp_test);
    
    % Prevent division by zero if C gets weirdly small
    C1 = max(C1, 10); C2 = max(C2, 10);
    
    % --- Step C: PREDICT STEP (The Math Model) ---
    % 1. Predict next state x_{k|k-1}
    soc_pred = soc_prev + (I_k1 * dt) / (C_nom * 3600) * 100;
    
    term1 = exp(-dt / (R1 * C1));
    v1_pred = v1_prev * term1 + R1 * (1 - term1) * I_k1;
    
    term2 = exp(-dt / (R2 * C2));
    v2_pred = v2_prev * term2 + R2 * (1 - term2) * I_k1;
    
    x_pred = [soc_pred; v1_pred; v2_pred];
    
    % 2. Calculate Process Jacobian (F)
    F = eye(3);
    F(2,2) = term1;
    F(3,3) = term2;
    
    % 3. Predict Covariance P_{k|k-1}
    P_pred = F * P * F' + Q;
    
    % --- Step D: PREDICT MEASUREMENT ---
    % Re-query OCV at the new predicted SOC
    soc_pred_clamp = max(0, min(100, soc_pred));
    OCV_pred = interp2(SOC_grid_ocv, Temp_grid_ocv, OCV_surf, soc_pred_clamp, Temp_test);
    
    % V_terminal = OCV + (I * R0) + V1 + V2 
    % Note: Current is negative, so adding (I*R0) drops the voltage correctly
    V_hat = OCV_pred + (current(k) * R0) + v1_pred + v2_pred;
    V_pred_log(k) = V_hat;
    
    % --- Step E: UPDATE STEP (The Reality Check) ---
    % 1. Calculate Measurement Jacobian (H)
    % We need d(OCV)/d(SOC). We approximate it numerically using a tiny step!
    delta_soc = 0.1;
    ocv_plus  = interp2(SOC_grid_ocv, Temp_grid_ocv, OCV_surf, min(100, soc_pred_clamp + delta_soc), Temp_test);
    ocv_minus = interp2(SOC_grid_ocv, Temp_grid_ocv, OCV_surf, max(0, soc_pred_clamp - delta_soc), Temp_test);
    dOCV_dSOC = (ocv_plus - ocv_minus) / (2 * delta_soc);
    
    H = [dOCV_dSOC, 1, 1];
    
    % 2. Calculate Kalman Gain (K)
    S = H * P_pred * H' + R;
    K = P_pred * H' / S;
    
    % 3. Calculate Innovation (Error)
    Error = V_meas(k) - V_hat;
    
    % 4. Update State and Covariance
    x_est(:, k) = x_pred + K * Error;
    P = (eye(3) - K * H) * P_pred;
    
end
fprintf('EKF Loop Complete!\n');

%% -------------------------------------------------------------------------
%  5. PLOT THE RESULTS
% -------------------------------------------------------------------------
figure('Name', 'EKF Validation', 'Position', [100, 100, 1200, 800]);

% Plot 1: Voltage Tracking
subplot(2,1,1);
plot(time, V_meas, 'b-', 'LineWidth', 1.5); hold on;
plot(time, V_pred_log, 'r--', 'LineWidth', 1.5);
title('EKF Terminal Voltage Tracking');
ylabel('Voltage (V)');
legend('Measured Voltage', 'EKF Predicted Voltage');
grid on;

% Plot 2: SOC Tracking
subplot(2,1,2);
plot(time, true_soc, 'k-', 'LineWidth', 2); hold on;
plot(time, x_est(1,:), 'g--', 'LineWidth', 2);
title('State of Charge (SOC) Estimation');
ylabel('SOC (%)'); xlabel('Time (s)');
legend('True SOC (Coulomb Counting)', 'EKF Estimated SOC');
grid on;

% Note: Look at the start of the green line. It should start at 80% 
% and aggressively snap up to the black line (90%) because the EKF 
% realizes the voltage error is huge and corrects itself. That's the magic.