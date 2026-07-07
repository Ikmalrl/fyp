%% =========================================================================
%  Phase3_EKF_Validation.m
%  Adaptive Extended Kalman Filter — SOC estimation on HWFET drive cycles
%  State: x = [SOC (%); V1 (V); V2 (V)]     Convention: discharge I < 0
%
%  WHAT CHANGED vs the old version:
%   1. PER-SAMPLE dt. The 25 degC HWFET file contains a 1240 s logging gap
%      at t=6 s holding 0.575 Ah (11.5% SOC). A fixed dt = mean(diff(t))
%      silently skipped it in both the EKF prediction and the "true SOC"
%      reference. Real dt fixes both.
%   2. GROUND TRUTH from the cycler's own Ah counter (meas.Ah), which
%      integrated through the gap. Counter resets (jumps back to ~0) are
%      detected and removed; legitimate large increments across logging
%      gaps are kept.
%   3. MEASURED CELL TEMPERATURE (meas.Battery_Temp_degC) drives every
%      lookup instead of a hardcoded test temperature. (The 0 degC file
%      actually STARTS at 24 degC and cools down in-file.)
%   4. SOC STATE CLAMP to [0, 100] after each update. Outside the lookup
%      table the OCV linearization is invalid: the query saturates but the
%      Jacobian keeps promising slope, which can run the estimate away
%      (observed: 146% SOC at 40 degC without the clamp).
%   5. ADAPTIVE MEASUREMENT NOISE (innovation-based, Mohamed & Schwarz).
%      When 2-RC model error grows (cold temperatures), R inflates
%      automatically and the filter leans on coulomb counting. This cut
%      the -10 degC RMSE roughly in half in prototyping.
%   6. Joseph-form covariance update (numerically stable), process noise
%      scaled by dt, and dOCV/dSOC read from the precomputed Phase 1b
%      gradient instead of on-line finite differencing.
%   7. Runs ALL FIVE temperature files and prints a summary table.
%
%  Reads : matlab output/OCV_SoC_results.mat   (Phase 1b)
%          matlab output/EKF_LookupTable.mat   (Phase 2)
%          driving cycle data/*.mat
% =========================================================================
clear; clc; close all;

ROOT = fileparts(mfilename('fullpath'));
ocv_data = load(fullfile(ROOT, 'matlab output', 'OCV_SoC_results.mat'));
lut_data = load(fullfile(ROOT, 'matlab output', 'EKF_LookupTable.mat'));
DRIVE_DIR = fullfile(ROOT, 'driving cycle data');

%% ------------------------------------------------------------------------
%  TUNING (validated in prototyping against all five files)
% -------------------------------------------------------------------------
CAP_NOM   = 5.0;          % Ah — same SOC convention as Phases 1/1b/2
SOC0      = 80;           % deliberately wrong initial guess (true ~100%)
P0        = diag([100, 1e-4, 1e-4]);   % initial covariance (sigma_SOC = 10%)
Q_RATE    = diag([1e-7, 1e-8, 1e-8]);  % process noise PER SECOND (x dt)
R_FLOOR   = 1e-4;         % V^2 — minimum measurement noise variance
R_ADAPT_WIN = 200;        % innovation window for adaptive R
T_CONV    = 300;          % s — metrics evaluated after this settling time

%% ------------------------------------------------------------------------
%  FAST LOOKUPS (griddedInterpolant needs ascending grid vectors)
% -------------------------------------------------------------------------
ocvT  = double(ocv_data.OCV_TEMPS(:));      % ascending from Phase 1b
socD  = double(ocv_data.SOC_DENSE(:));
F_ocv  = griddedInterpolant({ocvT, socD}, ocv_data.OCV_grid,  'linear', 'nearest');
F_docv = griddedInterpolant({ocvT, socD}, ocv_data.dOCV_grid, 'linear', 'nearest');

lutT = double(lut_data.LUT.Temp_vec(:));
lutS = double(lut_data.LUT.SOC_vec(:));
F_R0 = griddedInterpolant({lutT, lutS}, lut_data.LUT.R0_grid,   'linear', 'nearest');
F_R1 = griddedInterpolant({lutT, lutS}, lut_data.LUT.R1_grid,   'linear', 'nearest');
F_t1 = griddedInterpolant({lutT, lutS}, lut_data.LUT.tau1_grid, 'linear', 'nearest');
F_R2 = griddedInterpolant({lutT, lutS}, lut_data.LUT.R2_grid,   'linear', 'nearest');
F_t2 = griddedInterpolant({lutT, lutS}, lut_data.LUT.tau2_grid, 'linear', 'nearest');
Tmin = min(lutT); Tmax = max(lutT);

%% ------------------------------------------------------------------------
%  RUN THE EKF ON EVERY DRIVE-CYCLE FILE
% -------------------------------------------------------------------------
files = dir(fullfile(DRIVE_DIR, '*.mat'));
assert(~isempty(files), 'No drive cycle files found in %s', DRIVE_DIR);

summary = [];
for fi = 1:numel(files)
    fname = files(fi).name;
    tok = regexp(fname, '(-?\d+)degC', 'tokens', 'once');
    tempLabel = str2double(tok{1});

    raw = load(fullfile(files(fi).folder, fname));
    t  = double(raw.meas.Time(:));
    vM = double(raw.meas.Voltage(:));
    iM = double(raw.meas.Current(:));
    ah = double(raw.meas.Ah(:));
    tc = double(raw.meas.Battery_Temp_degC(:));
    N  = numel(t);

    % ---- ground truth from the cycler Ah counter -----------------------
    dAh = diff(ah);
    isReset = abs(dAh) > 0.1 & abs(ah(2:end)) < 0.01;   % jump back to ~0
    dAh(isReset) = 0;
    true_soc = 100 + [0; cumsum(dAh)] / CAP_NOM * 100;

    % ---- EKF ------------------------------------------------------------
    x = [SOC0; 0; 0];
    P = P0;
    Rm = R_FLOOR;
    soc_est = zeros(N,1);  soc_est(1) = SOC0;
    v_pred  = nan(N,1);
    Tq = min(max(tc, Tmin), Tmax);      % clip temp to table range

    for k = 2:N
        dtk = t(k) - t(k-1);
        Ik1 = iM(k-1);  Ik = iM(k);  Tk = Tq(k);

        % -- parameters at current state ----------------------------------
        socq = min(100, max(0, x(1)));
        R0 = F_R0(Tk, socq);  R1 = F_R1(Tk, socq);  tau1 = F_t1(Tk, socq);
        R2 = F_R2(Tk, socq);  tau2 = F_t2(Tk, socq);
        a1 = exp(-dtk/tau1);  a2 = exp(-dtk/tau2);

        % -- predict --------------------------------------------------------
        soc_p = x(1) + Ik1*dtk/(CAP_NOM*3600)*100;
        v1_p  = a1*x(2) + R1*(1-a1)*Ik1;
        v2_p  = a2*x(3) + R2*(1-a2)*Ik1;
        F  = diag([1, a1, a2]);
        Pp = F*P*F' + Q_RATE*dtk;

        % -- measurement prediction ----------------------------------------
        socc = min(100, max(0, soc_p));
        ocv  = F_ocv(Tk, socc);
        dv   = F_docv(Tk, socc);
        vhat = ocv + R0*Ik + v1_p + v2_p;

        % -- update ---------------------------------------------------------
        H = [dv, 1, 1];
        S = H*Pp*H' + Rm;
        K = (Pp*H') / S;
        innov = vM(k) - vhat;
        x = [soc_p; v1_p; v2_p] + K*innov;
        x(1) = min(100, max(0, x(1)));            % state clamp (see header)
        IKH = eye(3) - K*H;
        P = IKH*Pp*IKH' + K*Rm*K';                % Joseph form

        % -- innovation-adaptive measurement noise ---------------------------
        Rm = max(R_FLOOR, (1 - 1/R_ADAPT_WIN)*Rm + ...
                          (1/R_ADAPT_WIN)*(innov^2 - H*Pp*H'));

        soc_est(k) = x(1);
        v_pred(k)  = vhat;
    end

    % ---- metrics (after settling) ---------------------------------------
    m = t > T_CONV;
    err  = soc_est(m) - true_soc(m);
    rmse = sqrt(mean(err.^2));
    mae  = mean(abs(err));
    emax = max(abs(err));
    vrms = sqrt(mean((v_pred(m) - vM(m)).^2, 'omitnan')) * 1000;
    efin = soc_est(end) - true_soc(end);
    summary = [summary; tempLabel, rmse, mae, emax, vrms, efin]; %#ok<AGROW>

    fprintf('%6.0f degC | SOC RMSE %5.2f%% | MAE %5.2f%% | max %5.2f%% | V RMSE %6.1f mV | final err %+5.2f%%\n', ...
        tempLabel, rmse, mae, emax, vrms, efin);

    % ---- plots ------------------------------------------------------------
    figure('Name', sprintf('EKF — %d degC', tempLabel), ...
           'Position', [60+20*fi, 60+20*fi, 1100, 780]);

    subplot(3,1,1);
    plot(t/3600, true_soc, 'k-', 'LineWidth', 1.6); hold on;
    plot(t/3600, soc_est, 'g--', 'LineWidth', 1.4);
    ylabel('SOC (%)'); grid on;
    legend('True SOC (cycler Ah counter)', 'Adaptive EKF', 'Location','northeast');
    title(sprintf('HWFET %d\\circC — SOC tracking (init guess %d%%, true 100%%)', ...
        tempLabel, SOC0), 'FontWeight','bold');

    subplot(3,1,2);
    plot(t/3600, soc_est - true_soc, 'r-', 'LineWidth', 1.0);
    yline(0, 'k:');
    ylabel('SOC error (%)'); grid on;
    title(sprintf('RMSE after %d s settling: %.2f%%', T_CONV, rmse));

    subplot(3,1,3);
    plot(t/3600, vM, 'b-', 'LineWidth', 0.8); hold on;
    plot(t/3600, v_pred, 'r--', 'LineWidth', 0.8);
    xlabel('Time (h)'); ylabel('Voltage (V)'); grid on;
    legend('Measured', 'EKF predicted', 'Location','northeast');
    title(sprintf('Terminal voltage tracking (RMSE %.1f mV)', vrms));
end

%% ------------------------------------------------------------------------
%  SUMMARY
% -------------------------------------------------------------------------
summary = sortrows(summary, 1, 'descend');
results_table = array2table(summary, 'VariableNames', ...
    {'Temp_degC','SOC_RMSE_pct','SOC_MAE_pct','SOC_MaxErr_pct', ...
     'V_RMSE_mV','FinalErr_pct'});
disp(' '); disp('===== Adaptive EKF validation summary ====='); disp(results_table);

save(fullfile(ROOT, 'matlab output', 'EKF_results.mat'), 'results_table');
fprintf('Saved: %s\n', fullfile(ROOT, 'matlab output', 'EKF_results.mat'));
