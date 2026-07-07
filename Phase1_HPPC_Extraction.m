%% =========================================================================
%  Phase1_HPPC_Extraction.m
%  2-RC ECM Parameter Identification from HPPC Data  (rewritten)
%  Turnigy Graphene 5000mAh 65C Cell — Kollmeyer dataset (McMaster)
%
%  WHAT CHANGED vs the old version (and why):
%   1. The HPPC protocol in this dataset has ~12 SOC setpoints, each with
%      FOUR discharge pulses at different rates (-5/-10/-25/-50 A, 10 s).
%      The old nearest-target matching grabbed pulses from the wrong SOC
%      group (e.g. "2.5%" and "5%" rows were really 9.9% data) and mixed
%      C-rates randomly. We now group pulses properly, use ONE consistent
%      rate (1C = -5 A, closest to drive-cycle currents), and store the
%      ACTUAL coulomb-counted SOC of each pulse, never the target label.
%   2. Only 30 s of true rest follows each discharge pulse (then a +0.5 A
%      recharge starts). Relaxation-only fitting cannot identify a slow
%      time constant from that, so we JOINTLY fit the 10 s pulse AND the
%      30 s relaxation by simulating the 2-RC ODE against the measured
%      current profile. tau2 is bounded to [6, 80] s — what this window
%      can genuinely support, and what matters for drive-cycle dynamics.
%   3. No post-fit clamping of R/C values (the old caps silently rewrote
%      tau2 from ~100 s to ~15 s). Bad fits are FLAGGED instead.
%   4. OCV is taken only from the long (~10 min) rest before the FIRST
%      pulse of each group, plus one low-SOC anchor point from the rest
%      after the final discharge to cutoff — this recovers the steep OCV
%      tail below 10% SOC that the old table missed entirely.
%   5. Coulomb counting uses per-sample dt (backward rectangle), which is
%      also correct across the logger's long gaps during the -1.5 A
%      SOC-adjustment segments (current after a gap equals the current
%      during it).
%
%  Outputs (saved to  <repo>/matlab output/HPPC_params.mat):
%    params_table : [Temp, SOC, Ipulse, OCV, R0, R1, C1, R2, C2,
%                    tau1, tau2, RMSE_mV, RelaxWin_s, Flag]
%    ocv_table    : [Temp, SOC, OCV]  (group rests + low-SOC anchors)
%    capacity_table : usable Ah measured per temperature
% =========================================================================
clear; clc; close all;

%% ------------------------------------------------------------------------
%  CONFIGURATION (paths are relative to this script's folder)
% -------------------------------------------------------------------------
ROOT        = fileparts(mfilename('fullpath'));
DATA_FOLDER = fullfile(ROOT, 'hppc');
OUT_FOLDER  = fullfile(ROOT, 'matlab output');
if ~exist(OUT_FOLDER, 'dir'), mkdir(OUT_FOLDER); end
SAVE_PATH   = fullfile(OUT_FOLDER, 'HPPC_params.mat');

TEMP_LABELS = {'40degC', '25degC', '10degC', '0degC', '-10degC'};
TEMP_VALUES = [40, 25, 10, 0, -10];

CAP_NOM       = 5.0;    % Ah, nominal — SOC convention used in ALL phases
PULSE_THRESH  = -2.0;   % A, discharge pulse detection threshold
MIN_PULSE_DUR = 5.0;    % s   (pulses are 10 s)
MAX_PULSE_DUR = 30.0;   % s   (excludes the long -1.5 A adjustment segments)
RATE_SELECT   = -5.0;   % A — fit the 1C pulse (drive cycles run at ~1C)
QUIET_I       = 0.10;   % A, |I| below this counts as rest
RELAX_MAX     = 120.0;  % s, cap on relaxation window (real rest is ~30 s)
RELAX_MIN     = 15.0;   % s, minimum usable relaxation
GROUP_SOC_GAP = 2.0;    % % SOC separating pulse groups
RMSE_FLAG_MV  = 2.0;    % fits worse than this get flagged 'check'

OPT = optimoptions('lsqnonlin', 'Display','off', ...
    'MaxFunctionEvaluations', 6000, 'FunctionTolerance', 1e-10);

%% ------------------------------------------------------------------------
%  MAIN LOOP over temperatures
% -------------------------------------------------------------------------
results  = [];             % numeric rows for params_table
flags    = strings(0,1);
ocv_rows = [];             % [Temp, SOC, OCV]
cap_rows = [];             % [Temp, usable Ah]

for tIdx = 1:numel(TEMP_LABELS)
    tempLabel = TEMP_LABELS{tIdx};
    tempVal   = TEMP_VALUES(tIdx);

    fprintf('\n==============================================\n');
    fprintf(' Processing: %s\n', tempLabel);
    fprintf('==============================================\n');

    fileList = dir(fullfile(DATA_FOLDER, sprintf('*HPPC*%s*.mat', tempLabel)));
    if isempty(fileList)
        fprintf('  [WARN] No HPPC file for %s — skipping.\n', tempLabel);
        continue
    end
    raw  = load(fullfile(fileList(1).folder, fileList(1).name));
    meas = raw.meas;
    t = double(meas.Time(:));
    v = double(meas.Voltage(:));
    i = double(meas.Current(:));
    fprintf('  Loaded: %s (%d samples, %.1f h)\n', ...
        fileList(1).name, numel(t), t(end)/3600);

    % ---- Coulomb counting (backward rectangle, gap-safe) ----------------
    dt    = [0; diff(t)];
    cumAh = cumsum(i .* dt / 3600);
    soc   = 100 + cumAh / CAP_NOM * 100;

    % ---- Detect discharge pulses (5-30 s below -2 A) --------------------
    below  = i < PULSE_THRESH;
    dEdge  = diff(below);
    starts = find(dEdge == 1) + 1;
    ends   = find(dEdge == -1);          % last sample still below threshold
    if below(1),   starts = [1; starts]; end
    if below(end), ends   = [ends; numel(t)]; end
    keep = false(size(starts));
    for k = 1:numel(starts)
        dur = t(ends(k)) - t(starts(k));
        keep(k) = dur >= MIN_PULSE_DUR && dur <= MAX_PULSE_DUR;
    end
    starts = starts(keep);  ends = ends(keep);
    fprintf('  Detected %d discharge pulses\n', numel(starts));

    % ---- Group pulses by SOC (each group = one HPPC setpoint) -----------
    groupID = zeros(size(starts));
    g = 0; lastSOC = inf;
    for k = 1:numel(starts)
        sHere = soc(max(1, starts(k)-2));
        if abs(sHere - lastSOC) > GROUP_SOC_GAP, g = g + 1; end
        groupID(k) = g;  lastSOC = sHere;
    end
    nGroups = g;
    fprintf('  Grouped into %d SOC setpoints\n', nGroups);

    figure('Name', sprintf('HPPC joint fits — %s', tempLabel), ...
           'Position', [80, 80, 1500, 900]);
    nCols = 4;  nRows = ceil(nGroups/nCols);  plotIdx = 0;

    for gi = 1:nGroups
        gk  = find(groupID == gi);
        ps0 = starts(gk(1));

        % ---- OCV from the long rest before the group's first pulse ------
        preMask = t >= t(ps0)-10 & t < t(ps0)-0.05;
        if nnz(preMask) < 3, continue; end
        OCV   = mean(v(preMask));
        socG  = soc(max(1, ps0-2));
        ocv_rows = [ocv_rows; tempVal, socG, OCV]; %#ok<AGROW>

        % ---- Select the pulse closest to RATE_SELECT ---------------------
        meds = arrayfun(@(a,b) median(i(a:b)), starts(gk), ends(gk));
        [rateErr, sel] = min(abs(meds - RATE_SELECT));
        if rateErr > 2.5, continue; end           % selected rate missing
        ps = starts(gk(sel));  pe = ends(gk(sel));  Ip = meds(sel);

        % ---- Fit segment: 1.5 s pre-pulse .. end of clean rest ----------
        busy = find(t > t(pe)+0.5 & abs(i) >= QUIET_I, 1, 'first');
        segEndT = t(pe) + RELAX_MAX;
        if ~isempty(busy), segEndT = min(segEndT, t(busy) - 0.5); end
        idx  = find(t >= t(ps)-1.5 & t <= segEndT);
        tseg = t(idx);  iseg = i(idx);  vseg = v(idx);
        relaxSpan = segEndT - t(pe);
        if relaxSpan < RELAX_MIN
            fprintf('  [WARN] SOC %.1f%% — relax window %.0f s too short, skipped\n', ...
                socG, relaxSpan);
            continue
        end

        % ---- Joint 2-RC fit: theta = [R0, R1, tau1, R2, tau2] -----------
        Vpre  = mean(vseg(tseg < t(ps)-0.05));
        R0g   = max(2e-4, (Vpre - v(ps+1)) / abs(Ip));   % edge-based guess
        x0 = [R0g,  0.3*R0g, 1.5, 0.5*R0g, 25.0];
        lb = [1e-4, 1e-5,    0.1, 1e-5,     6.0];
        ub = [2e-2, 5e-2,    6.0, 1e-1,    80.0];
        resFun = @(th) ecm2rc_sim(th, tseg, iseg, OCV) - vseg;
        try
            [theta, resnorm] = lsqnonlin(resFun, x0, lb, ub, OPT);
        catch ME
            fprintf('  [WARN] SOC %.1f%% — fit failed: %s\n', socG, ME.message);
            continue
        end
        R0 = theta(1); R1 = theta(2); tau1 = theta(3);
        R2 = theta(4); tau2 = theta(5);
        C1 = tau1/R1;  C2 = tau2/R2;
        rmse_mV = sqrt(resnorm/numel(vseg)) * 1000;

        atBound = tau2 > 0.99*ub(5) || tau1 < 1.1*lb(3) || R2 > 0.99*ub(4);
        if rmse_mV < RMSE_FLAG_MV && ~atBound, flag = "ok"; else, flag = "check"; end

        fprintf('  SOC %5.1f%% | I=%5.1fA | R0=%.3fmO R1=%.3fmO tau1=%4.1fs R2=%.3fmO tau2=%5.1fs | RMSE=%.2f mV [%s]\n', ...
            socG, Ip, R0*1e3, R1*1e3, tau1, R2*1e3, tau2, rmse_mV, flag);

        results = [results; tempVal, socG, Ip, OCV, R0, R1, C1, R2, C2, ...
                   tau1, tau2, rmse_mV, relaxSpan]; %#ok<AGROW>
        flags(end+1,1) = flag; %#ok<SAGROW>

        % ---- Plot measured vs simulated -----------------------------------
        plotIdx = plotIdx + 1;
        subplot(nRows, nCols, plotIdx);
        plot(tseg - tseg(1), vseg, 'b-', 'LineWidth', 1.1); hold on;
        plot(tseg - tseg(1), ecm2rc_sim(theta, tseg, iseg, OCV), 'r--', 'LineWidth', 1.2);
        title(sprintf('SOC=%.1f%%  %.2f mV', socG, rmse_mV), 'FontSize', 8);
        xlabel('t (s)'); ylabel('V'); grid on;
        if plotIdx == 1, legend('Measured','2-RC fit','Location','southeast','FontSize',6); end
    end
    sgtitle(sprintf('HPPC joint pulse+relaxation fits — %s (I = %g A)', ...
        tempLabel, RATE_SELECT), 'FontWeight', 'bold');

    % ---- Low-SOC OCV anchor: rest after the deepest discharge -----------
    [minAh, jmin] = min(cumAh);
    restIdx = find(t > t(jmin) & abs(i) < 0.05);
    if ~isempty(restIdx)
        % contiguous rest starting at restIdx(1)
        r0 = restIdx(1);
        rEnd = r0 + find(abs(i(r0:end)) >= 0.05, 1, 'first') - 2;
        if isempty(rEnd), rEnd = numel(t); end
        if t(rEnd) - t(r0) >= 60
            socAnchor = 100 + minAh/CAP_NOM*100;
            ocvAnchor = v(rEnd);      % last sample = most relaxed
            ocv_rows  = [ocv_rows; tempVal, socAnchor, ocvAnchor]; %#ok<AGROW>
            fprintf('  Low-SOC OCV anchor: %.1f%% -> %.3f V (rest %.0f s)\n', ...
                socAnchor, ocvAnchor, t(rEnd)-t(r0));
        end
    end
    cap_rows = [cap_rows; tempVal, -minAh]; %#ok<AGROW>
    fprintf('  Usable capacity at %s: %.3f Ah\n', tempLabel, -minAh);
end

%% ------------------------------------------------------------------------
%  BUILD AND SAVE TABLES
% -------------------------------------------------------------------------
if isempty(results), error('No parameters extracted — check data paths.'); end

params_table = array2table(results, 'VariableNames', ...
    {'Temp_degC','SOC_pct','Ipulse_A','OCV','R0','R1','C1','R2','C2', ...
     'tau1','tau2','RMSE_mV','RelaxWin_s'});
params_table.Flag = flags;
params_table = sortrows(params_table, {'Temp_degC','SOC_pct'}, {'ascend','ascend'});

ocv_table = array2table(ocv_rows, 'VariableNames', {'Temp_degC','SOC_pct','OCV'});
ocv_table = sortrows(ocv_table, {'Temp_degC','SOC_pct'}, {'ascend','ascend'});

capacity_table = array2table(cap_rows, 'VariableNames', {'Temp_degC','UsableAh'});

disp(' '); disp('===== Extracted parameters ====='); disp(params_table);
disp('===== Usable capacity per temperature ====='); disp(capacity_table);

save(SAVE_PATH, 'params_table', 'ocv_table', 'capacity_table', ...
     'TEMP_VALUES', 'CAP_NOM', 'RATE_SELECT');
fprintf('\nSaved: %s\n', SAVE_PATH);

%% ------------------------------------------------------------------------
%  SUMMARY PLOT — parameters vs SOC for all temperatures
% -------------------------------------------------------------------------
pNames  = {'R0 (\Omega)','R1 (\Omega)','\tau_1 (s)','R2 (\Omega)','\tau_2 (s)'};
pFields = {'R0','R1','tau1','R2','tau2'};
colors  = lines(numel(TEMP_VALUES));

figure('Name','2-RC parameters vs SOC','Position',[150,150,1400,700]);
for p = 1:5
    subplot(2,3,p); hold on;
    for tIdx = 1:numel(TEMP_VALUES)
        m = params_table.Temp_degC == TEMP_VALUES(tIdx);
        if ~any(m), continue; end
        [s, ord] = sort(params_table.SOC_pct(m));
        y = params_table.(pFields{p})(m);  y = y(ord);
        plot(s, y, 'o-', 'Color', colors(tIdx,:), 'LineWidth', 1.4, ...
            'DisplayName', sprintf('%d\\circC', TEMP_VALUES(tIdx)));
    end
    xlabel('SOC (%)'); ylabel(pNames{p}); title(pNames{p});
    legend('Location','best','FontSize',7); grid on;
end
sgtitle('2-RC ECM parameters vs SOC (1C pulse, joint fit, no clamping)', ...
    'FontWeight','bold');

fprintf('\nPhase 1 complete. Next: Phase1b_OCV_SOC.m, then Phase2_ML_LookupTable.m\n');

%% ------------------------------------------------------------------------
%  LOCAL FUNCTION — 2-RC terminal-voltage simulation (ZOH, variable dt)
% -------------------------------------------------------------------------
function vout = ecm2rc_sim(theta, tseg, iseg, ocv)
% V(t) = OCV + R0*I(t) + V1(t) + V2(t), starting from rest (V1=V2=0).
% Discharge current is negative, so V1/V2 go negative and V drops.
    R0 = theta(1); R1 = theta(2); tau1 = theta(3);
    R2 = theta(4); tau2 = theta(5);
    n  = numel(tseg);
    vout = zeros(n,1);
    v1 = 0; v2 = 0;
    vout(1) = ocv + R0*iseg(1);
    for k = 2:n
        dtk = tseg(k) - tseg(k-1);
        a1  = exp(-dtk/tau1);  a2 = exp(-dtk/tau2);
        v1  = a1*v1 + R1*(1-a1)*iseg(k-1);
        v2  = a2*v2 + R2*(1-a2)*iseg(k-1);
        vout(k) = ocv + R0*iseg(k) + v1 + v2;
    end
end
