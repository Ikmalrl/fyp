%% =========================================================================
%  Phase1_HPPC_Extraction.m
%  2-RC ECM Parameter Identification from HPPC Data
%  Turnigy Graphene 5000mAh 65C Cell
%
%  Outputs:
%    - params_table : table [Temp, SOC, OCV, R0, R1, C1, R2, C2]
%    - HPPC_params.mat : saved results for Phase 1b and Phase 2
%    - Figures : pulse fits per temperature
%
%  Author  : Generated for Ikmal FYP - UTP
%  Model   : 2-RC Equivalent Circuit Model
% =========================================================================
clear; clc; close all;

%% -------------------------------------------------------------------------
%  USER CONFIGURATION  (edit these paths/values if needed)
% -------------------------------------------------------------------------
DATA_FOLDER   = 'C:\Users\ekmal\Documents\fyp\laboratory\hppc';
SAVE_PATH     = 'C:\Users\ekmal\Documents\fyp\laboratory\matlab output\HPPC_params.mat';

% Target SOC setpoints from HPPC protocol (%)
SOC_TARGETS   = [100, 95, 90, 80, 70, 60, 50, 40, 30, 20, 15, 10, 5, 2.5];

% Temperatures to process (must match strings in filenames)
TEMP_LABELS   = {'40degC', '25degC', '10degC', '0degC', '-10degC'};  % user's filenames
TEMP_VALUES   = [40, 25, 10, 0, -10];   % numeric equivalents

% Pulse detection thresholds
CURRENT_THRESHOLD  = -2.0;   % A  — pulse starts when I drops below this
MIN_PULSE_DURATION = 5;      % s  — ignore transients shorter than this
PRE_PULSE_WINDOW   = 10;     % s  — OCV window before each pulse
RELAX_WINDOW_MIN   = 35;     % s  — minimum relaxation window after pulse
RELAX_WINDOW_MAX   = 40;     % s  — cap relaxation window after pulse

% lsqcurvefit options
OPT = optimoptions('lsqcurvefit', ...
    'Display',       'off', ...
    'MaxIterations', 2000, ...
    'FunctionTolerance', 1e-8);

%% -------------------------------------------------------------------------
%  INITIALISE RESULTS STORAGE
% -------------------------------------------------------------------------
results = [];   % will grow as [Temp, SOC, OCV, R0, R1, C1, R2, C2]

%% =========================================================================
%  MAIN LOOP — iterate over each temperature
% =========================================================================
for tIdx = 1:length(TEMP_LABELS)

    tempLabel = TEMP_LABELS{tIdx};
    tempVal   = TEMP_VALUES(tIdx);

    fprintf('\n==============================================\n');
    fprintf(' Processing: %s  (%d degC)\n', tempLabel, tempVal);
    fprintf('==============================================\n');

    % ------------------------------------------------------------------
    %  1. FIND AND LOAD THE HPPC FILE FOR THIS TEMPERATURE
    % ------------------------------------------------------------------
    pattern  = fullfile(DATA_FOLDER, sprintf('*HPPC*%s*.mat', tempLabel));
    fileList = dir(pattern);

    if isempty(fileList)
        fprintf('  [WARN] No HPPC file found for %s — skipping.\n', tempLabel);
        continue
    end

    filePath = fullfile(fileList(1).folder, fileList(1).name);
    fprintf('  Loaded: %s\n', fileList(1).name);

    raw  = load(filePath);
    meas = raw.meas;

    % Extract columns (handle both row and column vectors)
    time    = double(meas.Time(:));
    voltage = double(meas.Voltage(:));
    current = double(meas.Current(:));
    ah      = double(meas.Ah(:));

    % ------------------------------------------------------------------
    %  2. COMPUTE SOC TRACE VIA COULOMB COUNTING
    %     The Ah column resets after each pulse/rest segment, so we must
    %     integrate current directly rather than trust the Ah counter.
    % ------------------------------------------------------------------
    CAPACITY_AH = 5.0;

    % Integrate current over time using the trapezoidal rule
    dt       = [0; diff(time)];          % time steps (s)
    dAh_int  = current .* dt / 3600;    % incremental Ah (negative = discharge)
    cumAh    = cumsum(dAh_int);          % cumulative Ah from start of file

    % At the very start of the HPPC file the cell is fully charged (100%)
    % Discharge makes current negative → cumAh decreases → SOC decreases
    soc_trace = 100 + (cumAh / CAPACITY_AH) * 100;
    soc_trace = max(0, min(100, soc_trace));

    % Debug: print SOC range to verify
    fprintf('  SOC trace range: %.1f%% to %.1f%%\n', ...
        max(soc_trace), min(soc_trace));

    % ------------------------------------------------------------------
    %  3. DETECT ALL DISCHARGE PULSES IN THE FILE
    %     A discharge pulse: current goes from ~0 to negative (<threshold)
    % ------------------------------------------------------------------
    % Smooth current slightly to avoid noise-triggered detections
    I_smooth = movmean(current, 5);

    % Find samples where current crosses the threshold going negative
    above = I_smooth >= CURRENT_THRESHOLD;
    pulse_starts_raw = find(diff(above) == -1) + 1;

    % Filter: keep only pulses that last at least MIN_PULSE_DURATION
    pulse_starts = [];
    for k = 1:length(pulse_starts_raw)
        ps = pulse_starts_raw(k);
        t_start = time(ps);
        % Find when current returns above threshold
        later = find(I_smooth(ps:end) >= CURRENT_THRESHOLD, 1, 'first');
        if isempty(later), later = length(I_smooth) - ps; end
        duration = time(min(ps + later - 1, length(time))) - t_start;
        if duration >= MIN_PULSE_DURATION
            pulse_starts = [pulse_starts; ps]; %#ok<AGROW>
        end
    end

    % Also find pulse ends (current returns to ~0)
    pulse_ends = zeros(size(pulse_starts));
    for k = 1:length(pulse_starts)
        ps   = pulse_starts(k);
        later = find(I_smooth(ps:end) >= CURRENT_THRESHOLD, 1, 'first');
        if isempty(later)
            pulse_ends(k) = length(time);
        else
            pulse_ends(k) = ps + later - 2;
        end
    end

    fprintf('  Detected %d discharge pulses\n', length(pulse_starts));

    % ------------------------------------------------------------------
    %  4. MATCH EACH PULSE TO A TARGET SOC SETPOINT
    % ------------------------------------------------------------------
    % For each pulse, read the SOC just before the pulse starts
    pulse_soc = zeros(size(pulse_starts));
    for k = 1:length(pulse_starts)
        pre_idx      = max(1, pulse_starts(k) - 5);
        pulse_soc(k) = soc_trace(pre_idx);
    end

    % Match pulses to SOC_TARGETS using nearest-neighbour
    matched_pulse_idx = nan(size(SOC_TARGETS));   % index into pulse_starts
    used_pulses       = false(size(pulse_starts));

    for s = 1:length(SOC_TARGETS)
        target = SOC_TARGETS(s);
        diffs  = abs(pulse_soc - target);
        diffs(used_pulses) = inf;   % don't reuse pulses
        [minDiff, bestK] = min(diffs);
        if minDiff < 12   % within 12% SOC tolerance (handles coulomb counting drift)
            matched_pulse_idx(s) = bestK;
            used_pulses(bestK)   = true;
            fprintf('  SOC target %5.1f%% -> pulse %2d (actual SOC = %.1f%%)\n', ...
                target, bestK, pulse_soc(bestK));
        else
            fprintf('  SOC target %5.1f%% -> NO MATCH (closest %.1f%% off)\n', ...
                target, minDiff);
        end
    end

    % ------------------------------------------------------------------
    %  5. EXTRACT WINDOWS AND FIT PARAMETERS FOR EACH SOC POINT
    % ------------------------------------------------------------------
    figure('Name', sprintf('HPPC Fits — %s', tempLabel), ...
           'Position', [100, 100, 1400, 900]);
    nRows = 3;
    nCols = ceil(length(SOC_TARGETS) / nRows);
    plotIdx = 0;

    for s = 1:length(SOC_TARGETS)

        if isnan(matched_pulse_idx(s)), continue; end

        k   = matched_pulse_idx(s);
        ps  = pulse_starts(k);
        pe  = pulse_ends(k);
        soc = SOC_TARGETS(s);

        % ---- 5a. Find sample indices for each window ----------------
        t_pulse_start = time(ps);
        t_pulse_end   = time(pe);

        % Pre-pulse OCV window
        pre_start_t = t_pulse_start - PRE_PULSE_WINDOW;
        pre_idx     = find(time >= pre_start_t & time < t_pulse_start);
        if isempty(pre_idx), pre_idx = max(1, ps-10):ps-1; end

        % OCV = mean voltage in last 3 seconds of pre-pulse rest
        ocv_idx = pre_idx(time(pre_idx) >= t_pulse_start - 3);
        if isempty(ocv_idx), ocv_idx = pre_idx; end
        OCV = mean(voltage(ocv_idx));

        % Pulse window
        pulse_idx = ps:pe;

        % Relaxation window — dynamic: from pulse end to next pulse start
        % but capped at RELAX_WINDOW_MAX
        if k < length(pulse_starts)
            next_ps = pulse_starts(k + 1);
            t_relax_end = min(time(pe) + RELAX_WINDOW_MAX, time(next_ps) - 1);
        else
            t_relax_end = time(pe) + RELAX_WINDOW_MAX;
        end
        t_relax_end = max(t_relax_end, time(pe) + RELAX_WINDOW_MIN);

        relax_idx = find(time > t_pulse_end & time <= t_relax_end);
        if length(relax_idx) < 10
            fprintf('  [WARN] SOC %.1f%% — relaxation window too short (%d pts), skipping\n', ...
                soc, length(relax_idx));
            continue
        end

        % ---- 5b. Extract R0 from instantaneous voltage drop ----------
        % Use the first few samples after pulse start to find initial drop
        n_r0 = min(5, length(pulse_idx));
        V_before = OCV;
        V_after  = median(voltage(pulse_idx(1:n_r0)));
        I_pulse  = median(current(pulse_idx(1:n_r0)));

        if abs(I_pulse) < 0.1
            fprintf('  [WARN] SOC %.1f%% — pulse current near zero, skipping R0\n', soc);
            continue
        end

        R0 = abs(V_before - V_after) / abs(I_pulse);
        R0 = max(R0, 1e-5);   % physical floor

        % ---- 5c. Fit 2-RC relaxation curve ---------------------------
        t_relax = time(relax_idx) - time(relax_idx(1));   % relative time
        V_relax = voltage(relax_idx);

        % Current magnitude during pulse (for scaling initial guesses)
        I_mag = abs(mean(current(pulse_idx)));

        % Voltage drop during pulse (approximate total polarisation)
        V_drop = abs(OCV - mean(voltage(pulse_idx)));

        % Voltage at end of pulse = starting point for relaxation
        V_end_pulse = voltage(pe);
        total_drop  = OCV - V_end_pulse;   % total polarisation at pulse end

        % Initial guesses and bounds for [delta1, tau1, delta2, tau2]
        x0_delta = [0.6*total_drop, 10,  0.4*total_drop, 50];
        lb_delta = [1e-4,           0.5, 1e-4,           1.0];   % user's tightened bounds
        ub_delta = [total_drop*2,   20,  total_drop*2,   100];   % user's tightened bounds

        % 2-RC relaxation model (voltage recovering after pulse):
        %   V(t) = OCV - delta1*exp(-t/tau1) - delta2*exp(-t/tau2)
        model_fn = @(x, t) OCV - x(1)*exp(-t/x(2)) - x(3)*exp(-t/x(4));

        try
            [x_fit, ~] = lsqcurvefit(model_fn, x0_delta, ...
                t_relax, V_relax, lb_delta, ub_delta, OPT);
        catch ME
            fprintf('  [WARN] SOC %.1f%% — lsqcurvefit failed: %s\n', soc, ME.message);
            continue
        end

        delta1 = x_fit(1);  tau1 = x_fit(2);
        delta2 = x_fit(3);  tau2 = x_fit(4);

        % Ensure tau1 < tau2 (tau1 = fast RC, tau2 = slow RC)
        if tau1 > tau2
            [delta1, delta2] = deal(delta2, delta1);
            [tau1,   tau2  ] = deal(tau2,   tau1);
        end

        % Convert back to R1, C1, R2, C2
        R1 = delta1 / I_mag;   C1 = tau1 / R1;
        R2 = delta2 / I_mag;   C2 = tau2 / R2;

        % Hard physical bounds (user's tightened values for a 5Ah cell)
        R1 = max(1e-4, min(R1, 0.1));    % cap at 0.1 Ohm
        R2 = max(1e-4, min(R2, 0.1));
        C1 = max(100,  min(C1, 5000));   % fast capacitance 100–5000 F
        C2 = max(500,  min(C2, 15000));  % slow capacitance 500–15000 F

        % Goodness-of-fit (RMSE)
        V_pred = model_fn(x_fit, t_relax);
        rmse   = sqrt(mean((V_pred - V_relax).^2)) * 1000;   % mV

        fprintf('  SOC %5.1f%% | R0=%.4f  R1=%.4f  C1=%.1f  R2=%.4f  C2=%.1f  RMSE=%.2f mV\n', ...
            soc, R0, R1, C1, R2, C2, rmse);

        % ---- 5d. Store results (NOW INCLUDES OCV) --------------------
        results = [results; tempVal, soc, OCV, R0, R1, C1, R2, C2]; %#ok<AGROW>

        % ---- 5e. Plot this pulse fit ----------------------------------
        plotIdx = plotIdx + 1;
        subplot(nRows, nCols, plotIdx);

        % Show pre-pulse, pulse, and relaxation together
        % Force all index arrays to row vectors to avoid horzcat errors
        all_idx = [pre_idx(:)', pulse_idx(:)', relax_idx(:)'];
        t_plot  = time(all_idx) - time(pre_idx(1));
        v_plot  = voltage(all_idx);

        % Fitted relaxation overlay
        t_fit_plot = t_relax + (time(relax_idx(1)) - time(pre_idx(1)));
        v_fit_plot = model_fn(x_fit, t_relax);

        plot(t_plot, v_plot, 'b-', 'LineWidth', 1.2); hold on;
        plot(t_fit_plot, v_fit_plot, 'r--', 'LineWidth', 1.5);
        xline(time(ps) - time(pre_idx(1)), 'k:', 'Pulse');
        xline(time(pe) - time(pre_idx(1)), 'g:', 'Relax');

        title(sprintf('SOC=%.1f%%  RMSE=%.1fmV', soc, rmse), 'FontSize', 8);
        xlabel('Time (s)'); ylabel('Voltage (V)');
        legend('Measured','Fitted','Location','southeast','FontSize',6);
        grid on;

    end  % SOC loop

    sgtitle(sprintf('HPPC 2-RC Fits — %s', tempLabel), 'FontWeight', 'bold');

end  % temperature loop

%% =========================================================================
%  6. BUILD AND SAVE RESULTS TABLE
% =========================================================================
if isempty(results)
    error('No results extracted. Check file paths and naming patterns.');
end

% Table now includes OCV column for Phase 1b
params_table = array2table(results, ...
    'VariableNames', {'Temp_degC','SOC_pct','OCV','R0','R1','C1','R2','C2'});

% Sort by temperature then SOC descending
params_table = sortrows(params_table, {'Temp_degC','SOC_pct'}, {'descend','descend'});

disp(' ');
disp('===== Extracted Parameters (first 10 rows) =====');
disp(params_table(1:min(10,height(params_table)), :));

save(SAVE_PATH, 'params_table', 'SOC_TARGETS', 'TEMP_VALUES');
fprintf('\nResults saved to: %s\n', SAVE_PATH);

%% =========================================================================
%  7. SUMMARY PLOTS — Parameters vs SOC for all temperatures
% =========================================================================
param_names  = {'R0 (Ohm)', 'R1 (Ohm)', 'C1 (F)', 'R2 (Ohm)', 'C2 (F)'};
param_fields = {'R0', 'R1', 'C1', 'R2', 'C2'};
colors = lines(length(TEMP_VALUES));

figure('Name', 'ECM Parameters vs SOC', 'Position', [200, 200, 1400, 700]);

for p = 1:5
    subplot(2, 3, p);
    hold on;

    for tIdx = 1:length(TEMP_VALUES)
        tv   = TEMP_VALUES(tIdx);
        mask = params_table.Temp_degC == tv;
        if ~any(mask), continue; end

        soc_vals   = params_table.SOC_pct(mask);
        param_vals = params_table.(param_fields{p})(mask);
        [soc_vals, ord] = sort(soc_vals);
        param_vals      = param_vals(ord);

        plot(soc_vals, param_vals, 'o-', ...
            'Color', colors(tIdx,:), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('%d°C', tv));
    end

    xlabel('SOC (%)');  ylabel(param_names{p});
    title(param_names{p});
    legend('Location', 'best', 'FontSize', 7);
    grid on;  set(gca, 'XDir', 'reverse');
end

sgtitle('2-RC ECM Parameters vs SOC — All Temperatures', 'FontWeight', 'bold');

fprintf('\nPhase 1 complete. Run Phase1b_OCV_SoC.m and Phase2_ML_LookupTable.m next.\n');