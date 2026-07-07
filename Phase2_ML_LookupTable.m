%% =========================================================================
%  Phase2_ML_LookupTable.m
%  Random Forest Regression — 2D EKF Parameter Lookup Table
%  Inputs  : SOC (%), Temperature (degC)
%  Outputs : R0, R1, C1, R2, C2
%
%  Reads   : HPPC_params.mat  (produced by Phase1_HPPC_Extraction.m)
%  Saves   : EKF_LookupTable.mat
%
%  Author  : Generated for Ikmal FYP - UTP
% =========================================================================
clear; clc; close all;

%% -------------------------------------------------------------------------
%  USER CONFIGURATION
% -------------------------------------------------------------------------
LOAD_PATH  = 'C:\Users\ekmal\Documents\fyp\laboratory\matlab output\HPPC_params.mat';
SAVE_PATH  = 'C:\Users\ekmal\Documents\fyp\laboratory\matlab output\EKF_LookupTable.mat';

% Dense grid for lookup table output
SOC_GRID   = 0:2.5:100;          % 41 SOC points
TEMP_GRID  = [-10, 0, 10, 25, 40];  % match your measured temperatures

N_TREES    = 500;   % Random Forest ensemble size

%% =========================================================================
%  1. LOAD PHASE 1 RESULTS
% =========================================================================
fprintf('Loading Phase 1 results...\n');
loaded = load(LOAD_PATH);
T = loaded.params_table;

fprintf('  Total parameter rows loaded: %d\n', height(T));
fprintf('  Temperatures present: %s\n', ...
    num2str(unique(T.Temp_degC)', '%d '));
fprintf('  SOC range: %.1f%% to %.1f%%\n', min(T.SOC_pct), max(T.SOC_pct));

%% =========================================================================
%  2. PREPARE TRAINING DATA
% =========================================================================
% Features: [SOC, Temp]  — both normalised for better tree splits
X_raw = [T.SOC_pct, T.Temp_degC];

% Normalise features to [0,1] range
X_min = min(X_raw);
X_max = max(X_raw);
X_norm = (X_raw - X_min) ./ (X_max - X_min);

% Targets — one column per parameter
param_names  = {'R0', 'R1', 'C1', 'R2', 'C2'};
param_labels = {'R_0 (\Omega)', 'R_1 (\Omega)', 'C_1 (F)', 'R_2 (\Omega)', 'C_2 (F)'};

Y = [T.R0, T.R1, T.C1, T.R2, T.C2];

fprintf('\nTraining data summary:\n');
fprintf('  Samples : %d\n', size(X_raw, 1));
fprintf('  Features : SOC (%%), Temperature (degC)\n');
fprintf('  Targets  : R0, R1, C1, R2, C2\n\n');

%% =========================================================================
%  3. TRAIN ONE RANDOM FOREST PER PARAMETER
% =========================================================================
models = cell(1, 5);
oob_errors = zeros(1, 5);

for p = 1:5
    fprintf('  Training Random Forest for %s ... ', param_names{p});
    tic;

    models{p} = TreeBagger(N_TREES, X_norm, Y(:, p), ...
        'Method',          'regression', ...
        'OOBPrediction',   'on', ...
        'MinLeafSize',     2, ...
        'NumPredictorsToSample', 2, ...   % sqrt(2 features) ~ 1, use 2
        'PredictorNames',  {'SOC_norm', 'Temp_norm'});

    oob_errors(p) = mean((oobPredict(models{p}) - Y(:, p)).^2);
    elapsed = toc;

    fprintf('done (%.1fs)  |  OOB MSE = %.6f\n', elapsed, oob_errors(p));
end

%% =========================================================================
%  4. EVALUATE ON TRAINING DATA — COMPARE PREDICTED VS MEASURED
% =========================================================================
fprintf('\nComputing training set predictions...\n');

X_norm_train = X_norm;
Y_pred_train = zeros(size(Y));

for p = 1:5
    Y_pred_train(:, p) = predict(models{p}, X_norm_train);
end

% Print per-parameter fit statistics
fprintf('\n%-6s  %-10s  %-10s  %-10s\n', 'Param', 'RMSE', 'MAE', 'R²');
fprintf('%s\n', repmat('-', 1, 42));

r2_scores = zeros(1, 5);
for p = 1:5
    y_true = Y(:, p);
    y_pred = Y_pred_train(:, p);
    rmse_v = sqrt(mean((y_pred - y_true).^2));
    mae_v  = mean(abs(y_pred - y_true));
    ss_res = sum((y_true - y_pred).^2);
    ss_tot = sum((y_true - mean(y_true)).^2);
    r2     = 1 - ss_res / ss_tot;
    r2_scores(p) = r2;
    fprintf('%-6s  %-10.6f  %-10.6f  %-10.4f\n', ...
        param_names{p}, rmse_v, mae_v, r2);
end

%% =========================================================================
%  5. GENERATE DENSE 2D LOOKUP GRID
% =========================================================================
fprintf('\nGenerating lookup table grid...\n');
fprintf('  SOC grid  : %d points (%.1f to %.1f%%)\n', ...
    length(SOC_GRID), SOC_GRID(1), SOC_GRID(end));
fprintf('  Temp grid : %s degC\n', num2str(TEMP_GRID, '%d '));

[SOC_mesh, TEMP_mesh] = meshgrid(SOC_GRID, TEMP_GRID);

% Flatten grid for batch prediction
X_grid_raw  = [SOC_mesh(:), TEMP_mesh(:)];
X_grid_norm = (X_grid_raw - X_min) ./ (X_max - X_min);

% Predict all 5 parameters across the grid
LUT = struct();
LUT.SOC_vec  = SOC_GRID;
LUT.Temp_vec = TEMP_GRID;

param_grid_names = {'R0_grid','R1_grid','C1_grid','R2_grid','C2_grid'};

fprintf('\nPredicting parameters on full grid...\n');
for p = 1:5
    y_flat = predict(models{p}, X_grid_norm);
    % Reshape to [nTemp x nSOC]
    grid_2d = reshape(y_flat, length(TEMP_GRID), length(SOC_GRID));
    LUT.(param_grid_names{p}) = grid_2d;
    fprintf('  %s grid: min=%.5f  max=%.5f\n', ...
        param_names{p}, min(y_flat), max(y_flat));
end

%% =========================================================================
%  6. SAVE LOOKUP TABLE AND MODELS
% =========================================================================
params_table = loaded.params_table;   % extract from loaded struct for saving
save(SAVE_PATH, 'LUT', 'models', 'X_min', 'X_max', ...
    'param_names', 'SOC_GRID', 'TEMP_GRID', 'params_table');
fprintf('\nLookup table saved to: %s\n', SAVE_PATH);

%% =========================================================================
%  7. PLOT A — Predicted vs Measured (Training Set)
% =========================================================================
figure('Name', 'Predicted vs Measured — Training Set', ...
    'Position', [50, 50, 1400, 800]);

for p = 1:5
    subplot(2, 3, p);

    y_true = Y(:, p);
    y_pred = Y_pred_train(:, p);

    % Colour points by temperature
    scatter(y_true, y_pred, 60, T.Temp_degC, 'filled', ...
        'MarkerEdgeColor', 'k', 'LineWidth', 0.3);

    % Perfect-fit reference line
    lims = [min([y_true; y_pred]), max([y_true; y_pred])];
    hold on;
    plot(lims, lims, 'r--', 'LineWidth', 1.5);

    colormap(gca, jet);
    cb = colorbar;
    cb.Label.String = 'Temperature (°C)';
    clim([min(TEMP_GRID), max(TEMP_GRID)]);

    xlabel(sprintf('Measured %s', param_labels{p}));
    ylabel(sprintf('Predicted %s', param_labels{p}));
    title(sprintf('%s  |  R²=%.4f', param_names{p}, r2_scores(p)));
    grid on; axis square;
end

sgtitle('Random Forest: Predicted vs Measured ECM Parameters', ...
    'FontWeight', 'bold');

%% =========================================================================
%  8. PLOT B — Lookup Table Surface Plots (2D: SOC × Temperature)
% =========================================================================
figure('Name', 'EKF Lookup Table — Parameter Surfaces', ...
    'Position', [100, 100, 1400, 800]);

for p = 1:5
    subplot(2, 3, p);

    grid_2d = LUT.(param_grid_names{p});   % [nTemp x nSOC]

    surf(SOC_GRID, TEMP_GRID, grid_2d, 'EdgeColor', 'none');
    colormap(jet); colorbar;
    shading interp;

    xlabel('SOC (%)');
    ylabel('Temperature (°C)');
    zlabel(param_labels{p});
    title(param_names{p});
    view([-35, 30]);
    grid on;
end

sgtitle('EKF 2D Lookup Table — ECM Parameters vs SOC and Temperature', ...
    'FontWeight', 'bold');

%% =========================================================================
%  9. PLOT C — Parameter vs SOC curves per temperature
%              (measured points overlaid on predicted lines)
% =========================================================================
colors = lines(length(TEMP_GRID));

figure('Name', 'Parameters vs SOC — Predicted Lines + Measured Points', ...
    'Position', [150, 150, 1400, 800]);

for p = 1:5
    subplot(2, 3, p);
    hold on;

    for tIdx = 1:length(TEMP_GRID)
        tv = TEMP_GRID(tIdx);

        % Predicted line (full SOC range from lookup table)
        y_line = LUT.(param_grid_names{p})(tIdx, :);
        plot(SOC_GRID, y_line, '-', 'Color', colors(tIdx,:), ...
            'LineWidth', 1.8, 'DisplayName', sprintf('%d°C pred', tv));

        % Measured scatter points
        mask = T.Temp_degC == tv;
        if any(mask)
            scatter(T.SOC_pct(mask), T.(param_names{p})(mask), 50, ...
                colors(tIdx,:), 'filled', 'MarkerEdgeColor', 'k', ...
                'LineWidth', 0.5, 'HandleVisibility', 'off');
        end
    end

    xlabel('SOC (%)');
    ylabel(param_labels{p});
    title(param_names{p});
    legend('Location', 'best', 'FontSize', 6);
    set(gca, 'XDir', 'reverse');
    grid on;
end

sgtitle('ECM Parameters: Predicted (lines) vs Measured (dots)', ...
    'FontWeight', 'bold');

%% =========================================================================
%  10. HOW TO USE THE LOOKUP TABLE IN YOUR EKF
% =========================================================================
fprintf('\n');
fprintf('=========================================================\n');
fprintf('  HOW TO USE EKF_LookupTable.mat IN YOUR EKF SCRIPT\n');
fprintf('=========================================================\n');
fprintf('\n');
fprintf('  %% Load once at the start of your EKF script:\n');
fprintf('  lut = load(''EKF_LookupTable.mat'');\n\n');
fprintf('  %% Query parameters at any SOC and temperature:\n');
fprintf('  soc_now  = 75;    %% percent\n');
fprintf('  temp_now = 25;    %% degC\n\n');
fprintf('  R0 = interp2(lut.LUT.SOC_vec, lut.LUT.Temp_vec, ...\n');
fprintf('               lut.LUT.R0_grid, soc_now, temp_now);\n');
fprintf('  R1 = interp2(lut.LUT.SOC_vec, lut.LUT.Temp_vec, ...\n');
fprintf('               lut.LUT.R1_grid, soc_now, temp_now);\n');
fprintf('  C1 = interp2(lut.LUT.SOC_vec, lut.LUT.Temp_vec, ...\n');
fprintf('               lut.LUT.C1_grid, soc_now, temp_now);\n');
fprintf('  R2 = interp2(lut.LUT.SOC_vec, lut.LUT.Temp_vec, ...\n');
fprintf('               lut.LUT.R2_grid, soc_now, temp_now);\n');
fprintf('  C2 = interp2(lut.LUT.SOC_vec, lut.LUT.Temp_vec, ...\n');
fprintf('               lut.LUT.C2_grid, soc_now, temp_now);\n\n');
fprintf('=========================================================\n');
fprintf('\nPhase 2 complete.\n');