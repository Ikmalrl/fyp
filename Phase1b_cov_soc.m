%% =========================================================================
%  Phase1b_OCV_SoC.m (THE UNFUCKED VERSION)
%  OCV-SoC Curve Extraction using Makima Interpolation
%  Turnigy Graphene 5000mAh 65C Cell
%
%  Reads   : HPPC_params.mat 
%  Method  : Modified Akima piecewise cubic Hermite interpolation (makima)
%            Guarantees zero overshoot/spikes between discrete HPPC points.
%
%  Outputs :
%    - OCV_SoC_results.mat  : dense OCV-SoC vectors for EKF
%    - Figure 1 : OCV vs SOC — measured points + perfectly smooth curves
%    - Figure 2 : OCV surface (SOC x Temperature) for EKF
% =========================================================================
clear; clc; close all;

%% -------------------------------------------------------------------------
%  USER CONFIGURATION
% -------------------------------------------------------------------------
LOAD_PATH  = 'C:\Users\ekmal\Documents\fyp\laboratory\matlab output\HPPC_params.mat';
SAVE_PATH  = 'C:\Users\ekmal\Documents\fyp\laboratory\matlab output\OCV_SoC_results.mat';

% Dense SOC vector for outputting fitted OCV curve (used in EKF lookup)
SOC_DENSE  = linspace(0, 100, 1000);   % 1000-point lookup

% Temperature list (must match Phase 1 output)
TEMP_VALUES = [40, 25, 10, 0, -10];

%% =========================================================================
%  1. LOAD PHASE 1 RESULTS
% =========================================================================
fprintf('Loading Phase 1 results from: %s\n', LOAD_PATH);
loaded = load(LOAD_PATH);
T = loaded.params_table;

% Verify OCV column exists
if ~ismember('OCV', T.Properties.VariableNames)
    error('OCV column not found in params_table. Go run Phase 1 again.');
end

fprintf('  Rows loaded      : %d\n', height(T));
fprintf('  Temperatures     : %s degC\n', num2str(unique(T.Temp_degC)', '%d '));
fprintf('  OCV range        : %.3f V to %.3f V\n', min(T.OCV), max(T.OCV));

%% =========================================================================
%  2. BUILD OCV-SOC INTERPOLATION PER TEMPERATURE
% =========================================================================
ocv_lut = struct();   % dense OCV vectors for EKF
colors = lines(length(TEMP_VALUES));

fprintf('\n%-8s  %-20s\n', 'Temp', 'Status');
fprintf('%s\n', repmat('-', 1, 30));

for tIdx = 1:length(TEMP_VALUES)
    tv = TEMP_VALUES(tIdx);

    % Extract this temperature's data
    mask    = T.Temp_degC == tv;
    soc_raw = T.SOC_pct(mask);
    ocv_raw = T.OCV(mask);

    if isempty(soc_raw)
        fprintf('%-8d  [NO DATA — skipping]\n', tv);
        continue
    end

    % MUST have unique SOC points for interpolation to work
    [soc_pts, unique_idx] = unique(soc_raw);
    ocv_pts = ocv_raw(unique_idx);

    % ---- THE MAGIC: Modified Akima Interpolation ---------------------
    % 'makima' prevents the wild oscillations and heartbeat spikes 
    % while maintaining a smooth continuous derivative.
    ocv_dense = interp1(soc_pts, ocv_pts, SOC_DENSE, 'makima', 'extrap');

    % Clamp to the physical voltage limits of the cell just in case
    % extreme extrapolation gets weird at 0% or 100%
    ocv_dense = max(2.5, min(ocv_dense, 4.25));

    fprintf('%-8d  %-20s\n', tv, 'Makima Interpolation Built');

    % ---- Store results -----------------------------------------------
    ocv_lut(tIdx).temp      = tv;
    ocv_lut(tIdx).soc_pts   = soc_pts;
    ocv_lut(tIdx).ocv_pts   = ocv_pts;
    ocv_lut(tIdx).SOC_vec   = SOC_DENSE;
    ocv_lut(tIdx).OCV_vec   = ocv_dense;
end

%% =========================================================================
%  3. BUILD 2D OCV LOOKUP GRID  [nTemp × nSOC]  for interp2 in EKF
% =========================================================================
valid_temps = [ocv_lut.temp];
OCV_grid    = zeros(length(valid_temps), length(SOC_DENSE));

for tIdx = 1:length(valid_temps)
    OCV_grid(tIdx, :) = ocv_lut(tIdx).OCV_vec;
end

%% =========================================================================
%  4. SAVE RESULTS
% =========================================================================
save(SAVE_PATH, 'ocv_lut', 'OCV_grid', 'SOC_DENSE', 'valid_temps', 'TEMP_VALUES');
fprintf('\nUnfucked OCV-SoC results saved to: %s\n', SAVE_PATH);

%% =========================================================================
%  5. PLOT A — OCV vs SOC: measured points + fitted curves (all temps)
% =========================================================================
figure('Name', 'OCV-SoC Curves — Makima Fit', 'Position', [50, 50, 900, 600]);
hold on;

legend_entries = {};

for tIdx = 1:length(valid_temps)
    tv = valid_temps(tIdx);
    idx = find([ocv_lut.temp] == tv);
    
    soc_pts   = ocv_lut(idx).soc_pts;
    ocv_pts   = ocv_lut(idx).ocv_pts;
    ocv_dense = ocv_lut(idx).OCV_vec;

    % Fitted line (Makima)
    plot(SOC_DENSE, ocv_dense, '-', 'Color', colors(tIdx,:), 'LineWidth', 2.0);

    % Measured scatter points
    scatter(soc_pts, ocv_pts, 60, colors(tIdx,:), 'filled', ...
        'MarkerEdgeColor', 'k', 'LineWidth', 0.8);

    legend_entries{end+1} = sprintf('%d°C (Makima Curve)', tv); %#ok<SAGROW>
    legend_entries{end+1} = sprintf('%d°C Measured Points', tv); %#ok<SAGROW>
end

xlabel('State of Charge (%)');
ylabel('Open Circuit Voltage (V)');
title('OCV-SoC Relationship — Turnigy Graphene 5Ah Cell (Makima Interpolation)', 'FontWeight', 'bold');
legend(legend_entries, 'Location', 'southeast', 'FontSize', 8);
set(gca, 'XDir', 'reverse');
xlim([0 100]); ylim([2.5 4.3]);
grid on; set(gca, 'FontSize', 11);

%% =========================================================================
%  6. PLOT B — 2D OCV Surface (SOC × Temperature)
% =========================================================================
[SOC_mesh, TEMP_mesh] = meshgrid(SOC_DENSE, valid_temps);

figure('Name', 'OCV Surface — SOC × Temperature', ...
    'Position', [150, 150, 900, 600]);

surf(SOC_mesh, TEMP_mesh, OCV_grid, 'EdgeColor', 'none');
shading interp;
colormap(jet); colorbar;

xlabel('SOC (%)');
ylabel('Temperature (°C)');
zlabel('OCV (V)');
title('OCV-SoC-Temperature Surface — EKF Lookup', 'FontWeight', 'bold');
view([-40, 30]);
grid on;

fprintf('\nPhase 1b is completely unfucked. Proceed.\n');