%% =========================================================================
%  Phase1b_OCV_SOC.m
%  OCV-SoC Curve Construction  (rewritten)
%  Turnigy Graphene 5000mAh 65C Cell
%
%  WHAT CHANGED vs the old version:
%   1. Reads the new ocv_table from Phase 1: OCV points at their ACTUAL
%      coulomb-counted SOC (not mislabeled targets), plus one low-SOC
%      anchor per temperature taken from the rest after the final
%      discharge to cutoff. This restores the steep OCV drop below 10%
%      SOC (~3.5 V -> ~3.0 V) that the old curve missed completely —
%      that slope is the EKF's observability at low SOC.
%   2. Interpolation uses PCHIP (shape-preserving piecewise cubic).
%      Like makima it cannot oscillate, but it additionally preserves
%      monotonicity between monotone points — important now that the
%      curve has a very steep tail next to a flat plateau.
%   3. Below the lowest measured point the curve is extended LINEARLY
%      with the local slope (floored at 2.5 V) instead of free cubic
%      extrapolation.
%   4. The dOCV/dSOC gradient grid is precomputed and saved so the EKF
%      uses a consistent analytic slope instead of re-differencing.
%
%  Reads  : matlab output/HPPC_params.mat   (Phase 1)
%  Saves  : matlab output/OCV_SoC_results.mat
%           OCV_grid  [nTemp x nSOC], dOCV_grid, SOC_DENSE, OCV_TEMPS
% =========================================================================
clear; clc; close all;

ROOT      = fileparts(mfilename('fullpath'));
LOAD_PATH = fullfile(ROOT, 'matlab output', 'HPPC_params.mat');
SAVE_PATH = fullfile(ROOT, 'matlab output', 'OCV_SoC_results.mat');

SOC_DENSE = linspace(0, 100, 1001);     % dense SOC axis for lookup
V_FLOOR   = 2.5;   V_CEIL = 4.25;       % physical clamps

%% ------------------------------------------------------------------------
%  LOAD PHASE 1 RESULTS
% -------------------------------------------------------------------------
fprintf('Loading %s\n', LOAD_PATH);
L = load(LOAD_PATH);
ocv_table = L.ocv_table;

OCV_TEMPS = sort(unique(ocv_table.Temp_degC))';       % ascending for interp2
nT = numel(OCV_TEMPS);
OCV_grid = zeros(nT, numel(SOC_DENSE));

fprintf('Temperatures: %s degC\n', num2str(OCV_TEMPS));

%% ------------------------------------------------------------------------
%  BUILD ONE PCHIP CURVE PER TEMPERATURE
% -------------------------------------------------------------------------
figure('Name','OCV-SoC curves','Position',[60,60,950,650]); hold on;
colors = lines(nT);
legendEntries = {};

for k = 1:nT
    tv = OCV_TEMPS(k);
    m  = ocv_table.Temp_degC == tv;
    [s, ord] = sort(ocv_table.SOC_pct(m));
    vv = ocv_table.OCV(m);  vv = vv(ord);
    [s, iu] = unique(s);  vv = vv(iu);

    % --- linear tail down to SOC = 0 using the two lowest points --------
    slope = (vv(2) - vv(1)) / (s(2) - s(1));
    v0 = max(V_FLOOR, vv(1) - slope*s(1));
    s  = [0; s];  vv = [v0; vv];

    % --- extend to SOC = 100 if the top point sits below it -------------
    if s(end) < 100
        slopeTop = (vv(end) - vv(end-1)) / (s(end) - s(end-1));
        vTop = vv(end) + slopeTop * (100 - s(end));
        s  = [s; 100];  vv = [vv; vTop];
    end

    ocv_dense = interp1(s, vv, SOC_DENSE, 'pchip');
    ocv_dense = min(max(ocv_dense, V_FLOOR), V_CEIL);
    OCV_grid(k, :) = ocv_dense;

    plot(SOC_DENSE, ocv_dense, '-', 'Color', colors(k,:), 'LineWidth', 1.8);
    scatter(s(2:end), vv(2:end), 45, colors(k,:), 'filled', 'MarkerEdgeColor','k');
    legendEntries{end+1} = sprintf('%d\\circC curve', tv);    %#ok<SAGROW>
    legendEntries{end+1} = sprintf('%d\\circC measured', tv); %#ok<SAGROW>
end

xlabel('SOC (%)'); ylabel('OCV (V)');
title('OCV-SoC — PCHIP with low-SOC anchors (Turnigy Graphene 5 Ah)', ...
    'FontWeight','bold');
legend(legendEntries, 'Location','southeast', 'FontSize', 7);
xlim([0 100]); ylim([2.5 4.3]); grid on;

%% ------------------------------------------------------------------------
%  GRADIENT GRID (EKF measurement Jacobian term)
% -------------------------------------------------------------------------
dOCV_grid = zeros(size(OCV_grid));
for k = 1:nT
    dOCV_grid(k, :) = gradient(OCV_grid(k, :), SOC_DENSE);
end

i25 = find(OCV_TEMPS == 25, 1);
if ~isempty(i25)
    q = @(sq) OCV_grid(i25, find(SOC_DENSE >= sq, 1, 'first'));
    fprintf('\nSanity check, 25 degC curve:\n');
    fprintf('  OCV(  0%%)=%.3f  OCV(5%%)=%.3f  OCV(10%%)=%.3f  OCV(50%%)=%.3f  OCV(100%%)=%.3f\n', ...
        q(0), q(5), q(10), q(50), q(100));
    g = @(sq) dOCV_grid(i25, find(SOC_DENSE >= sq, 1, 'first'));
    fprintf('  dOCV/dSOC at 5%% = %.4f V/%%   at 50%% = %.4f V/%%\n', g(5), g(50));
end

%% ------------------------------------------------------------------------
%  SAVE + SURFACE PLOT
% -------------------------------------------------------------------------
save(SAVE_PATH, 'OCV_grid', 'dOCV_grid', 'SOC_DENSE', 'OCV_TEMPS');
fprintf('\nSaved: %s\n', SAVE_PATH);

figure('Name','OCV surface','Position',[140,140,900,600]);
[SOCm, Tm] = meshgrid(SOC_DENSE, OCV_TEMPS);
surf(SOCm, Tm, OCV_grid, 'EdgeColor','none'); shading interp;
colormap(parula); colorbar;
xlabel('SOC (%)'); ylabel('Temperature (\circC)'); zlabel('OCV (V)');
title('OCV(SOC, T) lookup surface for EKF', 'FontWeight','bold');
view([-40, 30]); grid on;

fprintf('\nPhase 1b complete. Next: Phase2_ML_LookupTable.m\n');
