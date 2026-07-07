%% =========================================================================
%  Phase2_ML_LookupTable.m
%  Machine-Learning Parameter Surfaces — Gaussian Process Regression
%  Inputs  : SOC (%), Temperature (degC)
%  Outputs : R0, R1, tau1, R2, tau2  (C1, C2 derived as tau/R)
%
%  WHAT CHANGED vs the old version:
%   1. Random Forest -> GAUSSIAN PROCESS REGRESSION (fitrgp).
%      With ~60 training points a 500-tree forest memorises the data and
%      produces piecewise-constant "staircase" surfaces that cannot
%      extrapolate. A GP with an ARD squared-exponential kernel gives
%      smooth continuous surfaces, principled uncertainty bands, and
%      graceful interpolation between the 5 measured temperatures —
%      exactly what the EKF needs to query at arbitrary (SOC, T).
%      A Random Forest baseline is still trained and compared via the
%      SAME cross-validation folds, so the choice is evidence-based.
%   2. Targets are learned in LOG space: resistances/time-constants are
%      positive and vary ~10x across temperature, so log-space learning
%      guarantees positive predictions and evens out the error scale.
%   3. Model quality is reported with 5-fold CROSS-VALIDATION (the old
%      script printed R^2 on the training set, which for a memorising
%      forest is meaninglessly close to 1).
%   4. tau1/tau2 are modelled instead of C1/C2 (better behaved), and the
%      EKF grids export R0, R1, C1, R2, C2 plus GP uncertainty maps.
%
%  Reads : matlab output/HPPC_params.mat
%  Saves : matlab output/EKF_LookupTable.mat
% =========================================================================
clear; clc; close all;
rng(0);                                  % reproducible CV folds

ROOT      = fileparts(mfilename('fullpath'));
LOAD_PATH = fullfile(ROOT, 'matlab output', 'HPPC_params.mat');
SAVE_PATH = fullfile(ROOT, 'matlab output', 'EKF_LookupTable.mat');

SOC_GRID  = 0:0.5:100;                   % dense lookup axes (ascending)
TEMP_GRID = -10:2.5:40;
KFOLD     = 5;

%% ------------------------------------------------------------------------
%  LOAD TRAINING DATA
% -------------------------------------------------------------------------
L = load(LOAD_PATH);
T = L.params_table;
fprintf('Training rows: %d  (flags: %d ok / %d check)\n', height(T), ...
    nnz(T.Flag == "ok"), nnz(T.Flag == "check"));
% NOTE: 'check' rows (mostly cold, low SOC) are kept — they carry the real
% trend of rising resistance; the GP noise term absorbs their extra scatter.

X = [T.SOC_pct, T.Temp_degC];
targetNames = {'R0','R1','tau1','R2','tau2'};
Y = log([T.R0, T.R1, T.tau1, T.R2, T.tau2]);   % log-space targets

%% ------------------------------------------------------------------------
%  TRAIN GPR PER PARAMETER + 5-FOLD CV COMPARISON WITH RANDOM FOREST
% -------------------------------------------------------------------------
gprModels = cell(1,5);
cv = cvpartition(height(T), 'KFold', KFOLD);
cvRMSE_gpr = zeros(1,5);  cvRMSE_rf = zeros(1,5);

fprintf('\n%-6s  %-18s  %-18s\n', 'Param', 'GPR CV-RMSE(log)', 'RF CV-RMSE(log)');
fprintf('%s\n', repmat('-', 1, 48));

for p = 1:5
    y = Y(:,p);

    % ---- cross-validation with identical folds --------------------------
    errG = []; errF = [];
    for f = 1:KFOLD
        tr = training(cv,f);  te = test(cv,f);
        g  = fitrgp(X(tr,:), y(tr), ...
            'KernelFunction','ardsquaredexponential', ...
            'BasisFunction','constant', 'Standardize', true, 'Sigma', 0.2);
        errG = [errG; predict(g, X(te,:)) - y(te)]; %#ok<AGROW>
        rf = TreeBagger(200, X(tr,:), y(tr), 'Method','regression', ...
            'MinLeafSize', 2);
        errF = [errF; predict(rf, X(te,:)) - y(te)]; %#ok<AGROW>
    end
    cvRMSE_gpr(p) = sqrt(mean(errG.^2));
    cvRMSE_rf(p)  = sqrt(mean(errF.^2));
    fprintf('%-6s  %-18.3f  %-18.3f\n', targetNames{p}, cvRMSE_gpr(p), cvRMSE_rf(p));

    % ---- final GPR trained on all data ----------------------------------
    gprModels{p} = fitrgp(X, y, ...
        'KernelFunction','ardsquaredexponential', ...
        'BasisFunction','constant', 'Standardize', true, 'Sigma', 0.2);
end
fprintf('(lower is better; log-space RMSE of 0.2 ~ 20%% parameter error)\n');

%% ------------------------------------------------------------------------
%  DENSE LOOKUP GRIDS  [nTemp x nSOC]  + UNCERTAINTY
% -------------------------------------------------------------------------
[SOCm, Tm] = meshgrid(SOC_GRID, TEMP_GRID);
Xq = [SOCm(:), Tm(:)];

LUT = struct();
LUT.SOC_vec  = SOC_GRID;
LUT.Temp_vec = TEMP_GRID;

gridNames = {'R0_grid','R1_grid','tau1_grid','R2_grid','tau2_grid'};
sdNames   = {'R0_sd','R1_sd','tau1_sd','R2_sd','tau2_sd'};
for p = 1:5
    [mu, sd] = predict(gprModels{p}, Xq);
    LUT.(gridNames{p}) = reshape(exp(mu), size(SOCm));
    LUT.(sdNames{p})   = reshape(sd, size(SOCm));   % log-space SD (~relative)
    fprintf('  %s: min=%.5g  max=%.5g\n', targetNames{p}, ...
        min(exp(mu)), max(exp(mu)));
end
% capacitances for the EKF, derived consistently from tau and R
LUT.C1_grid = LUT.tau1_grid ./ LUT.R1_grid;
LUT.C2_grid = LUT.tau2_grid ./ LUT.R2_grid;

params_table = T;
save(SAVE_PATH, 'LUT', 'gprModels', 'targetNames', ...
     'cvRMSE_gpr', 'cvRMSE_rf', 'params_table');
fprintf('\nSaved: %s\n', SAVE_PATH);

%% ------------------------------------------------------------------------
%  PLOT A — parameter surfaces
% -------------------------------------------------------------------------
pLabels = {'R_0 (\Omega)','R_1 (\Omega)','\tau_1 (s)','R_2 (\Omega)','\tau_2 (s)'};
figure('Name','GPR parameter surfaces','Position',[60,60,1400,800]);
for p = 1:5
    subplot(2,3,p);
    surf(SOC_GRID, TEMP_GRID, LUT.(gridNames{p}), 'EdgeColor','none');
    shading interp; colormap(parula); colorbar;
    xlabel('SOC (%)'); ylabel('T (\circC)'); zlabel(pLabels{p});
    title(targetNames{p}); view([-35,30]); grid on;
end
sgtitle('GPR lookup surfaces — 2-RC parameters vs (SOC, T)', 'FontWeight','bold');

%% ------------------------------------------------------------------------
%  PLOT B — per-temperature slices with measured points and ±2σ bands
% -------------------------------------------------------------------------
measTemps = unique(T.Temp_degC)';
colors = lines(numel(measTemps));
figure('Name','GPR slices vs measurements','Position',[120,120,1400,800]);
for p = 1:5
    subplot(2,3,p); hold on;
    for k = 1:numel(measTemps)
        tv = measTemps(k);
        [mu, sd] = predict(gprModels{p}, [SOC_GRID', repmat(tv,numel(SOC_GRID),1)]);
        yline_ = exp(mu);
        lo = exp(mu - 2*sd);  hi = exp(mu + 2*sd);
        fill([SOC_GRID, fliplr(SOC_GRID)], [lo', fliplr(hi')], colors(k,:), ...
            'FaceAlpha', 0.10, 'EdgeColor','none', 'HandleVisibility','off');
        plot(SOC_GRID, yline_, '-', 'Color', colors(k,:), 'LineWidth', 1.6, ...
            'DisplayName', sprintf('%d\\circC', tv));
        m = T.Temp_degC == tv;
        scatter(T.SOC_pct(m), exp(Y(m,p)), 40, colors(k,:), 'filled', ...
            'MarkerEdgeColor','k', 'HandleVisibility','off');
    end
    xlabel('SOC (%)'); ylabel(pLabels{p}); title(targetNames{p});
    set(gca,'YScale','log'); legend('Location','best','FontSize',6); grid on;
end
sgtitle('GPR predictions (lines, \pm2\sigma bands) vs measured (dots)', ...
    'FontWeight','bold');

fprintf('\nPhase 2 complete. Next: Phase3_EKF_Validation.m\n');
