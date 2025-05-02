
clear; clc;

%% ---------------- ユーザ設定 ------------------------
noiseFilter    = false;               % true: Hampel で外れ値除去 (falseに変更してテスト、必要ならtrueに戻す)
pngName        = "const_RBF_white";   % 出力 PNG (result フォルダに保存)
calculateTime  = true;                % 計算時間を表示
testSetRatio   = 0.8;                 % テストデータの割合 (訓練データは 1-testSetRatio = 0.2)

% GPR カーネル : 係数付き RBF(=SquaredExponential) + WhiteNoise
kernelFcn      = 'squaredexponential';   % RBF と同等
sigma0         = 0.1;                    % 事前ノイズの初期値
numRestart     = 9;                      % ハイパーパラメータ最適化の再起動回数
%% ----------------------------------------------------

% --- データ読み込み ---
% 'result/merged.dat'ファイルが存在することを確認してください
dataPath = fullfile('result', 'merged.dat');
if ~exist(dataPath, 'file')
    error('データファイルが見つかりません: %s', dataPath);
end
tmp = importdata(dataPath, ',');

% データが期待通りに読み込まれているか確認 (例: 3行N列)
if size(tmp, 1) ~= 3
    error('データファイルの形式が不正です。3行 (omega, gain, phase) である必要があります。');
end

% --- データ整理 ---
% 周波数でソート
[omega, idx] = sort(tmp(1,:));
SysGain_raw = tmp(2,idx);
argG_raw    = tmp(3,idx);

% データの要素数 (サンプル数) を表示
n = length(omega);
disp(['データのサンプル数: ', num2str(n)]);

% --- ノイズ除去 (オプション) ---
if noiseFilter
    % Signal Processing Toolbox の hampel を試す
    if license('test', 'Signal_Toolbox') && exist('hampel', 'file')
        disp('Hampel フィルタ (Signal Processing Toolbox) を使用します。');
        SysGain = hampel(SysGain_raw, 15); % Hampelフィルタ適用 (ゲイン)
        argG    = hampel(argG_raw, 15);    % Hampelフィルタ適用 (位相)
    else
        disp('Hampel フィルタ (自家製) を使用します。');
        % 自家製 Hampel フィルタを使用する場合
        SysGain = hampelCustom(SysGain_raw, 15);
        argG    = hampelCustom(argG_raw, 15);
    end
else
    disp('Hampel フィルタは適用されません。');
    SysGain = SysGain_raw;
    argG    = argG_raw;
end
% 複素ゲイン (プロット等では使用しないが参考のため)
% G = SysGain .* exp(1i*argG); % 使わないのでコメントアウト

%% 学習データ準備
% --- 特徴量と目標値の作成 ---
% fitrgp は列ベクトルを期待するため、ここで転置 (.').
X = log10(omega).';           % 特徴量 : log10(ω) -> N x 1 列ベクトル
Y = 20*log10(SysGain).';      % 目標値 : 20*log10|G| -> N x 1 列ベクトル

% X と Y のサイズ確認 (デバッグ用)
disp(['Xのサイズ: ', num2str(size(X))]); % -> N x 1 になるはず
disp(['Yのサイズ: ', num2str(size(Y))]); % -> N x 1 になるはず

if size(X, 1) ~= size(Y, 1)
    error('内部エラー: X と Y の行数 (サンプル数) が一致しません。データ処理を確認してください。');
end
if size(X,1) ~= n || size(Y,1) ~= n
    error('内部エラー: X, Y のサンプル数が元のデータ数と一致しません。');
end


% --- 訓練 / テスト分割 (80‑20) ---
% testSetRatio はテストデータの割合
nTrain = floor(n * (1-testSetRatio)); % 訓練データの数 (例: 0.2 * n)
nTest  = n - nTrain;                  % テストデータの数 (例: 0.8 * n)

% ランダムなインデックスを生成して分割
idx = randperm(n);
trainIdx = idx(1:nTrain);
testIdx  = idx(nTrain+1:end);

% 列ベクトルとして分割
XTrain = X(trainIdx, :); % -> nTrain x 1
YTrain = Y(trainIdx, :); % -> nTrain x 1
XTest  = X(testIdx, :);  % -> nTest x 1
YTest  = Y(testIdx, :);  % -> nTest x 1

% 分割後のサイズ確認 (デバッグ用)
disp(['訓練データ数: ', num2str(nTrain)]);
disp(['テストデータ数: ', num2str(nTest)]);
disp(['XTrainのサイズ: ', num2str(size(XTrain))]);
disp(['YTrainのサイズ: ', num2str(size(YTrain))]);
disp(['XTestのサイズ: ', num2str(size(XTest))]);
disp(['YTestのサイズ: ', num2str(size(YTest))]);

if isempty(XTrain) || isempty(YTrain)
    error('訓練データが空です。データ数または testSetRatio を確認してください。');
end

%% GPR モデル学習
if calculateTime, startT = tic; end

% --- ハイパーパラメータ最適化あり ---
% ★★★ 修正点: fitrgp には訓練データ (XTrain, YTrain) を渡す ★★★
gprMdl = fitrgp( ...
    XTrain, YTrain, ... % ここを修正！ X, Y ではなく XTrain, YTrain を使う
    'KernelFunction',      kernelFcn,   ...
    'Sigma',               sigma0,      ...
    'Standardize',         true,        ...
    'OptimizeHyperparameters', 'all', ...
    'HyperparameterOptimizationOptions', ...
        struct(...
            'MaxObjectiveEvaluations', numRestart, ...
            'ShowPlots', false, ...
            'Verbose',   0) ...
);


if calculateTime
    elapsedT = toc(startT);
    fprintf("Elapsed time : %.2f s\n", elapsedT);
end

%% 予測 (細かい周波数グリッド & テスト / 訓練)
% --- 細かいグリッドでの予測 ---
% logspace はデフォルトで列ベクトルを生成する
omegaFine = logspace(log10(min(omega)), log10(max(omega)), 500).';
XFine = log10(omegaFine); % predict に渡すため、log10 を適用 (すでに列ベクトル)

% 予測実行
[YPredFine, YSD] = predict(gprMdl, XFine); % XFine は N_fine x 1

% --- 訓練/テストデータでの予測 ---
YPredTrain = predict(gprMdl, XTrain); % XTrain は nTrain x 1
YPredTest  = predict(gprMdl, XTest);  % XTest は nTest x 1

% MSE 計算
mseTrain = mean((YTrain - YPredTrain).^2);
mseTest  = mean((YTest  - YPredTest ).^2);
fprintf("Training MSE : %.4f\nTest MSE     : %.4f\n", mseTrain, mseTest);

%% プロット 1 : Bode ゲイン + GPR
figure('Units','pixels','Position',[100 100 800 520]);
hold on; grid on; box on;

% 元データ (フィルタ前)
semilogx(omega, 20*log10(SysGain_raw), 'b.', 'MarkerSize',4, ...
    'DisplayName','Raw data'); % HandleVisibility を変更して凡例に表示

% 訓練 / テスト (フィルタ後のデータ)
% YTrain, YTest は既に 20*log10(SysGain) 相当なのでそのまま使用
semilogx(10.^XTrain, YTrain, 'ro', 'MarkerSize',6, ...
    'DisplayName','Training data');
semilogx(10.^XTest , YTest , 'ms', 'MarkerSize',6, ...
    'DisplayName','Test data');

% GPR 予測平均
semilogx(omegaFine, YPredFine, 'g-', 'LineWidth',2, ...
    'DisplayName','GPR prediction');

% 信頼区間 (±2σ)
semilogx(omegaFine, YPredFine + 2*YSD, 'g--', 'HandleVisibility','off');
semilogx(omegaFine, YPredFine - 2*YSD, 'g--', 'HandleVisibility','off');
fill([omegaFine; flipud(omegaFine)], [YPredFine - 2*YSD; flipud(YPredFine + 2*YSD)], ...
     'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', '\pm 2\sigma Confidence Interval');