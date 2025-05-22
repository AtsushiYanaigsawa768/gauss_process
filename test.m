%% 0) 事前準備 ── Simulink 側の設定
% 「To File」ブロックのパラメータを下記のようにしておく
%   File name : data_flexible.mat     % ★拡張子 .mat を付ける
%   Variable  : raw_flexible           % ★後で MATLAB からロードする名前
%   Save format: Array                 % （既定値）
%   Decimation : 1                     % サンプリングごとに保存
%   Sample time: -1                    % -1 なら元信号と同じ Ts
N_d = 100;% # of sampling frequencies (O(N^3))
N_for = 5;% O(N)
Sim_time = 60; % [sec] O(N)
f_low = -1.0;
f_up = 2.3;% log(250) = 2.3979.., 250Hz is the Nyquist frequency
gain_tuning = 1.0; % Define gain tuning parameter
%% 1) Simulink を 10 [s] だけ実行
model = 'system_id_new';   % ←あなたのモデル名（拡張子は不要）
open_system(model)         % 開いていない場合は開く

set_param(model,'StopTime','10')        % 10[s] で停止
sim(model,'ReturnWorkspaceOutputs','off') % 実行（ワークスペースに直接吐く）
% Note: Sim_time should be set more than 50/(10^f_low) in practice, more than 10/(10^f_low) sec for simulation.   
% Note: Sim_time should be set more than 50/(10^f_low) in practice, morethan 10/(10^f_low) sec for simulation.   
freq_range = sort(freq_range);% sorting
frequency = 10.^(f_low:(f_up - f_low)/N_d:f_up-(f_up - f_low)/N_d)';% if specific frequencies are necessary
phase = rand(size(frequency));
gain = gain_tuning*rand(size(frequency));
tic 
% qc_start_model;% Quarc - Uncomment if you have QuaRC library
set_param(model,'SimulationCommand', 'start')% To start Simulink simulation
toc
pause(Sim_time + 2);
set_param(model,'SimulationCommand', 'stop')
%% 2) MAT-file を読み取り，.dat に書き出し
load('data_flexible.mat','raw_flexible')   % ① MAT ファイル → 配列を取得
%  raw_flexible は [N×(1+n_signals)] の行列で
%   先頭列: 時刻ベクトル  2列目以降: 信号値になります

% For MATLAB R2019a and later:
try
	writematrix(raw_flexible,'data_flexible_10s.dat','Delimiter','tab'); % ② ASCII 保存
catch
	% For older MATLAB versions:
	dlmwrite('data_flexible_10s.dat', raw_flexible, 'delimiter', '\t');
end
fprintf('✓ data_flexible_10s.dat として保存しました\n')
disp('✓ data_flexibile_10s.dat として保存しました')
