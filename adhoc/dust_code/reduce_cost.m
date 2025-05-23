%% fir_rt_identification_csv.m
% Real-time identification of a flexible link using an FIR model,
% FRF from CSV, I/O from .mat, end with savemat.

clearvars; close all; clc;

%% USER PARAMETERS
frf_csv      = 'predicted_G_values.csv';  % FRF in CSV: [omega, ReG, ImG]
io_file      = 'data_hour.mat';                     % recorded I/O to replay
lambda       = 0.995;                               % RLS forgetting factor
energy_cut   = 0.99;                                % keep ≥99% of |g| energy
plot_rate    = 100;                                 % samples between plot refreshes

%% 1) Load FRF from CSV
% 1行目がヘッダの場合は 'NumHeaderLines',1 を追加
M = readmatrix(frf_csv);           
omega = M(:,1);
ReG   = M(:,2);
ImG   = M(:,3);
G_pos = ReG + 1j*ImG;

% 均一周波数グリッド
Npos      = numel(omega);
omega_min = min(omega);
omega_max = max(omega);
Nfft      = 2^nextpow2(4*Npos);
omega_uni = linspace(omega_min,omega_max,Nfft/2+1).';

% 補間
G_uni = interp1(omega,G_pos,omega_uni,'pchip',0);

% Hermitian 対称スペクトルの生成
G_full = [ conj(G_uni(end-1:-1:2)); G_uni ];

%% 2) インパルス応答 via IFFT
g_full = real( ifft( ifftshift(G_full) ) );

% サンプリング周波数
Dw = omega_uni(2)-omega_uni(1);
Fs = Dw*Nfft/(2*pi);
Ts = 1/Fs;

% エネルギーでトリミング
Etotal = sum(abs(g_full).^2);
cumE   = cumsum(abs(g_full).^2);
L      = find(cumE/Etotal>=energy_cut,1,'first');
L      = max(L,4);

% 窓掛け
w      = hann(L);
h_init = g_full(1:L).*w;

fprintf('[INFO] FIR length L = %d  (Ts = %.4g s)\n',L,Ts);

%% 3) Load I/O data to replay
S = load(io_file);
fn = fieldnames(S);
% __ で始まるものを除外
fn = fn(~startsWith(fn,'__'));
mat = S.(fn{1});    % 最初の変数を取得

% mat は 3×N または N×3 を仮定
if size(mat,1)>=3 && size(mat,2)>3
  % 3行： time; y; u
  time = mat(1,:).';
  y    = mat(2,:).';
  u    = mat(3,:).';
elseif size(mat,2)>=3
  % 3列： [time,y,u]
  time = mat(:,1);
  y    = mat(:,2);
  u    = mat(:,3);
else
  error('I/O データのサイズが想定外です');
end

% dt の計算
if numel(time)>1
  dtv = diff(time);
  dt  = mean(dtv);
  if max(abs(dtv-dt)) > 0.01*dt
    warning('Non-uniform time steps detected.');
  end
else
  dt = Ts;
end
fprintf('Loaded I/O: %d samples, dt = %.4g s\n',numel(time),dt);

% 必要なら再サンプリング
if abs(dt-Ts)>1e-6
  fprintf('Resampling to FIR Ts...\n');
  t_old = time;
  t_new = (0:ceil(time(end)/Ts)).'*Ts;
  u = interp1(t_old,u,t_new,'linear','extrap');
  y = interp1(t_old,y,t_new,'linear','extrap');
  dt = Ts;
end
N = numel(u);

%% 4) RLS 初期化
%% 4) Partial‐Update LMS initialisation
h     = h_init(:);           % FIR taps
mu    = 1e-3;                % LMS stepsize (tune to your data)
M     = 10;                  % number of taps to update per sample
phi   = zeros(L,1);

yhat  = zeros(N,1);
err   = zeros(N,1);

%% 5) Real‐time loop with M‐Max partial‐update LMS
figure('Name','Partial‐Update LMS Identification');
ax1 = subplot(2,1,1); hold on; grid on;
h_meas = plot(NaN,NaN,'k');
h_pred = plot(NaN,NaN,'r--');
legend('Measured','Predicted');
xlabel('n'); ylabel('y');

ax2 = subplot(2,1,2); hold on; grid on;
h_err = plot(NaN,NaN,'b');
xlabel('n'); ylabel('error');

ax3 = subplot(3,1,3); 
h_coef = plot(1:L, h, 'b-o', 'MarkerSize',3); 
grid on;
xlabel('Tap index');
ylabel('h');
title(ax3,'Iteration 0');
ylim([min(h_init)*1.1, max(h_init)*1.1]); % 適宜レンジ調整

for n = 1:N
  % update input vector
  phi = [u(n); phi(1:end-1)];
  
  % prediction and error
  yhat(n) = phi.' * h;
  err(n)  = y(n) - yhat(n);
  
  % partial‐update LMS
  delta = mu * phi * err(n);
  [~, idx_sort] = sort(abs(delta), 'descend');
  S = idx_sort(1:min(M, L));
  h(S) = h(S) + delta(S);

  % store coefficient history
  H(:,n) = h;

  % update plots periodically
  if mod(n, plot_rate) == 0 || n == N
    set(h_meas, 'XData', 1:n, 'YData', y(1:n));
    set(h_pred, 'XData', 1:n, 'YData', yhat(1:n));
    set(h_err , 'XData', 1:n, 'YData', err(1:n));
    set(h_coef, 'YData', h);
    title(ax3, sprintf('Iteration %d', n));
    drawnow limitrate;
  end
end

%% 6) Error metrics & save (unchanged)
rmse  = sqrt(mean(err(L+1:end).^2));
yn    = y(L+1:end)-mean(y(L+1:end));
nrmse = 1 - norm(err(L+1:end))/norm(yn);
R2    = 1 - sum(err(L+1:end).^2)/sum(yn.^2);

fprintf('\n===== FINAL ERROR =====\n');
fprintf('RMSE  = %.4g\n',rmse);
fprintf('NRMSE = %.2f %%\n',nrmse*100);
fprintf('R^2   = %.3f\n',R2);

save('pu_lms_results.mat','h','rmse','nrmse','R2','yhat','err','Ts');
