%% fir_rt_identification.m  --------------------------------------------------------
% Real‑time identification of a flexible link using an FIR model.
% -------------------------------------------------------------------------
% Workflow
%   1)  Load complex frequency‑response data (FRF) measured by
%       repeated_daq.m ( variables  omega,  ReG,  ImG )
%   2)  Convert FRF → discrete‑time impulse response g[n] via inverse DFT
%   3)  Truncate / window g[n] ⇒ initial FIR coefficients  h_init
%   4)  Run a sample‑by‑sample loop, replaying a previously recorded
%       input sequence u[n] (or live DAQ) and
%          – predicting output  yhat[n]  with the current FIR
%          – updating FIR coefficients using Recursive Least Squares (RLS)
%          – streaming plots of measured vs. predicted motion
%   5)  After the run, compute RMSE, NRMSE, R² and display a summary
%
% REQUIREMENTS -------------------------------------------------------------
%   1)  frf_data.mat        –  complex FRF samples from repeated_daq.m
%                             ( omega [rad/s], ReG, ImG )
%   2)  trial_io_data.mat   –  replay I/O   ( u, y, t, dt )
%       *For live DAQ, replace the "replay" section with hardware reads.
%
% AUTHOR  : ChatGPT demo (2025‑05‑16)
% -------------------------------------------------------------------------

%% USER PARAMETERS  --------------------------------------------------------
frf_file   = 'frf_data.mat';       % <- FRF mat‑file
io_file    = 'trial_io_data.mat';  % <- recorded I/O to replay
lambda     = 0.995;                % RLS forgetting factor  (0.95…0.999)
energy_cut = 0.99;                 % keep ≥99 % of |g| energy
plot_rate  = 100;                  % samples between plot refreshes

%% 1) Load FRF -------------------------------------------------------------
load(frf_file,'omega','ReG','ImG');              % size N_d ×1
G_pos = ReG + 1j*ImG;

% Make a uniformly spaced frequency grid (required for plain IFFT).
Npos  = numel(omega);
omega_min = min(omega);
omega_max = max(omega);
Nfft  = 2^nextpow2( 4*Npos );                    % plenty of zero‑padding
omega_uni = linspace(omega_min,omega_max,Nfft/2+1).';

G_uni = interp1(omega,G_pos,omega_uni,'pchip',0);     % complex interp

% Build full Hermitian spectrum  (index 1 = DC)
G_full = [ conj(G_uni(end-1:-1:2)); G_uni ];          % length Nfft

%% 2) Impulse response via IFFT -------------------------------------------
g_full = real( ifft( ifftshift(G_full) ) );           % real impulse

% Time axis
Dw   = omega_uni(2)-omega_uni(1);
Fs   = Dw*Nfft/(2*pi);             % sampling frequency [Hz]
Ts   = 1/Fs;                       % Δt

% Trim g by cumulative energy
Etotal = sum( abs(g_full).^2 );
cumE   = cumsum( abs(g_full).^2 );
L      = find( cumE/Etotal >= energy_cut , 1 , 'first' );
L      = max(L,4);                 % at least 4 taps

% Window (Hann) and take first L taps
w      = hann(L);
h_init = g_full(1:L).*w;

fprintf('[INFO] FIR length  L = %d  (Δt = %.4g s)\n',L,Ts);

%% 3) Load I/O data to replay ---------------------------------------------
load(io_file,'u','y','dt');        % dt is original sampling period
if abs(dt - Ts) > 1e-6
    warning('dt in I/O data (%.4g) ≠ FIR Ts (%.4g). Resampling u,y.',dt,Ts);
    t_io = (0:length(u)-1).'*dt;
    t_fir= (0:ceil(t_io(end)/Ts)).'*Ts;
    u    = interp1(t_io,u,t_fir,'linear','extrap');
    y    = interp1(t_io,y,t_fir,'linear','extrap');
    dt   = Ts;   % now matched
end
N = length(u);

%% 4) RLS initialisation ---------------------------------------------------
Lvec = (L-1:-1:0).';               % for buffer indexing
h    = h_init(:);                  % current FIR coeffs (column)
P    = 1e4*eye(L);                 % covariance matrix
phi  = zeros(L,1);                 % regressor buffer

% Preallocate arrays for speed
yhat = zeros(N,1);
err  = zeros(N,1);

%% 5) Real‑time (offline replay) loop -------------------------------------
figure('Name','Real‑Time FIR Identification');
ax1 = subplot(2,1,1); hold on, grid on
h_meas = plot(NaN,NaN,'k');
h_pred = plot(NaN,NaN,'r--');
legend('Measured','Predicted')
xlabel('sample n'), ylabel('y');
ax2 = subplot(2,1,2); hold on, grid on
h_err  = plot(NaN,NaN,'b');
xlabel('sample n'), ylabel('error');

for n = 1:N
    %% Update regressor buffer  φ[n] = [u[n],u[n-1],...,u[n-L+1]]^T
    phi = [ u(n); phi(1:end-1) ];   % shift register

    if n >= L
        % Prediction ------------------------------------------------------
        yhat(n) = phi.' * h;
        err(n)  = y(n) - yhat(n);

        % RLS update ------------------------------------------------------
        K = (P*phi) / (lambda + phi.'*P*phi);
        h = h + K * err(n);
        P = (P - K*phi.'*P) / lambda;
    else
        yhat(n) = 0; err(n) = y(n);
    end

    %% Live plot refresh --------------------------------------------------
    if mod(n,plot_rate)==0 || n==N
        set(h_meas,'XData',1:n,'YData',y(1:n));
        set(h_pred,'XData',1:n,'YData',yhat(1:n));
        set(h_err ,'XData',1:n,'YData',err(1:n));
        drawnow limitrate
    end
end

%% 6) Error metrics --------------------------------------------------------
rmse  = sqrt( mean( err(L+1:end).^2 ) );
ynorm = y(L+1:end) - mean(y(L+1:end));
nrmse = 1 - norm(err(L+1:end))/norm(ynorm);
R2    = 1 - sum(err(L+1:end).^2)/sum(ynorm.^2);

fprintf('\n=====  FINAL ERROR  ====================================\n');
fprintf('RMSE   = %.4g\n',rmse);
fprintf('NRMSE  = %.2f %%\n',nrmse*100);
fprintf('R^2    = %.3f\n',R2);

% keep variables in workspace for further analysis
save('fir_rt_results.mat','h','rmse','nrmse','R2','yhat','err','Ts');