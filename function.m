function y_hat = fir_rt_id(u, y_meas)
%#codegen
% Real-time FIR identification (partial-update LMS) as a Simulink MATLAB Function.
% Inputs :
%   u       – plant input  (scalar)
%   y_meas  – measured plant output (scalar)
% Output:
%   y_hat   – model-predicted output (scalar)

% ──────────────────────────────────────────────────────────────
% PERSISTENT VARIABLES (kept across major time steps)
% ──────────────────────────────────────────────────────────────
persistent h        % FIR tap vector           (L×1)
persistent phi      % regressor buffer         (L×1) = [u(n); … ; u(n-L+1)]
persistent L mu M   % algorithm parameters
persistent initDone % flag

% ---------- INITIALIZATION (executes only at t = 0) ----------
if isempty(initDone)
    % User-tunable parameters
    lambda       = 0.995;  %#ok<NASGU>  (not used in PU-LMS, kept for reference)
    energy_cut   = 0.99;   % keep 99 % of |g| energy when trimming
    mu           = 1e-3;   % LMS stepsize
    M            = 10;     % taps updated per sample (M-max partial update)
    
    % --- 1) Read FRF & build initial FIR ---------------------
    %  ※ CSV や MAT の読み込みは run-time １回だけ行う。
    %     コード生成対象ならフラッシュ ROM に h_init を焼く方が安全。
    frf_csv = 'predicted_G_values.csv';   % [omega, ReG, ImG]
    data    = readmatrix(frf_csv,'NumHeaderLines',1);
    
    omega   = data(:,1);
    G_pos   = data(:,2) + 1j*data(:,3);
    
    Npos      = numel(omega);
    Nfft      = 2^nextpow2(4*Npos);
    omega_uni = linspace(min(omega),max(omega),Nfft/2+1).';
    
    G_uni  = interp1(omega,G_pos,omega_uni,'pchip',0);
    G_full = [conj(G_uni(end-1:-1:2)); G_uni];
    g_full = real(ifft(ifftshift(G_full)));
    
    % trimming
    Etot = sum(abs(g_full).^2);
    L    = find(cumsum(abs(g_full).^2)/Etot >= energy_cut,1,'first');
    L    = max(L,4);
    
    % Hann window
    h_init   = g_full(1:L).*hann(L);
    
    %  --- 2) Persistents ------------------------------------
    h        = h_init(:);          % L×1
    phi      = zeros(L,1);         % L×1
    initDone = true;
end
% ---------- END INITIALIZATION -------------------------------

% ──────────────────────────────────────────────────────────────
%  ONLINE SECTION (executes every major step)
% ──────────────────────────────────────────────────────────────
% 1. Shift-in newest input sample into regressor
phi = [u; phi(1:end-1)];

% 2. Predict output
y_hat = phi.' * h;

% 3. Error
e = y_meas - y_hat;

% 4. Partial-update LMS (M-max)
delta = mu * phi * e;                  % full-update delta
[~,idx] = sort(abs(delta),'descend');  % largest-magnitude indices
sel     = idx(1:min(M,L));             % selection mask
h(sel)  = h(sel) + delta(sel);         % update only selected taps

% (Optional) expose e as second output:
%   function [y_hat,e] = fir_rt_id(u,y_meas)
%   and add "e" to the outputs list.
end