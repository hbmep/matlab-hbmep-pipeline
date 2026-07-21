function data_emg = line_noise_kalman(data_emg, time, fs, data_cx, data_di, cfg)
% suppress_line_noise_kalman - Remove line noise and harmonics using a Kalman Filter.


if isempty(data_emg) || any(isnan(time)), return; end

% 0. Extract Configuration
if nargin < 6, cfg = struct(); end

% Defaults
t_ex_stim_start = -0.001; 
t_ex_stim_stop  = 0.015;
t_ex_mep_stop   = 0.075;
harmonics       = [60, 120];
q_cov           = 5e-7; % Process noise (could try 1e-7 if filtering is too broad)
r_cov           = 1;    % Measurement noise 

if isfield(cfg, 'line_suppression')
    ls = cfg.line_suppression;
    if isfield(ls, 't_exclude_stim_start'), t_ex_stim_start = double(ls.t_exclude_stim_start); end
    if isfield(ls, 't_exclude_stim_stop'),  t_ex_stim_stop  = double(ls.t_exclude_stim_stop); end
    if isfield(ls, 't_exclude_mep_stop'),   t_ex_mep_stop   = double(ls.t_exclude_mep_stop); end
    if isfield(ls, 'harmonics'),            harmonics       = double(ls.harmonics); end
    if isfield(ls, 'kalman_q'),             q_cov           = double(ls.kalman_q); end
    if isfield(ls, 'kalman_r'),             r_cov           = double(ls.kalman_r); end
end

% Check analysis window nesting
if isfield(cfg, 'mep') && isfield(cfg.mep, 'mep_window_start') && isfield(cfg.mep, 'mep_window_stop')
    m_start = double(cfg.mep.mep_window_start);
    m_stop  = double(cfg.mep.mep_window_stop);
    if (m_start < t_ex_stim_stop) || (m_stop > t_ex_mep_stop)
        persistent warned_window
        if isempty(warned_window)
            fprintf('[WARN] Line suppression: MEP analysis window [%0.3f, %0.3f] is not fully contained in suppression rejection window [%0.3f, %0.3f]. Leakage may occur.\n', ...
                m_start, m_stop, t_ex_stim_stop, t_ex_mep_stop);
            warned_window = true;
        end
    end
end

% Ensure time is column vector
time = time(:);
num_samples = size(data_emg, 1);
num_ch = size(data_emg, 2);

% 1. Identify 'Excluded' windows (Artifacts / MEPs)
trig = (diff([0; data_cx]) > 0.5) | (diff([0; data_di]) > 0.5);
trig_indices = find(trig);

exclude_mask = false(num_samples, 1);
n_ex_start = round(t_ex_stim_start * fs);
n_ex_stop  = round(t_ex_mep_stop * fs);

for i = 1:length(trig_indices)
    idx = trig_indices(i);
    start_idx = max(1, idx + n_ex_start);
    end_idx   = min(num_samples, idx + n_ex_stop);
    exclude_mask(start_idx : end_idx) = true;
end

% Handle existing NaNs in data (treat as missing measurements)
nan_mask = any(isnan(data_emg), 2);
exclude_mask = exclude_mask | nan_mask;

% 2. Kalman Filter Initialization
persistent kalman_x kalman_P kalman_A kalman_H kalman_Q kalman_R last_fs last_harmonics last_num_ch

num_h = length(harmonics);
dim_x = 2 * num_h;

% Reset state if environment or parameters change
if isempty(kalman_x) || isempty(last_fs) || last_fs ~= fs || ...
   length(last_harmonics) ~= num_h || any(last_harmonics ~= harmonics) || ...
   isempty(last_num_ch) || last_num_ch ~= num_ch
    
    kalman_A = zeros(dim_x, dim_x);
    for i = 1:num_h
        omega = 2 * pi * harmonics(i) / fs;
        idx = (i-1)*2 + 1;
        kalman_A(idx:idx+1, idx:idx+1) = [cos(omega), -sin(omega); sin(omega), cos(omega)];
    end
    
    kalman_H = zeros(1, dim_x);
    kalman_H(1:2:end) = 1;
    
    kalman_Q = q_cov * eye(dim_x);
    kalman_R = r_cov;
    
    kalman_x = zeros(dim_x, num_ch);
    kalman_P = 10 * eye(dim_x); % Initial uncertainty
    
    last_fs = fs;
    last_harmonics = harmonics;
    last_num_ch = num_ch;
end

% 3. Kalman Filter Loop
noise_pred = zeros(num_samples, num_ch);

for t = 1:num_samples
    % Time Update (Predict next state)
    x_pred = kalman_A * kalman_x;
    P_pred = kalman_A * kalman_P * kalman_A' + kalman_Q;
    
    if ~exclude_mask(t)
        % Measurement Update (Correct with actual EMG)
        y_pred = kalman_H * x_pred;
        err = data_emg(t, :) - y_pred;
        
        S = kalman_H * P_pred * kalman_H' + kalman_R;
        K = (P_pred * kalman_H') / S;
        
        kalman_x = x_pred + K * err;
        kalman_P = P_pred - K * (kalman_H * P_pred);
    else
        % Skip measurement update, purely extrapolate noise state
        kalman_x = x_pred;
        kalman_P = P_pred;
    end
    
    % Track the predicted noise
    noise_pred(t, :) = kalman_H * kalman_x;
end

% 4. Denoise (Subtraction)
data_emg = data_emg - noise_pred;

end