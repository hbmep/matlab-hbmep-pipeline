function [data_emg, state_out] = line_noise_kalman(data_emg, time, fs, data_cx, data_di, cfg, state_in)
% suppress_line_noise_kalman - Remove line noise and harmonics using a Kalman Filter.
%
% Syntax:
%   data_emg = suppress.line_noise_kalman(data_emg, time, fs, data_cx, data_di, cfg)
%   [data_emg, state_out] = suppress.line_noise_kalman(data_emg, time, fs, data_cx, data_di, cfg, state_in)

persistent kalman_x kalman_P kalman_A kalman_H kalman_Q kalman_R last_fs last_harmonics last_num_ch warned_window

use_explicit_state = (nargin >= 7);
if use_explicit_state
    state_out = state_in;
else
    state_out = struct();
end

if isempty(data_emg)
    return;
end
if ~use_explicit_state && any(isnan(time))
    return;
end

% 0. Extract Configuration
if nargin < 6 || isempty(cfg), cfg = struct(); end

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
        if isempty(warned_window)
            fprintf('[WARN] Line suppression: MEP analysis window [%0.3f, %0.3f] is not fully contained in suppression rejection window [%0.3f, %0.3f]. Leakage may occur.\n', ...
                m_start, m_stop, t_ex_stim_stop, t_ex_mep_stop);
            warned_window = true;
        end
    end
end

% Ensure time is column vector
time = time(:);

% =========================================================================
% PATH A: Explicit State Tracking Across Chunks / Sliding Windows
% =========================================================================
if use_explicit_state
    valid_t_mask = ~isnan(time);
    if ~any(valid_t_mask)
        return;
    end

    valid_idx = find(valid_t_mask);
    sub_time  = time(valid_idx);
    sub_emg   = data_emg(valid_idx, :);
    sub_cx    = data_cx(valid_idx);
    sub_di    = data_di(valid_idx);

    N = numel(sub_time);
    num_ch = size(sub_emg, 2);
    num_h = length(harmonics);
    dim_x = 2 * num_h;

    % Check if state_in is valid and matches current filter parameters
    is_valid_state = isstruct(state_in) && isfield(state_in, 'x') && ~isempty(state_in.x) && ...
        isfield(state_in, 'fs') && state_in.fs == fs && ...
        isfield(state_in, 'harmonics') && isequal(state_in.harmonics(:), harmonics(:)) && ...
        isfield(state_in, 'q_cov') && state_in.q_cov == q_cov && ...
        isfield(state_in, 'r_cov') && state_in.r_cov == r_cov && ...
        isfield(state_in, 'num_ch') && state_in.num_ch == num_ch && ...
        isfield(state_in, 'last_time') && (sub_time(end) >= state_in.last_time - 0.5 / fs);

    if ~is_valid_state
        A_mat = zeros(dim_x, dim_x);
        for i = 1:num_h
            omega = 2 * pi * harmonics(i) / fs;
            idx = (i - 1) * 2 + 1;
            A_mat(idx:idx+1, idx:idx+1) = [cos(omega), -sin(omega); sin(omega), cos(omega)];
        end
        H_mat = zeros(1, dim_x);
        H_mat(1:2:end) = 1;

        state = struct();
        state.A = A_mat;
        state.H = H_mat;
        state.Q = q_cov * eye(dim_x);
        state.R = r_cov;
        state.x = zeros(dim_x, num_ch);
        state.P = 10 * eye(dim_x);
        state.fs = fs;
        state.harmonics = harmonics;
        state.q_cov = q_cov;
        state.r_cov = r_cov;
        state.num_ch = num_ch;
        state.last_time = -inf;
        state.last_cx = 0;
        state.last_di = 0;
        state.recent_trig_times = zeros(0, 1);
        state.window_time = zeros(0, 1);
        state.window_noise_pred = zeros(0, num_ch);
        state.hist_time = zeros(0, 1);
        state.hist_x = zeros(dim_x, num_ch, 0);
        state.hist_P = zeros(dim_x, dim_x, 0);
    else
        state = state_in;
    end

    dt_tol = 0.25 / fs;
    new_mask = sub_time > (state.last_time + dt_tol);

    % If no new samples arrived (identical window), reuse cached noise prediction
    if ~any(new_mask)
        if numel(state.window_time) == N && all(abs(state.window_time - sub_time) <= dt_tol)
            data_emg(valid_idx, :) = sub_emg - state.window_noise_pred;
        end
        state_out = state;
        return;
    end

    idx_first_new = find(new_mask, 1, 'first');

    % Check if overlapping prefix (1 : idx_first_new - 1) matches cached window tail
    prefix_len = idx_first_new - 1;
    can_reuse_prefix = false;
    if prefix_len == 0
        can_reuse_prefix = true;
    elseif numel(state.window_time) >= prefix_len
        cached_tail_t = state.window_time(end - prefix_len + 1 : end);
        if all(abs(cached_tail_t - sub_time(1:prefix_len)) <= dt_tol)
            can_reuse_prefix = true;
        end
    end

    if ~can_reuse_prefix
        % Disjoint jump backwards or non-matching window: re-filter full sub_time cleanly
        idx_first_new = 1;
        prefix_len = 0;
        state.x = zeros(dim_x, num_ch);
        state.P = 10 * eye(dim_x);
        state.last_time = -inf;
        state.last_cx = 0;
        state.last_di = 0;
        state.recent_trig_times = zeros(0, 1);
        state.hist_time = zeros(0, 1);
        state.hist_x = zeros(dim_x, num_ch, 0);
        state.hist_P = zeros(dim_x, dim_x, 0);
    end

    % Detect rising-edge triggers on the new samples (including boundary edge from previous sample)
    if idx_first_new > 1
        c_prev = sub_cx(idx_first_new - 1);
        d_prev = sub_di(idx_first_new - 1);
    else
        c_prev = state.last_cx;
        d_prev = state.last_di;
    end
    trig_new = (diff([c_prev; sub_cx(idx_first_new:N)]) > 0.5) | ...
               (diff([d_prev; sub_di(idx_first_new:N)]) > 0.5);
    new_trig_times = sub_time(idx_first_new - 1 + find(trig_new));
    all_trig_times = unique([state.recent_trig_times(:); new_trig_times(:)]);

    % Default start of Kalman stepping is idx_first_new
    idx_start = idx_first_new;
    state_x   = state.x;
    state_P   = state.P;

    % If a new trigger occurred near the start of the new chunk such that its pre-trigger
    % exclusion window (t_ex_stim_start, e.g. -1 ms) reaches into the end of the previous chunk,
    % roll back to the saved pre-trigger snapshot if available.
    if prefix_len > 0 && ~isempty(new_trig_times) && (min(new_trig_times) + t_ex_stim_start <= state.last_time + dt_tol)
        if ~isempty(state.hist_time)
            t_need = min(new_trig_times) + t_ex_stim_start;
            rb_idx = find(state.hist_time <= t_need + dt_tol, 1, 'last');
            if isempty(rb_idx)
                rb_idx = 1;
            end
            t_rb = state.hist_time(rb_idx);
            match_pos = find(abs(sub_time(1:prefix_len) - t_rb) <= dt_tol, 1, 'first');
            if ~isempty(match_pos)
                idx_start = match_pos + 1;
                state_x   = state.hist_x(:, :, rb_idx);
                state_P   = state.hist_P(:, :, rb_idx);
            end
        end
    end

    noise_pred = zeros(N, num_ch);
    if idx_start > 1
        reuse_len = idx_start - 1;
        noise_pred(1:reuse_len, :) = state.window_noise_pred(end - prefix_len + 1 : end - prefix_len + reuse_len, :);
    end

    % Build exclusion mask for idx_start : N
    n_step = N - idx_start + 1;
    step_time = sub_time(idx_start:N);
    exclude_step = any(isnan(sub_emg(idx_start:N, :)), 2);
    for i = 1:numel(all_trig_times)
        t_tr = all_trig_times(i);
        exclude_step = exclude_step | ...
            (step_time >= (t_tr + t_ex_stim_start - dt_tol) & step_time <= (t_tr + t_ex_mep_stop + dt_tol));
    end

    state_A = state.A;
    state_H = state.H;
    state_Q = state.Q;
    state_R = state.R;

    % Keep a short rollback history (last n_rb samples) for pre-trigger window across chunks
    n_rb = max(2, abs(round(t_ex_stim_start * fs)) + 2);
    step_x_hist = zeros(dim_x, num_ch, n_step);
    step_P_hist = zeros(dim_x, dim_x, n_step);

    for k = 1:n_step
        t_idx = idx_start + k - 1;

        % Time Update (Predict next state)
        x_pred = state_A * state_x;
        P_pred = state_A * state_P * state_A' + state_Q;

        if ~exclude_step(k)
            % Measurement Update (Correct with actual EMG)
            y_pred = state_H * x_pred;
            err = sub_emg(t_idx, :) - y_pred;

            S = state_H * P_pred * state_H' + state_R;
            K = (P_pred * state_H') / S;

            state_x = x_pred + K * err;
            state_P = P_pred - K * (state_H * P_pred);
        else
            % Skip measurement update, purely extrapolate noise state
            state_x = x_pred;
            state_P = P_pred;
        end

        noise_pred(t_idx, :) = state_H * state_x;
        step_x_hist(:, :, k) = state_x;
        step_P_hist(:, :, k) = state_P;
    end

    % Update rollback history buffer (keep up to n_rb most recent samples)
    if n_step >= n_rb
        state.hist_time = step_time(end - n_rb + 1 : end);
        state.hist_x    = step_x_hist(:, :, end - n_rb + 1 : end);
        state.hist_P    = step_P_hist(:, :, end - n_rb + 1 : end);
    else
        keep_old = state.hist_time < step_time(1) - dt_tol;
        comb_t = [state.hist_time(keep_old); step_time];
        comb_x = cat(3, state.hist_x(:, :, keep_old), step_x_hist);
        comb_P = cat(3, state.hist_P(:, :, keep_old), step_P_hist);
        if numel(comb_t) > n_rb
            comb_t = comb_t(end - n_rb + 1 : end);
            comb_x = comb_x(:, :, end - n_rb + 1 : end);
            comb_P = comb_P(:, :, end - n_rb + 1 : end);
        end
        state.hist_time = comb_t;
        state.hist_x    = comb_x;
        state.hist_P    = comb_P;
    end

    % Update state_out for next chunk
    state.x = state_x;
    state.P = state_P;
    state.last_time = sub_time(N);
    state.last_cx   = sub_cx(N);
    state.last_di   = sub_di(N);
    if ~isempty(all_trig_times)
        state.recent_trig_times = all_trig_times(all_trig_times + t_ex_mep_stop >= sub_time(N) - 1 / fs);
    else
        state.recent_trig_times = zeros(0, 1);
    end
    state.window_time = sub_time;
    state.window_noise_pred = noise_pred;

    data_emg(valid_idx, :) = sub_emg - noise_pred;
    state_out = state;
    return;
end

% =========================================================================
% PATH B: Legacy 6-Argument Mode (Unperturbed Persistent State)
% =========================================================================
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

if nargout >= 2
    state_out = struct('x', kalman_x, 'P', kalman_P, 'A', kalman_A, ...
        'H', kalman_H, 'Q', kalman_Q, 'R', kalman_R, ...
        'fs', fs, 'harmonics', harmonics, 'num_ch', num_ch);
end

end
