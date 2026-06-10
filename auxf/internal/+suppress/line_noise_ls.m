function data_emg = line_noise_ls(data_emg, time, fs, data_cx, data_di, cfg)
% Remove 60Hz line noise and harmonics using Least-Squares Regression.

if isempty(data_emg) || any(isnan(time)), return; end

% Persistent variables to cache filters for performance
persistent fir_cache last_fs last_order

% 0. Extract Configuration
if nargin < 6, cfg = struct(); end

% Defaults
t_ex_stim_start = -0.001; 
t_ex_stim_stop  = 0.015;
t_ex_mep_stop   = 0.075;
harmonics       = [60, 120];
independent_channels = false;

if isfield(cfg, 'line_suppression')
    ls = cfg.line_suppression;
    if isfield(ls, 't_exclude_stim_start'), t_ex_stim_start = double(ls.t_exclude_stim_start); end
    if isfield(ls, 't_exclude_stim_stop'),  t_ex_stim_stop  = double(ls.t_exclude_stim_stop); end
    if isfield(ls, 't_exclude_mep_stop'),   t_ex_mep_stop   = double(ls.t_exclude_mep_stop); end
    if isfield(ls, 'harmonics'),            harmonics       = double(ls.harmonics); end
    if isfield(ls, 'independent_channels'), independent_channels = logical(ls.independent_channels); end
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

% Prediction delay derived from total exclusion span
t_delay = t_ex_mep_stop - t_ex_stim_start;
N_delay = round(t_delay * fs);

% Ensure time is column vector
time = time(:);

% 1. Identify 'Excluded' windows
trig = (diff([0; data_cx]) > 0.5) | (diff([0; data_di]) > 0.5);
trig_indices = find(trig);

% Combined mask for training rejection
exclude_mask = false(size(time));
n_ex_start = round(t_ex_stim_start * fs);
n_ex_stop  = round(t_ex_mep_stop * fs);

for i = 1:length(trig_indices)
    idx = trig_indices(i);
    start_idx = max(1, idx + n_ex_start);
    end_idx   = min(length(time), idx + n_ex_stop);
    exclude_mask(start_idx : end_idx) = true;
end

% 2. Generate multi-harmonic reference signals
num_ch = size(data_emg, 2);
if independent_channels, n_iter = num_ch; else, n_iter = 1; end

for i_iter = 1:n_iter
    if independent_channels
        target_cols = i_iter;
        emg_ref_source = data_emg(:, i_iter);
    else
        target_cols = 1:num_ch;
        emg_ref_source = mean(data_emg, 2);
    end

    emg_ref_source(~isfinite(emg_ref_source)) = 0;

    % Mask out the triggered responses BEFORE filtering to prevent artifact 
    % energy from entering the reference phase. We use linear interpolation 
    % across the gap to minimize FIR filter ringing.
    for i = 1:length(trig_indices)
        idx = trig_indices(i);
        start_idx = max(1, idx + n_ex_start);
        end_idx   = min(length(time), idx + n_ex_stop);
        
        if start_idx > 1 && end_idx < length(time)
            val_start = emg_ref_source(start_idx - 1);
            val_end   = emg_ref_source(end_idx + 1);
            len = end_idx - start_idx + 1;
            emg_ref_source(start_idx:end_idx) = linspace(val_start, val_end, len);
        else
            emg_ref_source(start_idx:end_idx) = 0;
        end
    end

    % Bandpass parameters
    nyq = fs / 2;
    bw = 2; % +/- 1Hz bandwidth
    order = 200; 
    if length(emg_ref_source) < 3*order
        order = max(10, floor(length(emg_ref_source)/3) - 1);
    end

    % Reset cache if environment changed
    if isempty(fir_cache) || fs ~= last_fs || order ~= last_order
        fir_cache = containers.Map('KeyType', 'double', 'ValueType', 'any');
        last_fs = fs;
        last_order = order;
    end

    num_h = length(harmonics);
    ref_full = zeros(length(time), 2 * num_h);

    for ih = 1:num_h
        freq = harmonics(ih);
        
        % Get or design filter
        if ~fir_cache.isKey(freq)
            f_low = max(0.1, freq - bw);
            f_high = min(nyq - 0.1, freq + bw);
            fir_cache(freq) = fir1(order, [f_low, f_high]/nyq, 'bandpass');
        end
        
        % Filter and Hilbert
        s_filt = filtfilt(fir_cache(freq), 1, emg_ref_source);
        a_filt = hilbert(s_filt);
        
        % Add [cos, sin] pair
        ref_full(:, (ih-1)*2 + 1) = real(a_filt);
        ref_full(:, (ih-1)*2 + 2) = imag(a_filt);
    end

    % 3. Training: Learn the mapping from the reference to the EMG data
    valid_indices = find(all(isfinite(data_emg(:, target_cols)), 2) & all(isfinite(ref_full), 2));

    % A. Instantaneous Mapping (used for clean, baseline periods)
    train_mask_inst = ~exclude_mask(valid_indices);
    idx_train_inst = valid_indices(train_mask_inst);

    if isempty(idx_train_inst)
        if independent_channels, continue; else, return; end
    end
    weights_inst = ref_full(idx_train_inst, :) \ data_emg(idx_train_inst, target_cols);

    % B. Delayed Mapping (used to predict noise inside the artifact/MEP window from past phase)
    valid_target_delay = valid_indices(valid_indices > N_delay);
    train_mask_delay = ~exclude_mask(valid_target_delay) & ~exclude_mask(valid_target_delay - N_delay);
    idx_train_delay = valid_target_delay(train_mask_delay);

    if isempty(idx_train_delay)
        weights_delay = weights_inst; % Fallback if not enough data
    else
        weights_delay = ref_full(idx_train_delay - N_delay, :) \ data_emg(idx_train_delay, target_cols);
    end

    % 4. Denoise
    % Start with the instantaneous prediction for all time points
    noise_pred = ref_full * weights_inst;

    % Overwrite with the delayed prediction ONLY inside the exclusion windows.
    % This prevents us from using the gap-corrupted instantaneous reference,
    % and safely predicts the noise from the phase *before* the stimulation occurred.
    for i = 1:length(trig_indices)
        idx = trig_indices(i);
        start_idx = max(1, idx + n_ex_start);
        end_idx   = min(length(time), idx + n_ex_stop);
        
        win_idx = start_idx : end_idx;
        
        source_idx = win_idx - N_delay;
        valid_mask = source_idx > 0;
        
        win_idx = win_idx(valid_mask);
        source_idx = source_idx(valid_mask);
        
        if ~isempty(win_idx)
            noise_pred(win_idx, :) = ref_full(source_idx, :) * weights_delay;
        end
    end

    % Subtract the predicted noise from the specific channel(s)
    data_emg(:, target_cols) = data_emg(:, target_cols) - noise_pred;
end

end
