function [p_csv_out, cfg_table] = write_ramp_tables(raw_table, trigger_table, cfg_loaded, cfg_table)
% WRITE_RAMP_TABLES Process recruitment curve trials, compute MEP size / AUC, and export tables & figures
%
% Supports both single-pulse stimuli and multi-pulse stimulus trains (e.g. 60 pulses over 0.5s).
% For multi-pulse trains with AUC ('uVs'):
%   1. Blanks out stimulation artifacts around each trigger ([t_blank_pre, t_blank_post]).
%   2. Replaces artifacts with linear interpolation.
%   3. Expands the AUC window from t_min_size (relative to 1st pulse) to t_max_size (relative to last pulse).
%   4. Normalizes train AUC by pulse count to maintain alignment with Bayesian recruitment curve priors.
%   5. Generates a verification figure showing raw vs. blanked & interpolated signals.
%
% If Peak-to-Peak ('mV') is requested for multi-pulse trains, issues a prominent warning
% and sticks to the standard single-pulse window size.

if not(isfield(cfg_table, 'slice'))
    cfg_table.slice = struct(...
        't_min'       , -10e-3, ...
        't_max'       , 100e-3, ...
        't_min_size'  ,   5e-3, ...
        't_max_size'  ,  85e-3, ...
        't_blank_pre' ,  0.5e-3, ...
        't_blank_post',  2.0e-3  ...
        );
end
if not(isfield(cfg_table.slice, 't_blank_pre'))
    cfg_table.slice.t_blank_pre = 0.5e-3;
end
if not(isfield(cfg_table.slice, 't_blank_post'))
    cfg_table.slice.t_blank_post = 2.0e-3;
end

if not(isfield(cfg_table, 'units_mepsize')) || isempty(cfg_table.units_mepsize)
    if isfield(cfg_loaded, 'ramp') && isfield(cfg_loaded.ramp, 'type') && strcmpi(cfg_loaded.ramp.type, 'cx')
        cfg_table.units_mepsize = 'mV';
    elseif isfield(cfg_loaded, 'ramp') && isfield(cfg_loaded.ramp, 'units') && (strcmpi(cfg_loaded.ramp.units, 'P-MSO') || strcmpi(cfg_loaded.ramp.units, '%'))
        cfg_table.units_mepsize = 'mV';
    else
        % Default to AUC (uVs) for SCS
        cfg_table.units_mepsize = 'uVs';
    end
end
if not(isfield(cfg_table, 'throw_slicing_error'))
    cfg_table.throw_slicing_error = true;
end

%% Directories and output paths
d_proc = fullfile(getenv('D_PROC'));
if isempty(d_proc)
    % fall back on D_MHBMEP_PROC if there is no D_PROC
    d_proc = fullfile(getenv('D_MHBMEP_PROC'));
end
if isempty(d_proc)
    error('set env first');
end

d_out = fullfile(d_proc, cfg_loaded.filename);
p_csv_out = fullfile(d_out, sprintf('%s_mepsize.csv', cfg_loaded.filename));
if not(exist(d_out, 'dir') == 7)
    mkdir(d_out);
end
p_toml_out = strrep(p_csv_out, '.csv', '.toml');
p_mat_out = strrep(p_csv_out, '.csv', '.mat');

%% Prepare data and trigger indices
mepsize_table = trigger_table;
fs = double(cfg_loaded.daq.fs);
cfg_a = cfg_table.slice;

ix_stim = find(raw_table.(cfg_loaded.ramp.trigger));

if isempty(ix_stim) || height(mepsize_table) == 0
    error('Cannot write ramp tables: No stimulation triggers or trials found in the dataset.');
end

if size(mepsize_table, 1) == length(ix_stim)
    % ok
else
    ix_stim = ix_stim(1:height(mepsize_table));
    if cfg_table.throw_slicing_error
        error('Number of intensities and triggers are different!');
    end
end

vec_channel_name = string(raw_table.Properties.VariableNames);
n_trials = height(mepsize_table);

%% Time zero augment (baseline augmentation)
if isfield(cfg_table, 'time_zero_augment')
    time_zero_augment = cfg_table.time_zero_augment;
else
    time_zero_augment = [];
end
if not(isempty(time_zero_augment))
    assert(all(isfinite(time_zero_augment)), "?");
    mepsize_table.augmented(:) = false;

    case_augment = mepsize_table.cx_amplitude > 0 | mepsize_table.es_amplitude > 0;
    mepsize_table_augment_base = mepsize_table(case_augment, :);

    mepsize_table_augment_base.augmented(:) = true;
    ix_stim_augment_base = ix_stim(case_augment);
    mepsize_table_augment_base.cx_amplitude(:) = 0;
    mepsize_table_augment_base.es_amplitude(:) = 0;
    mepsize_table_augment_base.intensity(:) = 0;

    for ix_augment = 1:length(time_zero_augment)
        t_zero = time_zero_augment(ix_augment);
        mepsize_table_augment = mepsize_table_augment_base;
        mepsize_table_augment.datetime = mepsize_table_augment_base.datetime - t_zero*seconds;
        ix_stim_augment = ix_stim_augment_base - t_zero * fs;

        mepsize_table = [mepsize_table; mepsize_table_augment];
        ix_stim = [ix_stim; ix_stim_augment];
    end
    [~, vec_sort] = sort(mepsize_table.datetime);
    mepsize_table = mepsize_table(vec_sort, :);
    ix_stim = ix_stim(vec_sort);
    n_trials = height(mepsize_table);
end

%% Multi-pulse train detection
trial_pulse_indices = cell(n_trials, 1);
if isfield(cfg_loaded, 'trial_pulse_indices') && ~isempty(cfg_loaded.trial_pulse_indices)
    trial_pulse_indices = cfg_loaded.trial_pulse_indices;
    if numel(trial_pulse_indices) > n_trials
        trial_pulse_indices = trial_pulse_indices(1:n_trials);
    end
elseif ismember('pulse_count', mepsize_table.Properties.VariableNames) || ...
       ismember('sc_count', mepsize_table.Properties.VariableNames) || ...
       ismember('count', mepsize_table.Properties.VariableNames)
    if ismember('pulse_count', mepsize_table.Properties.VariableNames)
        counts_v = double(mepsize_table.pulse_count);
    elseif ismember('sc_count', mepsize_table.Properties.VariableNames)
        counts_v = double(mepsize_table.sc_count);
    else
        counts_v = double(mepsize_table.count);
    end

    if ismember('stim_pulse_trigger', raw_table.Properties.VariableNames)
        all_p_trigs = find(raw_table.stim_pulse_trigger);
    elseif isfield(cfg_loaded, 'ramp') && isfield(cfg_loaded.ramp, 'stim_hw') && ismember(cfg_loaded.ramp.stim_hw, raw_table.Properties.VariableNames)
        tr_sig = raw_table.(cfg_loaded.ramp.stim_hw) > 0.5;
        all_p_trigs = find([0; diff(tr_sig)] > 0);
    else
        all_p_trigs = ix_stim;
    end

    if ~isempty(counts_v) && sum(counts_v) == numel(all_p_trigs)
        p_offs = [1; 1 + cumsum(counts_v(1:end-1))];
        for j_tr = 1:n_trials
            trial_pulse_indices{j_tr} = all_p_trigs(p_offs(j_tr) : p_offs(j_tr) + counts_v(j_tr) - 1);
        end
    else
        for j_tr = 1:n_trials
            s_i = ix_stim(j_tr);
            if j_tr < n_trials, e_i = ix_stim(j_tr + 1) - 1; else, e_i = height(raw_table); end
            m_p = all_p_trigs(all_p_trigs >= s_i & all_p_trigs <= e_i);
            if numel(counts_v) >= j_tr && counts_v(j_tr) > 0 && numel(m_p) >= counts_v(j_tr)
                m_p = m_p(1:counts_v(j_tr));
            end
            trial_pulse_indices{j_tr} = m_p;
        end
    end
else
    for j_tr = 1:n_trials
        trial_pulse_indices{j_tr} = ix_stim(j_tr);
    end
end

trial_n_pulses = cellfun(@numel, trial_pulse_indices);
max_pulses = max(trial_n_pulses);
is_train = max_pulses > 1;
is_auc = ismember(lower(cfg_table.units_mepsize), {'µvs', 'uvs'});

%% Handle Peak-to-Peak (pk-pk) with Multi-Pulse Trains
if is_train && ~is_auc
    fprintf('\n================================================================================\n');
    fprintf('  WARNING: Multi-pulse stimulus train detected (%d pulses), but MEP size metric\n', max_pulses);
    fprintf('           is configured as Peak-to-Peak (''%s'').\n', cfg_table.units_mepsize);
    fprintf('           Peak-to-Peak across a multi-pulse train is ill-defined.\n');
    fprintf('           STICKING WITH STANDARD SINGLE-PULSE WINDOW: [%.1f, %.1f] ms!\n', ...
        cfg_table.slice.t_min_size * 1e3, cfg_table.slice.t_max_size * 1e3);
    fprintf('           For multi-pulse recruitment curves, please use AUC (units_mepsize = ''uVs'').\n');
    fprintf('================================================================================\n\n');
end

%% Timing and Window Configuration
if is_train && is_auc
    cfg_table.units_mepsize = 'uVs';

    trial_train_dur_s = zeros(n_trials, 1);
    trial_auc_end_rel = zeros(n_trials, 1);
    for j_tr = 1:n_trials
        p_idxs = trial_pulse_indices{j_tr};
        if numel(p_idxs) > 1
            trial_train_dur_s(j_tr) = (p_idxs(end) - p_idxs(1)) / fs;
        else
            trial_train_dur_s(j_tr) = 0;
        end
        trial_auc_end_rel(j_tr) = trial_train_dur_s(j_tr) + cfg_table.slice.t_max_size;
    end
    max_auc_end_rel = max(trial_auc_end_rel);
    t_max_vis = max(cfg_table.slice.t_max, max_auc_end_rel + 20e-3);

    fprintf('\n================== Multi-Pulse Train AUC Processing ==================\n');
    fprintf('  Detected stimulus pulse train with up to %d pulses per trial.\n', max_pulses);
    fprintf('  Max train duration        : %.3f s\n', max(trial_train_dur_s));
    fprintf('  Stim artifact blanking    : [%.2f, +%.2f] ms relative to each pulse trigger\n', ...
        -cfg_table.slice.t_blank_pre * 1e3, cfg_table.slice.t_blank_post * 1e3);
    fprintf('  Artifact interpolation    : Linear interpolation across blanked intervals\n');
    fprintf('  Modified AUC window       : [t_first + %.1f ms, t_last + %.1f ms]\n', ...
        cfg_table.slice.t_min_size * 1e3, cfg_table.slice.t_max_size * 1e3);
    fprintf('  Max AUC window duration   : %.3f s (standard single-pulse is %.1f ms)\n', ...
        max_auc_end_rel - cfg_table.slice.t_min_size, (cfg_table.slice.t_max_size - cfg_table.slice.t_min_size) * 1e3);
    fprintf('  AUC Normalization         : Normalized by pulse count (AUC / N_pulses)\n');
    fprintf('                              to align with single-pulse recruitment curve priors.\n');
    fprintf('======================================================================\n\n');
else
    t_max_vis = cfg_table.slice.t_max;
    max_auc_end_rel = cfg_table.slice.t_max_size;
end

%% Slice signals and perform blanking & interpolation
ix_slice = floor(fs * cfg_table.slice.t_min) : ceil(fs * t_max_vis);
t_sliced = ix_slice / fs;
n_samples_vis = numel(ix_slice);

ep_sliced = nan(n_trials, n_samples_vis, length(vec_channel_name));
ep_sliced_raw = nan(n_trials, n_samples_vis, length(vec_channel_name));

is_emg = ~ismember(vec_channel_name, ["Time", "cx", "di", "stim_trigger", "stim_pulse_trigger", "chunkstart", "chunksignature"]);

t_blank_pre = cfg_table.slice.t_blank_pre;
t_blank_post = cfg_table.slice.t_blank_post;

% Cap post-blanking if inter-pulse interval is very short
if is_train && is_auc
    multi_p = trial_pulse_indices(trial_n_pulses > 1);
    if ~isempty(multi_p)
        dt_list = [];
        for m_i = 1:numel(multi_p)
            dt_list = [dt_list; diff(multi_p{m_i}(:))];
        end
        avg_dt = mean(dt_list) / fs;
        if isfinite(avg_dt) && avg_dt > 0 && t_blank_post >= 0.45 * avg_dt
            t_blank_post = 0.40 * avg_dt;
        end
    end
end

if is_train && is_auc
    fprintf('Calculating MEP size as Train-Normalized AUC [units: %s]... ', cfg_table.units_mepsize);
elseif is_auc
    fprintf('Calculating MEP size as AUC [units: %s]... ', cfg_table.units_mepsize);
else
    fprintf('Calculating MEP size as Peak-to-Peak (pk-pk) [units: %s]... ', cfg_table.units_mepsize);
end

for ix = 1:length(vec_channel_name)
    str_ch = vec_channel_name(ix);
    if str_ch == "Time"
        continue;
    end
    x_col = raw_table.(str_ch);

    for j_tr = 1:n_trials
        s_first = ix_stim(j_tr);
        mat_idx = s_first + ix_slice;
        case_oob = mat_idx < 1 | mat_idx > height(raw_table);
        mat_idx(case_oob) = 1;
        y_raw = x_col(mat_idx);
        y_raw(case_oob) = 0;

        ep_sliced_raw(j_tr, :, ix) = y_raw;

        if is_train && is_auc && is_emg(ix)
            p_idxs = trial_pulse_indices{j_tr};
            p_rel = p_idxs - s_first - ix_slice(1) + 1;

            [y_clean, ~] = local_blank_and_interpolate(y_raw, p_rel, fs, t_blank_pre, t_blank_post);
            ep_sliced(j_tr, :, ix) = y_clean;

            % Calculate AUC from t_first + t_min_size to t_last + t_max_size
            i_auc_start = round((cfg_table.slice.t_min_size - cfg_table.slice.t_min) * fs) + 1;
            t_last_rel = (p_idxs(end) - p_idxs(1)) / fs;
            i_auc_end = round((t_last_rel + cfg_table.slice.t_max_size - cfg_table.slice.t_min) * fs) + 1;

            i_auc_start = max(1, min(n_samples_vis, i_auc_start));
            i_auc_end = max(1, min(n_samples_vis, i_auc_end));

            if i_auc_end > i_auc_start
                y_auc_seg = y_clean(i_auc_start:i_auc_end);
                t_auc_seg = (0:(numel(y_auc_seg)-1)) / fs;
                auc_raw = trapz(t_auc_seg, abs(y_auc_seg)) * 1e3; % in uVs
            else
                auc_raw = 0;
            end

            n_p = max(1, trial_n_pulses(j_tr));
            mepsize_table.(str_ch)(j_tr) = auc_raw / n_p;
        else
            ep_sliced(j_tr, :, ix) = y_raw;
        end
    end
end

% Standard single-pulse (or pk-pk) size calculation
if ~(is_train && is_auc)
    ix_slice_size = floor(fs * cfg_table.slice.t_min_size) : ceil(fs * cfg_table.slice.t_max_size);
    [ep_sliced_size, t_sliced_size] = slice_mat(raw_table, ix_stim, ix_slice_size, fs, vec_channel_name);
    for ix = 1:length(vec_channel_name)
        str_ch = vec_channel_name(ix);
        if is_auc
            cfg_table.units_mepsize = 'uVs';
            y_val = trapz(t_sliced_size, abs(ep_sliced_size), 2) * 1e3;
        elseif strcmpi(cfg_table.units_mepsize, 'mV')
            y_val = max(ep_sliced_size, [], 2) - min(ep_sliced_size, [], 2);
        else
            error('Bad units_mepsize specified.');
        end
        mepsize_table.(str_ch) = y_val(:, :, ix);
    end
end
fprintf('done.\n');

%% Figure visibility
show_figures = false;
if isfield(cfg_table, 'show_figures')
    show_figures = cfg_table.show_figures;
end

fig_vis = 'off';
if (islogical(show_figures) && show_figures) || strcmpi(string(show_figures), 'on')
    fig_vis = 'on';
end

%% Blanking Verification Figure (for multi-pulse trains)
if is_train && is_auc
    local_plot_blanking_verification(ep_sliced_raw, ep_sliced, t_sliced, ...
        trial_pulse_indices, ix_stim, fs, cfg_table, cfg_loaded, mepsize_table, ...
        vec_channel_name, is_emg, trial_n_pulses, max_pulses, d_out, fig_vis);
end

%% Per-channel MEP waveform figures
for ix_response = 1:length(vec_channel_name)
    str_channel_plot = vec_channel_name(ix_response);

    h_f = figure('Name', sprintf('mep_%s_%s', cfg_loaded.filename, str_channel_plot), 'Visible', fig_vis);

    width_ = 15; % Width in cm
    height_ = 25; % Height in cm
    set(h_f, 'Units', 'centimeters', 'Position', [5, 5, width_, height_]);

    case_channel = vec_channel_name == str_channel_plot;

    hold on;
    o = mepsize_table.intensity;
    if ~isempty(o) && ~isempty(ep_sliced) && size(ep_sliced, 1) > 0
        plot(t_sliced, o + ep_sliced(:, :, case_channel));
        plot(ones(1, 2) * cfg_table.slice.t_min_size, get(gca, 'ylim'), 'r', 'LineWidth', 1.2);
        plot(ones(1, 2) * max_auc_end_rel, get(gca, 'ylim'), 'r', 'LineWidth', 1.2);
        xlim([cfg_table.slice.t_min, t_max_vis]);
        grid on;
        xlabel('Time (s)');
        ylabel(sprintf('Intensity (%s) | MEP (mV)', cfg_loaded.ramp.units));
        if is_train && is_auc
            title(sprintf('%s (Artifact-Interpolated Train)', str_channel_plot));
        else
            title(str_channel_plot);
        end
    end

    saveas(h_f, fullfile(d_out, [h_f.Name, '.fig']));
    print(h_f, fullfile(d_out, [h_f.Name, '.pdf']), '-dpdf', '-bestfit');
    if ~strcmp(fig_vis, 'on')
        close(h_f);
    end
end

%% MEP Size / Recruitment curve summary figure
h_f = figure('Name', sprintf('mep_%s_size', cfg_loaded.filename), 'Visible', fig_vis);
width_ = 20; % Width in cm
height_ = 20; % Height in cm
set(h_f, 'Units', 'centimeters', 'Position', [5, 5, width_, height_]);

emg_indices = find(is_emg);
if isempty(emg_indices)
    emg_indices = 1:length(vec_channel_name);
end

for i_sub = 1:length(emg_indices)
    ix_response = emg_indices(i_sub);
    str_channel_plot = vec_channel_name(ix_response);
    nexttile; hold on;
    plot(mepsize_table.intensity, mepsize_table.(str_channel_plot), 'o', 'LineWidth', 1.2);
    title(str_channel_plot);
    xlabel(sprintf('Intensity (%s)', cfg_loaded.ramp.units));
    if is_train && is_auc
        ylabel(sprintf('MEP size (%s / pulse)', cfg_table.units_mepsize));
    else
        ylabel(sprintf('MEP size (%s)', cfg_table.units_mepsize));
    end
    grid on;
end

print(h_f, fullfile(d_out, [h_f.Name, '.pdf']), '-dpdf', '-bestfit');
saveas(h_f, fullfile(d_out, [h_f.Name, '.fig']));
if ~strcmp(fig_vis, 'on')
    close(h_f);
end

fprintf('MEP figures (.fig and .pdf) saved to:\n  %s\n', d_out);

%% Export TOML configuration, CSV table, and MAT slices
cfg_loaded.ramp.analysis = cfg_a;

cfg_to_write = cfg_loaded;
if isfield(cfg_to_write, 'trial_pulse_indices')
    cfg_to_write = rmfield(cfg_to_write, 'trial_pulse_indices');
end

try
    toml.write(p_toml_out, toml.struct_to_map(cfg_to_write));
catch
    fprintf('Failed to write toml: %s\n', p_toml_out);
end
writetable(mepsize_table, p_csv_out);
save(p_mat_out, 'ep_sliced', 't_sliced');
fprintf('Recruitment table written to:\n  %s\n', p_csv_out);

end

%% Helper: Blank out stimulation artifacts and linearly interpolate
function [y_interp, is_blanked] = local_blank_and_interpolate(y_raw, pulse_rel_indices, fs, t_blank_pre, t_blank_post)
n = numel(y_raw);
is_blanked = false(size(y_raw));
n_pre = round(t_blank_pre * fs);
n_post = round(t_blank_post * fs);

for ip = 1:numel(pulse_rel_indices)
    k = pulse_rel_indices(ip);
    if k >= 1 && k <= n
        i1 = max(1, k - n_pre);
        i2 = min(n, k + n_post);
        is_blanked(i1:i2) = true;
    end
end

y_nan = y_raw;
y_nan(is_blanked) = NaN;
y_interp = fillmissing(y_nan, 'linear', 'EndValues', 'nearest');
y_interp(isnan(y_interp)) = 0;
end

%% Helper: Generate comprehensive Blanking & Interpolation verification figure
function local_plot_blanking_verification(ep_sliced_raw, ep_sliced, t_sliced, ...
    trial_pulse_indices, ix_stim, fs, cfg_table, cfg_loaded, mepsize_table, ...
    vec_channel_name, is_emg, trial_n_pulses, max_pulses, d_out, fig_vis)

% Pick representative trial: trial with max pulses and max intensity
cand_rep = find(trial_n_pulses == max_pulses);
if isempty(cand_rep), cand_rep = 1:height(mepsize_table); end
[~, ix_sub] = max(mepsize_table.intensity(cand_rep));
j_rep = cand_rep(ix_sub);

% Primary EMG channel to display
emg_cand = find(is_emg);
if isempty(emg_cand), emg_cand = 2; end
ix_rep_ch = emg_cand(1);
str_rep_ch = vec_channel_name(ix_rep_ch);

y_raw_rep = squeeze(ep_sliced_raw(j_rep, :, ix_rep_ch));
y_clean_rep = squeeze(ep_sliced(j_rep, :, ix_rep_ch));
p_idxs_rep = trial_pulse_indices{j_rep};
p_rel_s = (p_idxs_rep - ix_stim(j_rep)) / fs;

h_v = figure('Name', sprintf('mep_%s_blanking_verification', cfg_loaded.filename), 'Visible', fig_vis);
set(h_v, 'Units', 'centimeters', 'Position', [3, 3, 24, 18]);

%% Subplot 1: Full Train Overview (Raw vs. Interpolated)
subplot(2, 2, [1, 2]); hold on;
plot(t_sliced, y_raw_rep, 'Color', [0.85 0.35 0.35 0.65], 'LineWidth', 0.9, 'DisplayName', 'Raw EMG (with artifacts)');
plot(t_sliced, y_clean_rep, 'Color', [0.05 0.35 0.75], 'LineWidth', 1.3, 'DisplayName', 'Blanked & Interpolated');

% Pulse triggers
if ~isempty(p_rel_s)
    y_lims = get(gca, 'ylim');
    for ip = 1:numel(p_rel_s)
        if ip == 1
            plot([p_rel_s(ip) p_rel_s(ip)], y_lims, ':', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.8, 'DisplayName', 'Stim Triggers');
        else
            plot([p_rel_s(ip) p_rel_s(ip)], y_lims, ':', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.8, 'HandleVisibility', 'off');
        end
    end
end

% AUC Window limits
t_auc_start = cfg_table.slice.t_min_size;
t_auc_end = p_rel_s(end) + cfg_table.slice.t_max_size;
xline(t_auc_start, '--g', 'LineWidth', 1.5, 'DisplayName', 'AUC Window Start (t_{first} + 5 ms)');
xline(t_auc_end, '--m', 'LineWidth', 1.5, 'DisplayName', 'AUC Window End (t_{last} + 85 ms)');

grid on;
xlabel('Time from 1st pulse (s)');
ylabel('EMG (mV)');
title(sprintf('Stimulus Artifact Blanking: Full Train Overview [%s] (Trial %d, %d pulses, Intensity %.1f %s)', ...
    str_rep_ch, j_rep, numel(p_idxs_rep), mepsize_table.intensity(j_rep), cfg_loaded.ramp.units));
legend('Location', 'northeast', 'FontSize', 8);

%% Subplot 2: Zoomed-in Detail (Artifact blanking & interpolation across first pulses)
subplot(2, 2, 3); hold on;
n_zoom_pulses = min(3, numel(p_rel_s));
if n_zoom_pulses > 1
    t_zoom_end = p_rel_s(n_zoom_pulses) + 12e-3;
else
    t_zoom_end = 25e-3;
end
t_zoom_start = -2e-3;
zoom_mask = (t_sliced >= t_zoom_start) & (t_sliced <= t_zoom_end);

% Shade blanking regions
y_min_val = min(y_raw_rep(zoom_mask));
y_max_val = max(y_raw_rep(zoom_mask));
if isempty(y_min_val) || ~isfinite(y_min_val), y_min_val = -1; end
if isempty(y_max_val) || ~isfinite(y_max_val), y_max_val = 1; end

for ip = 1:n_zoom_pulses
    b_start = p_rel_s(ip) - cfg_table.slice.t_blank_pre;
    b_end = p_rel_s(ip) + cfg_table.slice.t_blank_post;
    patch([b_start b_end b_end b_start], [y_min_val y_min_val y_max_val y_max_val], ...
        [1.0 0.88 0.88], 'EdgeColor', 'none', 'FaceAlpha', 0.6, ...
        'DisplayName', 'Blanked Window');
end

plot(t_sliced(zoom_mask), y_raw_rep(zoom_mask), 'Color', [0.85 0.25 0.25], 'LineWidth', 1.1, 'DisplayName', 'Raw Artifact');
plot(t_sliced(zoom_mask), y_clean_rep(zoom_mask), 'Color', [0.05 0.35 0.75], 'LineWidth', 1.6, 'DisplayName', 'Linear Interpolation');

for ip = 1:n_zoom_pulses
    xline(p_rel_s(ip), ':', 'Color', [0.3 0.3 0.3], 'LineWidth', 1.0, 'HandleVisibility', 'off');
end

xlim([t_zoom_start, t_zoom_end]);
grid on;
xlabel('Time (s)');
ylabel('EMG (mV)');
title(sprintf('Zoomed Artifact Detail (Blanking: [%.1f, +%.1f] ms)', ...
    -cfg_table.slice.t_blank_pre * 1e3, cfg_table.slice.t_blank_post * 1e3));
legend('Location', 'northeast', 'FontSize', 8);

%% Subplot 3: Rectified Train EMG & Area Under Curve
subplot(2, 2, 4); hold on;
auc_mask = (t_sliced >= t_auc_start) & (t_sliced <= t_auc_end);
t_auc_vec = t_sliced(auc_mask);
y_rect = abs(y_clean_rep(auc_mask));

area(t_auc_vec, y_rect, 'FaceColor', [0.3 0.6 0.9], 'FaceAlpha', 0.4, 'EdgeColor', [0.1 0.3 0.7], 'LineWidth', 1.0);
xline(t_auc_start, '--g', 'LineWidth', 1.2);
xline(t_auc_end, '--m', 'LineWidth', 1.2);

auc_val_trial = mepsize_table.(str_rep_ch)(j_rep);
n_p_rep = max(1, numel(p_idxs_rep));
grid on;
xlabel('Time from 1st pulse (s)');
ylabel('|EMG| (mV)');
title(sprintf('Rectified EMG (Normalized AUC: %.2f \\muVs [%d pulses])', auc_val_trial, n_p_rep));

%% Save verification figure
p_verify_fig = fullfile(d_out, sprintf('mep_%s_blanking_verification.fig', cfg_loaded.filename));
p_verify_pdf = fullfile(d_out, sprintf('mep_%s_blanking_verification.pdf', cfg_loaded.filename));
saveas(h_v, p_verify_fig);
print(h_v, p_verify_pdf, '-dpdf', '-bestfit');
if ~strcmp(fig_vis, 'on')
    close(h_v);
end

fprintf('Artifact blanking verification figure saved to:\n  %s\n', p_verify_pdf);
end
