function [p_csv_out, cfg_table] = write_ramp_tables(raw_table, trigger_table, cfg_loaded, cfg_table)
if not(isfield(cfg_table, 'slice'))
    cfg_table.slice = struct(...
        't_min'      , -10e-3, ...
        't_max'      , 100e-3, ...
        't_min_size' ,   5e-3, ...
        't_max_size' ,  85e-3  ...
        );
end
if not(isfield(cfg_table, 'units_mepsize'))
    cfg_table.units_mepsize = 'µVs';
end

%%
d_proc = fullfile(getenv('D_PROC'));
if isempty(d_proc)
    % fall back on D_MHBMEP_PROC if there is no D_PROC
    d_proc = fullfile(getenv('D_MHBMEP_PROC'));
end
d_out = fullfile(d_proc, cfg_loaded.filename);
p_csv_out = fullfile(d_out, sprintf('%s_mepsize.csv', cfg_loaded.filename));
if not(exist(d_out, 'dir') == 7)
    mkdir(d_out);
end
p_toml_out = strrep(p_csv_out, '.csv', '.toml');
p_mat_out = strrep(p_csv_out, '.csv', '.mat');

%%
if isempty(d_proc)
    error('set env first');
end

%%
mepsize_table = trigger_table;
%%
fs = double(cfg_loaded.daq.fs);

%%
cfg_a = cfg_table.slice;

%%
ix_stim = find(raw_table.(cfg_loaded.ramp.trigger));
vec_channel_name = string(raw_table.Properties.VariableNames);

%% Slice MEPs for visualisation
ix_slice = floor(fs * cfg_table.slice.t_min): ceil(cfg_table.slice.t_max *fs);
[ep_sliced, t_sliced] = slice_mat(raw_table, ix_stim, ix_slice, fs, vec_channel_name);

%% Slice MEPs for AUC
ix_slice = floor(fs * cfg_table.slice.t_min_size): ceil(cfg_table.slice.t_max_size*fs);
[ep_sliced_size, t_sliced_size] = slice_mat(raw_table, ix_stim, ix_slice, fs, vec_channel_name);

%% AUC calculation
for ix = 1:length(vec_channel_name)
    str_ch = vec_channel_name(ix);
    if strcmpi(cfg_table.units_mepsize, 'µVs')
        y_auc = trapz(t_sliced_size, abs(ep_sliced_size), 2) * 1e3;  % units are mV so now we get uVs
    elseif strcmpi(cfg_table.units_mepsize, 'mV')
        y_auc = max(ep_sliced_size, [], 2) - min(ep_sliced_size, [], 2); % keep in mV
    else
        error('Bad units_mepsize specified.')
    end
    mepsize_table.(str_ch) = y_auc(:, :, ix);

end

%%
for ix_response = 1:length(vec_channel_name)
    str_channel_plot = vec_channel_name(ix_response);

    h_f = figure('Name', sprintf('mep_%s_%s', cfg_loaded.filename, str_channel_plot));

    width = 15; % Width in cm
    height = 25; % Height in cm
    set(h_f, 'Units', 'centimeters', 'Position', [5, 5, width, height]);

    case_channel = vec_channel_name == str_channel_plot;

    hold on;
    o = mepsize_table.intensity;
    plot(t_sliced, o + ep_sliced(:, :, case_channel));
    plot(ones(1, 2) * cfg_table.slice.t_min_size, get(gca, 'ylim'), 'r');
    plot(ones(1, 2) * cfg_table.slice.t_max_size, get(gca, 'ylim'), 'r');
    xlim([cfg_table.slice.t_min, cfg_table.slice.t_max]);
    grid on;
    xlabel('Time (s)');
    ylabel(sprintf('Intensity (%s) | MEP (mV)', cfg_loaded.ramp.units));
    title(str_channel_plot);

    saveas(h_f, fullfile(d_out, [h_f.Name, '.fig']));
    print(h_f, fullfile(d_out, [h_f.Name, '.pdf']), '-dpdf', '-bestfit');

end

%%
clf;
h_f = figure('Name', sprintf('mep_%s_size', cfg_loaded.filename));
width = 20; % Width in cm
height = 20; % Height in cm
set(h_f, 'Units', 'centimeters', 'Position', [5, 5, width, height]);

for ix_response = 1:length(vec_channel_name)
    str_channel_plot = vec_channel_name(ix_response);
    nexttile; hold on;
    plot(mepsize_table.intensity, mepsize_table.(str_channel_plot), 'o');
    title(str_channel_plot);
    xlabel('Intensity (mA/%)');
    ylabel(sprintf('MEP size (%s)', cfg_table.units_mepsize));
    grid on;

end

print(h_f, fullfile(d_out, [h_f.Name, '.pdf']), '-dpdf', '-bestfit');

%%
cfg_loaded.ramp.analysis = cfg_a;

%%
toml.write(p_toml_out, cfg_loaded);
writetable(mepsize_table, p_csv_out);
save(p_mat_out, 'ep_sliced', 't_sliced')
fprintf('Recruitment table written to:\n%s\n', p_csv_out);

end

