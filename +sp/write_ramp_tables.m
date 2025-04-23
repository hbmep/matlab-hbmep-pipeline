function p_csv_out = write_ramp_tables(raw_table, trigger_table, cfg_aq, cfg_proc)
% d = struct(...
%     't_min'      , -10e-3, ...
%     't_max'      , 100e-3, ...
%     't_min_size' ,   5e-3, ...
%     't_max_size' ,  85e-3  ...
%     );
% v = parse_inputs(d, varargin{:});

%%
d_proc = fullfile(getenv('D_PROC'));
d_out = fullfile(d_proc, cfg_aq.filename);
p_csv_out = fullfile(d_out, sprintf('%s_mepsize.csv', cfg_aq.filename));
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
fs = double(cfg_aq.daq.fs);

%%
cfg_a = cfg_proc.slice;

%%
ix_stim = find(raw_table.(cfg_aq.ramp.trigger));
vec_channel_name = string(raw_table.Properties.VariableNames);

%% Slice MEPs for visualisation
ix_slice = floor(fs * cfg_proc.slice.t_min): ceil(cfg_proc.slice.t_max *fs);
[ep_sliced, t_sliced] = slice_mat(raw_table, ix_stim, ix_slice, fs, vec_channel_name);

%% Slice MEPs for AUC
ix_slice = floor(fs * cfg_proc.slice.t_min_size): ceil(cfg_proc.slice.t_max_size*fs);
[ep_sliced_size, t_sliced_size] = slice_mat(raw_table, ix_stim, ix_slice, fs, vec_channel_name);

%% AUC calculation
for ix = 1:length(vec_channel_name)
    str_ch = vec_channel_name(ix);
    y_auc = trapz(t_sliced_size, abs(ep_sliced_size), 2) * 1e6;  % assuming raw units are s, V, so going to uVs
    mepsize_table.(str_ch) = y_auc(:, :, ix);

end

%%
for ix_response = 1:length(vec_channel_name)
    str_channel_plot = vec_channel_name(ix_response);

    h_f = figure('Name', sprintf('%s', cfg_aq.filename));
        h_f = figure('Name', sprintf('mep_%s_%s', cfg_aq.filename, str_channel_plot));

    width = 15; % Width in cm
    height = 25; % Height in cm
    set(h_f, 'Units', 'centimeters', 'Position', [5, 5, width, height]);

    case_channel = vec_channel_name == str_channel_plot;

    hold on;
    plot(t_sliced, mepsize_table.intensity + ep_sliced(:, :, case_channel) * 1e3);
    plot(ones(1, 2) * cfg_proc.slice.t_min_size, get(gca, 'ylim'), 'r');
    plot(ones(1, 2) * cfg_proc.slice.t_max_size, get(gca, 'ylim'), 'r');
    xlim([cfg_proc.slice.t_min, cfg_proc.slice.t_max]);
    grid on;
    xlabel('Time (s)');
    ylabel('Intensity (mA/%) | MEP size (mV)');
    title(str_channel_plot);

    saveas(h_f, fullfile(d_out, [h_f.Name, '.fig']));
    print(h_f, fullfile(d_out, [h_f.Name, '.pdf']), '-dpdf', '-bestfit');

end

%%
clf;
h_f = figure('Name', sprintf('mep_%s_size', cfg_aq.filename));
width = 20; % Width in cm
height = 20; % Height in cm
set(h_f, 'Units', 'centimeters', 'Position', [5, 5, width, height]);

for ix_response = 1:length(vec_channel_name)
    str_channel_plot = vec_channel_name(ix_response);
    nexttile; hold on;
    plot(mepsize_table.intensity, mepsize_table.(str_channel_plot), 'o');
    title(str_channel_plot);
    xlabel('Intensity (mA/%)');
    ylabel('MEP size (\muVs)');
    grid on;

end

print(h_f, fullfile(d_out, [h_f.Name, '.pdf']), '-dpdf', '-bestfit');

%%
cfg_aq.ramp.analysis = cfg_a;

%%
toml.write(p_toml_out, cfg_aq);
writetable(mepsize_table, p_csv_out);
save(p_mat_out, 'ep_sliced', 't_sliced')
fprintf('Recruitment table written to:\n%s\n', p_csv_out);

end

