% Get a large example dataset and downsample it so that it can be included
% as part of this repository.

%%
clear;
addpath(fullfile('..', '..'))
set_env;

%%
f_data = 'SUBID_V1_IMM_data.csv';
p_data = fullfile('..', 'dnc_example-data', f_data);
p_config = replace(p_data, '_data.csv', '_meta.toml');
p_trigger = replace(p_data, '_data.csv', '_trigger.csv');

%%
data = table2array(readtable(p_data));
trigger = table2array(readtable(p_trigger));
cfg_aq = toml.map_to_struct(toml.read(p_config));

%%
fs_new = 500;
fs = cfg_aq.daq.fs;
cfg_out = cfg_aq;
cfg_out.daq.fs = fs_new;
assert(mod(fs, fs_new) == 0, '?');

%%
ix_trigger = find(strcmpi(cfg_aq.data.column_name, cfg_aq.ramp.trigger));

%%
% simplify the trigger channel since it might not resample well
trig = data(:, ix_trigger) > 2.5;
trigb = [0; diff(trig)] > 0;
trigf = filter(ones(1, (fs/fs_new) * 3), 1, trigb);

% resample/downsample
data_ds = resample(data, fs_new, fs);
y_trig = downsample(trigf, fs/fs_new);
y_trig = [0; diff(y_trig)] > 0;

% re-insert
data_ds(:, ix_trigger) = y_trig;

%% reduce the number of triggers
n_trig = ceil(sum(data_ds(:, ix_trigger)) * 0.4);
ix_y_trig = find(data_ds(:, ix_trigger));

ix_keep = ix_y_trig(n_trig + 1) - 1;  % keep up to one sample prior to the next trigger

data_ds = data_ds(1:ix_keep, :);
trigger = trigger(1:n_trig, :);

%%
% just make the trigger channel (a bit) more realistic again
data_ds(:, ix_trigger) = filter(ones(1, 3), 1, data_ds(:, ix_trigger));
data_ds(:, ix_trigger) = data_ds(:, ix_trigger) * 5.0;

%% save
% data
write_csv(f_data, data_ds);

% trigger table
[~, f_trigger_out, ext_trigger_out] = fileparts(p_trigger);
write_csv(sprintf('%s%s', f_trigger_out, ext_trigger_out), trigger);

% toml
[~, f_cfg_out] = fileparts(p_config);
toml.write(sprintf('%s.toml', f_cfg_out), cfg_out);

%%
function write_csv(filename, data)
fid = fopen(filename, 'w');

fmt = [repmat('%.4f,', 1, size(data, 2)-1), '%.4f\n'];  % One row format string
fprintf(fid, fmt, data.');  % Transpose is needed because fprintf writes column-wise

fclose(fid);
end