function [raw_table, trigger_table, cfg_aq] = loader_mrl(p_data, cfg_load)
% This produces:
% raw_table, trigger_table, cfg_aq
% necessary for subsequent processing
% this version is specific for the recording setup used in the movement
% recovery laboratory (Columbia U.)


%%
p_trigger = replace(p_data, '.csv', '_table.mat');
p_config = replace(p_data, '.csv', '.toml');
raw_table = readtable(p_data);
trigger_table = load(p_trigger);

%%
if isfield(trigger_table, 'T_pre')
    trigger_table = trigger_table.T_pre;
else
    trigger_table = trigger_table.T;
    trigger_table = trigger_table(not(isnat(trigger_table.datetime)), :);
end
trigger_table = trigger_table(trigger_table.stim, :);

cfg = trigger_table.Properties.UserData;

%%
cfg_sequence = cfg.(trigger_table.Properties.UserData.exp_sequence);
ramp_type = cfg_sequence.ramp_type;

n_emg = 8;
emg_channels = repmat("", 1, n_emg);
emg_channels(1:length(cfg.participant.emg_channels)) = string(cfg.participant.emg_channels);

vec_channel_name = string([{'Time'}, emg_channels, cfg.participant.return_channels]);
vec_channel_type = ["TIME", repmat("EMG", 1, n_emg), repmat("STIM", 1, length(cfg.participant.return_channels))];

% temp
case_mod = (vec_channel_name == "");
vec_channel_name(case_mod) = string(arrayfun(@(ix) sprintf('dc%d', ix), [1:sum(case_mod)], 'UniformOutput', false));

raw_table.Properties.VariableNames = string(vec_channel_name);
if ramp_type == "cx"
    stim_hw = ramp_type;
    cfg.ramp.units = 'P-MSO';
else
    stim_hw = "di";
    cfg.ramp.units = 'mA';
end

%%
cfg.ramp.trigger = sprintf('%s', stim_hw);
cfg.data.units.EMG = 'V';

%%
trigger_table.condition(:) = ramp_type;

raw_table.(cfg.ramp.trigger) = raw_table.(cfg.ramp.trigger) > 0.5;
raw_table.(cfg.ramp.trigger) = [0; diff(raw_table.(cfg.ramp.trigger))] > 0;

for ix_channel = 1:length(raw_table.Properties.VariableNames)
    str_channel = string(raw_table.Properties.VariableNames(ix_channel));
    str_type = vec_channel_type(ix_channel);
    if str_channel == cfg.ramp.trigger
        continue;
    end

    if str_channel == cfg.ramp.trigger
        continue;
    end
    if str_type == "EMG"
        raw_table.(str_channel) = raw_table.(str_channel) - median(raw_table.(str_channel));

        % deal with units
        if isfield(cfg.data.units, str_type)
            % MEP size should be in mV at this stage
            str_type_units = cfg.data.units.(str_type);
            if str_type_units == "mV"
                raw_table.(str_channel) = raw_table.(str_channel);
            elseif str_type_units == "V"
                raw_table.(str_channel) = raw_table.(str_channel) * 1e3;
            else
                error('Units not handled.')
            end
        end
    end
end

assert(size(trigger_table, 1) == sum(raw_table.(cfg.ramp.trigger)), 'Number of intensities and triggers are different!');

%% Assign the intensity
trigger_table.intensity = trigger_table.(sprintf('%s_amplitude', ramp_type));

%% Minimum config information:
cfg_aq = struct;
[~, cfg_aq.filename, ~] = fileparts(p_data);  % filename (to name directories)
cfg_aq.daq.fs = double(cfg.daq.fs);  % sampling rate
cfg_aq.ramp.trigger = cfg.ramp.trigger;  % the name of the channel where the sampling rate is
cfg_aq.ramp.units = cfg.ramp.units;  % e.g. mA or %MSO

end