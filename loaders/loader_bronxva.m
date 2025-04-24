function [raw_table, trigger_table, cfg_aq] = loader_bronxva(p_data, varargin)
% This produces:
% raw_table, trigger_table, cfg_aq
% necessary for subsequent processing
% this version is specific for the recording setup used in Noam Harel's lab
% at the Bronx VA


%%
p_trigger = replace(p_data, '_data.csv', '_trigger.csv');
p_config = replace(p_data, '_data.csv', '_meta.toml');
raw_table = readtable(p_data);
trigger_table = readtable(p_trigger);
cfg = toml.map_to_struct(toml.read(p_config));


%%
raw_table.Properties.VariableNames = string(cfg.data.column_name);
trigger_table.Properties.VariableNames = string(cfg.trigger.column_name);
trigger_table.participant(:) = "X";
trigger_table.condition(:) = string(cfg.trigger.column_name);

raw_table.(cfg.ramp.trigger) = raw_table.(cfg.ramp.trigger) > 2.5;
raw_table.(cfg.ramp.trigger) = [0; diff(raw_table.(cfg.ramp.trigger))] > 0;

for ix_channel = 1:length(raw_table.Properties.VariableNames)
    str_channel = string(raw_table.Properties.VariableNames(ix_channel));
    str_type = string(cfg.data.column_type(str_channel == string(cfg.data.column_name)));
    if str_channel == cfg.ramp.trigger
        continue;
    end

    if str_channel == cfg.ramp.trigger
        continue;
    end
    if str_type == "emg"
        % deal with offset, despite hardware filter...
        raw_table.(str_channel) = raw_table.(str_channel) - median(raw_table.(str_channel));

        % deal with units
        if isfield(cfg.data.units, str_type)
            str_type_units = cfg.data.units.(str_type);
            if str_type_units == "mV"  % convert to uV
                raw_table.(str_channel) = raw_table.(str_channel) * 1e-3;
            else
                error('Units not handled.')
            end
        end
    end
end

assert(size(trigger_table, 1) == sum(raw_table.(cfg.ramp.trigger)), 'Number of intensities and triggers are different!');


%% Minimum config information:
cfg_aq = struct;
[~, cfg_aq.filename, ~] = fileparts(p_data);  % filename (to name directories)
cfg_aq.daq.fs = double(cfg.daq.fs);  % sampling rate
cfg_aq.ramp.trigger = cfg.ramp.trigger;  % the name of the channel where the sampling rate is


end