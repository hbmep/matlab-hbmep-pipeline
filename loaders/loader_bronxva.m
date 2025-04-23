function [raw_table, trigger_table, cfg_aq] = loader_bronxva(p_data, varargin)
p_trigger = replace(p_data, '_data.csv', '_trigger.csv');
p_config = replace(p_data, '_data.csv', '_meta.toml');
raw_table = readtable(p_data);
trigger_table = readtable(p_trigger);
cfg_aq = toml.map_to_struct(toml.read(p_config));
[~, fp, ~] = fileparts(p_data);
cfg_aq.filename = fp;

%%
raw_table.Properties.VariableNames = string(cfg_aq.data.column_name);
trigger_table.Properties.VariableNames = string(cfg_aq.trigger.column_name);
trigger_table.participant(:) = "X";
trigger_table.condition(:) = string(cfg_aq.trigger.column_name);

raw_table.(cfg_aq.ramp.trigger) = raw_table.(cfg_aq.ramp.trigger) > 2.5;
raw_table.(cfg_aq.ramp.trigger) = [0; diff(raw_table.(cfg_aq.ramp.trigger))] > 0;

for ix_channel = 1:length(raw_table.Properties.VariableNames)
    str_channel = string(raw_table.Properties.VariableNames(ix_channel));
    str_type = string(cfg_aq.data.column_type(str_channel == string(cfg_aq.data.column_name)));
    if str_channel == cfg_aq.ramp.trigger
        continue;
    end

    if str_channel == cfg_aq.ramp.trigger
        continue;
    end
    if str_type == "emg"
        % deal with offset, despite hardware filter...
        raw_table.(str_channel) = raw_table.(str_channel) - median(raw_table.(str_channel));

        % deal with units
        if isfield(cfg_aq.data.units, str_type)
            str_type_units = cfg_aq.data.units.(str_type);
            if str_type_units == "mV"  % convert to uV
                raw_table.(str_channel) = raw_table.(str_channel) * 1e-3;
            else
                error('Units not handled.')
            end
        end
    end
end

assert(size(trigger_table, 1) == sum(raw_table.(cfg_aq.ramp.trigger)), 'Number of intensities and triggers are different!');
end