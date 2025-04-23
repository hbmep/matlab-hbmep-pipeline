function analyse_ramp(h_loader, p_data, response, varargin)
%%
set_env;  % because of this, call to analyse_ramp has to be in this directory

%%
if contains(p_data, 'example')
    p_data = fullfile(getenv('D_GIT'), 'auxf', p_data, 'SUBID_V1_IMM_data.csv');  % reduce this and make it not dnc_
end

%%
loader_custom = eval(sprintf('@(p, v) %s(p, v)', h_loader));
[raw_table, trigger_table, cfg_aq] = loader_custom(p_data, varargin);

%%
cfg_proc = struct;
if not(isfield(cfg_proc, 'mep_size'))
    cfg_proc.slice = struct(...
        't_min'      , -10e-3, ...
        't_max'      , 100e-3, ...
        't_min_size' ,   5e-3, ...
        't_max_size' ,  85e-3  ...
        );
end
cfg_proc.response = response;

%%
fprintf('Writing tables... ');
p_csv = sp.write_ramp_tables(raw_table, trigger_table, cfg_aq, cfg_proc);
fprintf('done.\n');

%%

sp.call_hbmep(p_csv, string(cfg_proc.response));

end