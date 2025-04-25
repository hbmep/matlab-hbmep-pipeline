function analyse_ramp(h_loader, p_data, response, cfg_proc)
%%
set_env;  % because of this, call to analyse_ramp has to be in this directory

%%
if nargin < 4
    cfg_proc = struct;
end
if not(isfield(cfg_proc, 'loader'))
    cfg_proc.loader = [];
end
if not(isfield(cfg_proc, 'table'))
    cfg_proc.table = [];
end
if not(isfield(cfg_proc, 'hbmep'))
    cfg_proc.hbmep = [];
end

%%
if contains(p_data, 'example')
    p_data = fullfile(getenv('D_GIT'), 'auxf', p_data, 'SUBID_V1_IMM_data.csv');  % reduce this and make it not dnc_
end

%%
loader_custom = eval(sprintf('@(p, c) %s(p, c)', h_loader));
[raw_table, trigger_table, cfg_loaded] = loader_custom(p_data, cfg_proc.loader);

%%
fprintf('Writing tables... ');
p_csv = sp.write_ramp_tables(raw_table, trigger_table, cfg_loaded, cfg_proc.table);
fprintf('done.\n');

%%
cfg_proc.hbmep.units_intensity = cfg_loaded.ramp.units;
cfg_proc.hbmep.units_mepsize = cfg_proc.table.units_mepsize;
sp.call_hbmep(p_csv, response, cfg_proc.hbmep);

end