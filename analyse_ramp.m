function analyse_ramp(h_loader, p_data, response, cfg_proc)
%%
set_env_mhbmep;  % because of this, call to analyse_ramp has to be in this directory

%%
if nargin < 4
    cfg_proc = struct;
end
if not(isfield(cfg_proc, 'loader'))
    cfg_proc.loader = struct;
end
if isempty(cfg_proc.loader) || not(isfield(cfg_proc.loader, 'line_noise_suppression'))
    cfg_proc.loader.line_noise_suppression = 'kalman';
end
if not(isfield(cfg_proc, 'table'))
    cfg_proc.table = [];
end
if not(isfield(cfg_proc, 'hbmep'))
    cfg_proc.hbmep = [];
end

%%
if (ischar(p_data) || (isstring(p_data) && isscalar(p_data))) && contains(p_data, 'example')
    p_data = fullfile(getenv('D_MHBMEP_GIT'), 'auxf', p_data, 'SUBID_V1_IMM_data.csv');  % reduce this and make it not dnc_
end

%%
loader_custom = eval(sprintf('@(p, c) %s(p, c)', h_loader));
[raw_table, trigger_table, cfg_loaded] = loader_custom(p_data, cfg_proc.loader);

if ~isfield(cfg_loaded, 'p_data') || isempty(cfg_loaded.p_data)
    cfg_loaded.p_data = p_data;
end
if ~isfield(cfg_loaded, 'folder') || isempty(cfg_loaded.folder)
    if iscell(p_data), p_first = char(p_data{1}); elseif isstring(p_data), p_first = char(p_data(1)); else, p_first = char(p_data); end
    [p_dir_1, ~, ~] = fileparts(p_first);
    while endsWith(p_dir_1, '/') || endsWith(p_dir_1, '\')
        p_dir_1 = p_dir_1(1:end-1);
    end
    [~, folder_source, ~] = fileparts(p_dir_1);
    cfg_loaded.folder = folder_source;
end

%%
fprintf('Writing tables... ');
[p_csv, cfg_table] = sp.write_ramp_tables(raw_table, trigger_table, cfg_loaded, cfg_proc.table);
fprintf('done.\n');

%%
cfg_proc.hbmep.units_intensity = cfg_loaded.ramp.units;
cfg_proc.hbmep.units_mepsize = cfg_table.units_mepsize;
if not(isfield(cfg_proc.hbmep, 'model')) || isempty(cfg_proc.hbmep.model)
    if ismember(lower(cfg_table.units_mepsize), {'uvs', 'µvs'})
        cfg_proc.hbmep.model = 'rlaucSmall';
    else
        cfg_proc.hbmep.model = 'rlpkpkSmall';
    end
end
if not(isfield(cfg_proc.hbmep, 'stim_type')) || isempty(cfg_proc.hbmep.stim_type)
    if isfield(cfg_loaded.ramp, 'type') && strcmpi(cfg_loaded.ramp.type, 'cx')
        cfg_proc.hbmep.stim_type = 'tms';
    else
        cfg_proc.hbmep.stim_type = 'scs';
    end
end
if ismember(lower(cfg_table.units_mepsize), {'uvs', 'µvs'})
    metric_str = 'AUC (Area Under Curve)';
else
    metric_str = 'Peak-to-Peak (pk-pk)';
end

fprintf('\n================== hbMEP Pipeline Configuration ==================\n');
fprintf('  hbMEP Model      : %s\n', cfg_proc.hbmep.model);
fprintf('  Stimulation Mode : %s\n', upper(cfg_proc.hbmep.stim_type));
fprintf('  MEP Size Metric  : %s [%s]\n', metric_str, cfg_table.units_mepsize);
fprintf('  Intensity Units  : %s\n', cfg_proc.hbmep.units_intensity);
fprintf('==================================================================\n\n');

if not(isfield(cfg_proc, 'run_hbmep')) || cfg_proc.run_hbmep
    sp.call_hbmep(p_csv, response, cfg_proc.hbmep);
else
    fprintf('Skipping hbMEP MCMC model execution (run_hbmep is false).\n');
end

end