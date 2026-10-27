function call_hbmep(p_csv, response, cfg_hbmep)
if not(isfield(cfg_hbmep, 'es'))
    cfg_hbmep.es = '';
end
if not(isfield(cfg_hbmep, 'units_intensity'))
    cfg_hbmep.units_intensity = 'A. U.';
end
if not(isfield(cfg_hbmep, 'units_mepsize'))
    cfg_hbmep.units_mepsize = 'A. U.';
end
if not(isfield(cfg_hbmep, 'p_postproc'))
    cfg_hbmep.p_postproc = '';
end
if not(isfield(cfg_hbmep, 'model')) || isempty(cfg_hbmep.model)
    if isfield(cfg_hbmep, 'units_mepsize') && ismember(lower(cfg_hbmep.units_mepsize), {'uvs', 'µvs'})
        cfg_hbmep.model = 'rlaucSmall';
    else
        cfg_hbmep.model = 'rlpkpkSmall';
    end
end
if not(isfield(cfg_hbmep, 'stim_type')) || isempty(cfg_hbmep.stim_type)
    cfg_hbmep.stim_type = 'tms';
end
if not(isfield(cfg_hbmep, 'use_mixture')) || isempty(cfg_hbmep.use_mixture) || (ischar(cfg_hbmep.use_mixture) && isempty(cfg_hbmep.use_mixture))
    vec_mixture = false;
elseif islogical(cfg_hbmep.use_mixture)
    vec_mixture = cfg_hbmep.use_mixture;
elseif strcmpi(string(cfg_hbmep.use_mixture), "both")
    vec_mixture = [false, true];
elseif strcmpi(string(cfg_hbmep.use_mixture), "true")
    vec_mixture = true;
else
    vec_mixture = false;
end

if not(isfield(cfg_hbmep, 'use_ima')) || isempty(cfg_hbmep.use_ima)
    use_ima = false;
else
    use_ima = logical(cfg_hbmep.use_ima);
end
if not(isfield(cfg_hbmep, 'ima_params'))
    cfg_hbmep.ima_params = struct;
end

if not(isfield(cfg_hbmep, 'num_warmup')) || isempty(cfg_hbmep.num_warmup)
    cfg_hbmep.num_warmup = 400;
end
if not(isfield(cfg_hbmep, 'num_samples')) || isempty(cfg_hbmep.num_samples)
    cfg_hbmep.num_samples = 400;
end
if not(isfield(cfg_hbmep, 'num_chains')) || isempty(cfg_hbmep.num_chains)
    cfg_hbmep.num_chains = 4;
end
if not(isfield(cfg_hbmep, 'thinning')) || isempty(cfg_hbmep.thinning)
    cfg_hbmep.thinning = 1;
end
if not(isfield(cfg_hbmep, 'max_tree_depth')) || isempty(cfg_hbmep.max_tree_depth)
    cfg_hbmep.max_tree_depth = 10;
end
if not(isfield(cfg_hbmep, 'target_accept_prob')) || isempty(cfg_hbmep.target_accept_prob)
    cfg_hbmep.target_accept_prob = 0.8;
end

%%
response = response.join(' ');

% quote operation
q = @(x) ['"', x, '"'];

% Paths and arguments
d_hbmep = fullfile(fileparts(getenv('D_MHBMEP_GIT')), 'hbmep', '.venv');
if ispc
    p_hbmep = q(fullfile(d_hbmep, 'Scripts', 'python.exe'));
else
    p_hbmep = q(fullfile(d_hbmep, 'bin', 'python'));
end

if ~exist(replace(p_hbmep, '"', ''), 'file')
    warning('sp:call_hbmep:PythonNotFound', ...
        'hbMEP Python environment not found at: %s.\nSkipping hbMEP MCMC fitting.', replace(p_hbmep, '"', ''));
    return;
end

p_hbmep_function = q(fullfile(getenv('D_MHBMEP_GIT'), '+sp', 'hbmep_caller.py'));
d_output = q(strrep(p_csv, '.csv', sprintf('_hbmep%s', cfg_hbmep.es)));
p_postproc = q(cfg_hbmep.p_postproc);
p_csv = q(p_csv);
response = char(response);
units_intensity = char(cfg_hbmep.units_intensity);
units_mepsize = char(cfg_hbmep.units_mepsize);
model_name = char(cfg_hbmep.model);
stim_type = char(cfg_hbmep.stim_type);

for use_mixture = vec_mixture
    suffix = '';
    if use_mixture
        suffix = [suffix '_with_mixture'];
        str_mixture = '--use_mixture';
    else
        str_mixture = '';
    end
    if use_ima
        suffix = [suffix '_with_ima'];
        str_ima = '--use_ima';
        % Prepare output directory and write ima_params.json
        d_out_unquoted = replace(d_output, '"', '');
        p_target_dir = sprintf('%s%s', d_out_unquoted, suffix);
        if ~exist(p_target_dir, 'dir')
            mkdir(p_target_dir);
        end
        p_ima_json = fullfile(p_target_dir, 'ima_params.json');
        fid = fopen(p_ima_json, 'w');
        if fid ~= -1
            fprintf(fid, '%s', jsonencode(cfg_hbmep.ima_params));
            fclose(fid);
            str_ima_params = sprintf('--p_ima_params "%s"', p_ima_json);
        else
            str_ima_params = '';
        end
    else
        str_ima = '';
        str_ima_params = '';
    end
    d_output_local = sprintf('%s%s', d_output, suffix);
    
    if ismember(lower(units_mepsize), {'uvs', 'µvs'})
        metric_str = 'AUC (Area Under Curve)';
    else
        metric_str = 'Peak-to-Peak (pk-pk)';
    end
    fprintf('\n------------------ Executing hbMEP MCMC ------------------\n');
    fprintf('  Model            : %s\n', model_name);
    fprintf('  Stimulation Mode : %s\n', upper(stim_type));
    fprintf('  MEP Metric       : %s [%s]\n', metric_str, units_mepsize);
    fprintf('  Mixture Outliers : %s\n', string(use_mixture));
    fprintf('  IMA (Involuntary): %s\n', string(use_ima));
    fprintf('  MCMC Parameters  : warmup=%d, samples=%d, chains=%d, thinning=%d\n', ...
        cfg_hbmep.num_warmup, cfg_hbmep.num_samples, cfg_hbmep.num_chains, cfg_hbmep.thinning);
    fprintf('----------------------------------------------------------\n\n');

    % Build the system call
    command = sprintf('%s %s --p_csv %s --response %s --units_mepsize %s --units_intensity %s --d_output %s --p_postproc %s --model %s --stim_type %s --num_warmup %d --num_samples %d --num_chains %d --thinning %d --max_tree_depth %d --target_accept_prob %0.2f %s %s %s', ...
        p_hbmep, p_hbmep_function, p_csv, response, units_mepsize, units_intensity, d_output_local, p_postproc, model_name, stim_type, ...
        cfg_hbmep.num_warmup, cfg_hbmep.num_samples, cfg_hbmep.num_chains, cfg_hbmep.thinning, cfg_hbmep.max_tree_depth, cfg_hbmep.target_accept_prob, ...
        str_mixture, str_ima, str_ima_params);
    
    fprintf('Calling:\n')
    fprintf('%s\n\n', command);
    status = system(command);

% Check execution status
if status ~= 0
    error('System call failed. Check the inputs.');
end
fprintf('Results written to:\n')
fprintf('%s', d_output_local);
fprintf('\n---\n');
end