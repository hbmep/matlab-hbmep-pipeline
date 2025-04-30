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
%%
response = response.join(' ');

% quote operation
q = @(x) ['"', x, '"'];

% Paths and arguments
d_hbmep = fullfile(getenv('D_MHBMEP_GIT'), 'auxf', 'hbmep', '.venv');
if ispc
    p_hbmep = q(fullfile(d_hbmep, 'Scripts', 'python.exe'));
else
    p_hbmep = q(fullfile(d_hbmep, 'bin', 'python'));
end

p_hbmep_function = q(fullfile(getenv('D_MHBMEP_GIT'), '+sp', 'hbmep_caller.py'));
p_hbmep_config = q(fullfile(getenv('D_MHBMEP_GIT'), 'auxf', 'internal', 'hbmep_config.toml'));
d_output = q(strrep(p_csv, '.csv', sprintf('_hbmep%s', cfg_hbmep.es)));
p_csv = q(p_csv);
d_output = q(d_output);
response = char(response);
units_intensity = char(cfg_hbmep.units_intensity);
units_mepsize = char(cfg_hbmep.units_mepsize);

% Build the system call
command = sprintf('%s %s --p_hbmep_config %s --p_csv %s --response %s --units_mepsize %s --units_intensity %s --d_output %s', ...
    p_hbmep, p_hbmep_function, p_hbmep_config, p_csv, response, units_mepsize, units_intensity, d_output);

fprintf('Calling:\n')
fprintf('%s\n\n', command);
status = system(command);

% Check execution status
if status ~= 0
    error('System call failed. Check the inputs.');
end
fprintf('Results written to:\n')
fprintf('%s', d_output);
fprintf('\n---\n');
end