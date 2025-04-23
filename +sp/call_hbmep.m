function call_hbmep(p_csv, response, varargin)
d.es = '';

v = parse_inputs(d, varargin{:});

%%
response = response.join(' ');

% quote operation
q = @(x) ['"', x, '"'];

% Paths and arguments
d_hbmep = fullfile(getenv('D_GIT'), 'auxf', 'hbmep', '.venv');
if ispc
    p_hbmep = q(fullfile(d_hbmep, 'Scripts', 'python.exe'));
else
        p_hbmep = q(fullfile(d_hbmep, 'bin', 'python'));
end
p_hbmep_function = q(fullfile(getenv('D_GIT'), '+sp', 'hbmep_caller.py'));
p_hbmep_config = q(fullfile(getenv('D_GIT'), 'auxf', 'internal', 'hbmep_config.toml'));
d_output = q(strrep(p_csv, '.csv', sprintf('_hbmep%s', v.es)));
p_csv = q(p_csv);
d_output = q(d_output);
response = char(response);

% Build the system call
command = sprintf('%s %s --p_hbmep_config %s --p_csv %s --response %s --d_output %s', ...
    p_hbmep, p_hbmep_function, p_hbmep_config, p_csv, response, d_output);

fprintf('Calling:\n')
fprintf('%s\n\n', command);
status = system(command);

% Check execution status
if status ~= 0
    error('System call failed. Check the inputs.');
end
end