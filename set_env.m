function set_env
if ispc
    userDir = winqueryreg('HKEY_CURRENT_USER',...
        ['Software\Microsoft\Windows\CurrentVersion\' ...
        'Explorer\Shell Folders'],'Personal');
    userDir = fileparts(userDir);
else
    userDir = char(java.lang.System.getProperty('user.home'));
end

% Defaults
project = 'matlab-hbmep-pipeline';
setenv('D_USER', userDir);
setenv('D_PROC', fullfile(userDir, project, 'proc'));
setenv('D_GIT', fileparts(mfilename("fullpath")));
setenv('D_TEMP', tempdir);  % session persistent tempdir

addpath(genpath(fullfile(getenv('D_GIT'), 'auxf', 'internal')));
addpath(genpath(fullfile(getenv('D_GIT'), 'auxf', 'matlab-toml')));
if isempty(which('dnc_set_env'))
    % rely on defaults above
else
    dnc_set_env(userDir);
end

%%
if isempty(getenv('DATETIME_SESSION'))
    % get a datetime, but only once per session
    dt = datetime;
    dt.Format = 'uuuu-MM-dd';
    setenv('DATETIME_SESSION', char(dt));
end

end
