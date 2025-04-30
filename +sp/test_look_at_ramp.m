clear;
addpath(fullfile('..', '..'));
addpath(genpath(fullfile('..', '..', 'auxf')));
set_env_mhbmep;
%%
o_data = '2024-11-19';
f_data = 'P02_s01r02_2024-11-19T142059';
p_csv = write_ramp_tables(o_data, f_data);
response = ["rfdi"];

call_hbmep(p_csv, response);