function [y_mep, t_slice] = slice_mat(X, ix_stim, ix_slice, fs, vec_channel_name)
mat_slice = ix_stim + ix_slice;
case_rm = (mat_slice<1) | mat_slice>size(X, 1);
mat_slice(case_rm) = 1;
y_mep = nan([size(mat_slice), length(vec_channel_name)]);
for ix = 1:length(vec_channel_name)
    str_ch = vec_channel_name(ix);
    x = X.(str_ch);
    y_mep(:, :, ix) = x(mat_slice);
end
y_mep(case_rm) = nan;
t_slice = ix_slice / fs;
end