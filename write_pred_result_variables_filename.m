function [str] = write_pred_result_variables_filename(path_par, pred_par)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    str = sprintf('%s\\pred_result_variables %s %s.mat', path_par.temp_var_dir, path_par.time_series_dir, sprintf_pred_param(pred_par));

end

