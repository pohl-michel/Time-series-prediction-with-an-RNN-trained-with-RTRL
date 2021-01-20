function [pred_par] = load_pred_par(path_par)
% Load the parameters concerning prediction,
% which are initially stored in the file named path_par.pred_par_filename.
%
% Author : Pohl Michel
% Date : January 20th, 2021
% Version : v1.1
% License : 3-clause BSD License

    path_par.pred_par_filename = sprintf('%s\\%s', path_par.input_seq_dir, path_par.pred_par_filename_suffix);

    opts = detectImportOptions(path_par.pred_par_filename);
    opts = setvartype(opts,'double');
    opts.DataRange = '2:2';
    pred_par = table2struct(readtable(path_par.pred_par_filename, opts));
    
    pred_par.pred_meth_str = 'RTRL_RNN';
    pred_par.NORMALIZE_DATA = true;
    pred_par.update_meth = 2;
    switch(pred_par.update_meth)
        case 1
            pred_par.update_meth_str = 'gradient descent';
        case 2
            pred_par.update_meth_str = 'gradient descent with clipping';
    end
    
end

