function [ pred_param_str ] = sprintf_pred_param(pred_par)
% Returns a character string which contains information concerning the prediction parameters for saving and loading temporary variables
% 
% Author : Pohl Michel
% Date : August 12th, 2020
% Version : v1.0
% License : 3-clause BSD License

    if pred_par.NORMALIZE_DATA
        nrm_data_str = string('normalized data');
    else
        nrm_data_str = string('no normalization');
    end

    pred_param_str = sprintf('k=%d q=%d eta=%g sg=%g grd_tshld=%g h=%d %s', pred_par.SHL, pred_par.rnn_state_space_dim, ...
        pred_par.learn_rate, pred_par.Winit_std_dev, pred_par.grad_threshold, pred_par.horizon, nrm_data_str);
    % k = nb of time steps for performing one prediction
    % q = nb of neurons in the hidden layer
    % eta = learning rate
    % sg = standard deviation of the gaussian distribution of the initial weights values
    % grd_tshld = clipping value
        
end