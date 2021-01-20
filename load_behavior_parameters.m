function [ beh_par ] = load_behavior_parameters()
% The structure beh_par contains important information about the behavior of the whole algorithm.
%
% Author : Pohl Michel
% Date : January 20th, 2021
% Version : v2.0
% License : 3-clause BSD License

%% IMPORTANT PARAMETERS
beh_par.TRAIN_AND_PREDICT = true;
    % if set to true, the RNN will be trained and predict the future data.
beh_par.EVALUATE_PREDICTION = true;
    % if set to true, the RNN performance will be evaluated on the test set.
beh_par.SAVE_PREDICTION_PLOT = true;
    % if set to true, the predicted positions of the objects as well as the error loss function will be saved.
    
beh_par.SAVE_PRED_RESULTS = true;    
    % if set to true, the algorithm saves the prediction loss function, the
    % prediction time for each time step and the predicted position of the
    % objects

beh_par.GPU_COMPUTING = false;
    % used only with RTRL at the moment
    
end