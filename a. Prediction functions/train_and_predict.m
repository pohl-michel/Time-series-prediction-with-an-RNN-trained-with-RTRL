function [Ypred, avg_pred_time, pred_loss_function] = train_and_predict(path_par, pred_par, beh_par )
% This function trains a recurrent neural network (RNN) with real-time recurrent learning (RTRL)
% for the prediction of time series data.
% Training and prediction are performed multiple times to account for the random initialization of the initial synaptic weights.
% 
% Note : even though the evaluation of the RNN performance is done later using the assumption that the data represent the 3D position of multiple objects,
% the function train_and_predict does not use such assumption and enables predicting any type of multidimensional time series data.
%
% Author : Pohl Michel
% Date : January 20th, 2020
% Version : v2.0
% License : 3-clause BSD License

    % loading the "past" data matrix X and the "future" data matrix Y.
    [ X, Y, Mu, Sg] = load_pred_data_XY( path_par, pred_par, beh_par);
            
    fprintf('Training a Recurrent Neural Network (RNN) with real time recurrent learning (RTRL) \n');

    [p, M] = size(Y); %p is the RNN output dimension
    Ypred = zeros([size(Y), pred_par.nb_runs]);
    avg_pred_time = zeros(pred_par.nb_runs, 1);
    pred_loss_function = zeros(M, pred_par.nb_runs);

    myRNN = initialize_rnn( pred_par, beh_par, p, M);

    for run_idx=1:pred_par.nb_runs
        % we run the prediction algorithm with different initial weigths

        %if (mod(run_idx,10) == 1)
            fprintf('Running the %d-th test (random initialization) \n', run_idx);
        %end

        myRNN = rnn_RTRL( myRNN, pred_par, beh_par, X, Y); 

        Ypred(:,:,run_idx) =  myRNN.Ypred;
        avg_pred_time(run_idx) = mean(myRNN.pred_time_array);
        pred_loss_function(:,run_idx) = myRNN.pred_loss_function;

        myRNN = reset_rnn(myRNN, pred_par, beh_par);

    end
    
    if pred_par.NORMALIZE_DATA
        for run_idx=1:pred_par.nb_runs
            Ypred(:,:,run_idx) = uncenterZ( Ypred(:,:,run_idx), Mu, Sg );
        end
    end
    
    if beh_par.SAVE_PRED_RESULTS
        pred_results_filename = write_pred_result_variables_filename(path_par, pred_par);
        save(pred_results_filename, 'Ypred', 'avg_pred_time', 'pred_loss_function');
    end

end

