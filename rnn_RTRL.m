function [ myRNN] = rnn_RTRL( myRNN, pred_par, beh_par, Xdata, Ydata)
% rnn_RTRL performs the training and prediction of a recurrent neural network (RNN) trained with real-time recurrent learning (RTRL) and gradient
% clipping.
% Input variables :
%   - myRNN : RNN structure previously initialized by the function "initialize_rnn"
%   - pred_par is the structure containing the parameters used for prediction
%   - Xdata is the matrix of the "past data" and Ydata the matrix of the "future" data given by the function "load_pred_data_XY".
% Output variables :
%   - myRNN is the updated RNN structure containing in particular :
%       - the predicted time series "myRNN.Ypred"
%       - the values of the loss function "myRNN.pred_loss_function"
%       - the array containing the time for making each prediction "myRNN.pred_time_array"
% 
% Author : Pohl Michel
% Date : January 20th, 2020
% Version : v2.0
% License : 3-clause BSD License

   [~, M] = size(Xdata);
    m = myRNN.input_space_dim;
    q = myRNN.state_space_dim;
    p = myRNN.output_space_dim;

    % initializing some auxiliary variables
    switch beh_par.GPU_COMPUTING
        case true
            aux_vec = gpuArray.ones(q, 1);      % used for calculating Phi_n
            I = reshape(gpuArray.eye(q),q,1,q); % used for calculating U
        case false
            aux_vec = ones(q, 1);
            I = reshape(eye(q),q,1,q);
    end  
    
    for t=1:M

        tic
        
        myRNN.w(1:q,:) = transpose(myRNN.Wa);
        myRNN.w((q+1):(m+q+1),:) = transpose(myRNN.Wb);
        
        % I] predicting the future data and calculating the instantaneous prediction error
        myRNN.Ypred(:,t) = myRNN.Wc*myRNN.x;
        e = Ydata(:,t) - myRNN.Ypred(:,t);
       
        % II] update of the synaptic weights w and Wc
        aux_mat = transpose(myRNN.Wc)*e;
        switch beh_par.GPU_COMPUTING
            case true
                lmbda_transpose = pagefun(@transpose, myRNN.LBDA);
                myRNN.w_gradient = - squeeze(pagefun(@mtimes, lmbda_transpose, aux_mat));
            case false
                for j = 1:q
                    myRNN.w_gradient(:,j) = - transpose(myRNN.LBDA(:,:,j))*aux_mat;
                end
        end  
        Wc_gradient = - kron(e,transpose(myRNN.x)); %tensor multiplication 
        
        updated_weights = update_param_optim([myRNN.w; myRNN.Wc], [myRNN.w_gradient; Wc_gradient], pred_par);
        w_next = updated_weights(1:(m+q+1), :);
        myRNN.Wc = updated_weights((m+q+2):(m+p+q+1), :);
        
        % III] update of the dynamics matrix LAMBDA 
        u = Xdata(:,t); % input vector of size m+1
        ksi_n_transposed = [transpose(myRNN.x), u.']; % input-state vector
        aux_fun = @(i) myRNN.phi_prime(ksi_n_transposed*myRNN.w(:,i));
        Phi_n = diag(aux_fun(aux_vec));
        myRNN.U = I.*ksi_n_transposed;     
        
        switch beh_par.GPU_COMPUTING
            case true
                a = pagefun(@mtimes, myRNN.Wa, myRNN.LBDA);
                b = pagefun(@plus, a, myRNN.U);
                myRNN.LBDA = pagefun(@mtimes, Phi_n, b);
            case false
                for j=1:q
                    myRNN.LBDA(:,:,j) = Phi_n*(myRNN.Wa*myRNN.LBDA(:,:,j)+myRNN.U(:,:,j));
                end
        end          
        
        % IV] update of the state vector
        myRNN.x = myRNN.phi(myRNN.Wa*myRNN.x + myRNN.Wb*u);
        
        % "V] or IIbis]" update of Wa and Wb
        myRNN.Wa = transpose(w_next(1:q,:));        
        myRNN.Wb = transpose(w_next((q+1):(q+m+1),:));
        
        myRNN.pred_time_array(t) = toc;
        
        myRNN.pred_loss_function(t) = 0.5*(e.')*e;
            % error evaluation so it is performed after time performance evaluation
        
    end
    
    if beh_par.GPU_COMPUTING
        myRNN.Ypred = gather(myRNN.Ypred);
        myRNN.pred_time_array = gather(myRNN.pred_time_array);
        myRNN.pred_loss_function = gather(myRNN.pred_loss_function);
    end
    
end

