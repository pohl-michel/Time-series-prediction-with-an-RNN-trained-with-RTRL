function [ myRNN ] = reset_rnn(myRNN, pred_par, beh_par)
% Re-initialize the RNN after the computation task for one run
%
% Author : Pohl Michel
% Date : January 20th, 2020
% Version : v2.0
% License : 3-clause BSD License

    m = myRNN.input_space_dim;
    q = myRNN.state_space_dim;
    p = myRNN.output_space_dim;

    sg = pred_par.Winit_std_dev;
    
    myRNN.Ypred(:) = 0;
    myRNN.pred_loss_function(:) = 0;
    myRNN.pred_time_array(:) = 0; 
    
    % weights re-initialization
    myRNN.Wa = normrnd(0, sg, [q, q]);
    myRNN.Wb = normrnd(0, sg, [q, m+1]); % "+1" à cause du biais
    myRNN.Wc = normrnd(0, sg, [p, q]);

    % states re-initialization
    myRNN.x(:) = 0;

    % state space dynamics 3D tensor  
    myRNN.LBDA(:) = 0;
        % myRNN.LBDA(:,:,j) is the matrix LBDA_{j,n} , ie the Jacobian matrix of x_n as a function of w_j (j in 1,...q)
        % cf the Haykin's book

    % w(:,j) corresponds to the w_j matrix at time n in Haykin's book
    % w_j = [wa_j , wb_j]
    % with Wa^T = [wa_1, ..., wa_q] and Wb^T = [wb_1, ..., wb_q]
    myRNN.w(:) = 0;

    % gradient of the instantaneous energy loss En with respect to each entry in w (corresponding to the gradients with respect to Wa and Wb) - cf Haykin's book
    myRNN.w_gradient(:) = 0;

    % matrix U{:,:,j) ( "U_{j,n}" in Haykin's book)
    myRNN.U(:) = 0;    
    
    if beh_par.GPU_COMPUTING
        
        myRNN.Wa = gpuArray(myRNN.Wa); 
        myRNN.Wb = gpuArray(myRNN.Wb);
        myRNN.Wc = gpuArray(myRNN.Wc);
        myRNN.Ypred(:) = gpuArray(myRNN.Ypred);
        myRNN.pred_loss_function = gpuArray(myRNN.pred_loss_function);
        myRNN.pred_time_array = gpuArray(myRNN.pred_time_array);
        
    end     
        
end