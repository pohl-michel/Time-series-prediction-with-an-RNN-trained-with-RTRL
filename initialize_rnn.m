function [ myRNN ] = initialize_rnn( pred_par, beh_par, p, M)
% Initialization of the variables controlling the internal dynamics of the RNN.
% The variable p represents the dimension of the RNN output space.
% myRNN is a cell array and myRNN{k} contains the variables associated to the k-th run. 
% We perform several runs for taking into account the randomness of the weights initialization.
% The symnaptic weights are indeed randomly distributed according to a normal distribution of standard deviation sg = pred_par.Winit_std_dev. 
%
% Author : Pohl Michel
% Date : January 20th, 2020
% Version : v2.0
% License : 3-clause BSD License

    m = pred_par.SHL*p;
    q = pred_par.rnn_state_space_dim;
    sg = pred_par.Winit_std_dev;

    % Redundant but makes the variables easier to access in other functions
    myRNN.input_space_dim = m;
    myRNN.state_space_dim = q;
    myRNN.output_space_dim = p;
    myRNN.nb_weights = q*(p+q+m+1);
    
    % evaluation variables initialization
    myRNN.Ypred = zeros(p, M);
    myRNN.pred_loss_function = zeros(M, 1, 'single');
    myRNN.pred_time_array = zeros(M, 1, 'single');  
    
    % weights initialization
    myRNN.Wa = normrnd(0, sg, [q, q]);
    myRNN.Wb = normrnd(0, sg, [q, m+1]); % "+1" à cause du biais
    myRNN.Wc = normrnd(0, sg, [p, q]);

    % states initialization
    myRNN.x = zeros(q, 1);

    % activation function and its derivative
    myRNN.phi = @(v) tanh(v);
    myRNN.phi_prime = @(v) 1./((cosh(v)).^2);

    % state space dynamics 3D tensor  
    myRNN.LBDA = zeros(q, q + m + 1, q);
        % myRNN.LBDA(:,:,j) is the matrix LBDA_{j,n} , ie the Jacobian matrix of x_n as a function of w_j (j in 1,...q)
        % cf the Haykin's book

    % w(:,j) corresponds to the w_j matrix at time n in Haykin's book
    % w_j = [wa_j , wb_j]
    % with Wa^T = [wa_1, ..., wa_q] and Wb^T = [wb_1, ..., wb_q]
    myRNN.w = zeros(m+q+1, q);

    % gradient of the instantaneous energy loss En with respect to each entry in w (corresponding to the gradients with respect to Wa and Wb) - cf Haykin's book
    myRNN.w_gradient = zeros(m+q+1, q);

    % matrix U{:,:,j) ( "U_{j,n}" in Haykin's book)
    myRNN.U = zeros(q, m+q+1, q);    
    
    if beh_par.GPU_COMPUTING
        
        myRNN.Ypred = gpuArray(myRNN.Ypred);
        myRNN.pred_loss_function = gpuArray(myRNN.pred_loss_function);
        myRNN.pred_time_array = gpuArray(myRNN.pred_time_array);

        myRNN.Wa = gpuArray(myRNN.Wa);
        myRNN.Wb = gpuArray(myRNN.Wb);
        myRNN.Wc = gpuArray(myRNN.Wc);
        myRNN.x = gpuArray(myRNN.x);
        
        myRNN.LBDA = gpuArray(myRNN.LBDA);
        myRNN.w = gpuArray(myRNN.w);
        myRNN.w_gradient = gpuArray(myRNN.w_gradient);
        myRNN.U = gpuArray(myRNN.U);
        
    end
        
end