function [new_theta] = update_param_optim(theta, dtheta, optim_par)
% Optimization function
% theta and dtheta need to have the same dimensions, this is the only requirement
%
% v2.0 : correction : use of the Frobenius norm instead of the spectral 2 norm
%
% Author : Pohl Michel
% Date : April 11th, 2021
% Version : v2.0
% License : 3-clause BSD License

    eta = optim_par.learn_rate;
    thresh = optim_par.grad_threshold;

    switch(optim_par.update_meth)
        case 1 % gradient descent
            new_theta = theta - eta*dtheta;
        case 2 % gradient descent with clipping 
            grad_norm = sqrt(sum(dtheta.^2, 'all'));
            if (grad_norm > thresh)
                dtheta = (thresh/grad_norm)*dtheta;
            end    
            new_theta = theta - eta*dtheta;
    end


end

