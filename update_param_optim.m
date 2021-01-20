function [new_theta] = update_param_optim(theta, dtheta, optim_par)
% Optimization function
% theta and dtheta need to have the same dimensions, this is the only requirement

    eta = optim_par.learn_rate;
    thresh = optim_par.grad_threshold;

    switch(optim_par.update_meth)
        case 1 % gradient descent
            new_theta = theta - eta*dtheta;
        case 2 % gradient descent with clipping 
            grad_norm = norm(dtheta);
            if (grad_norm > thresh)
                dtheta = (thresh/grad_norm)*dtheta;
            end    
            new_theta = theta - eta*dtheta;
    end


end

