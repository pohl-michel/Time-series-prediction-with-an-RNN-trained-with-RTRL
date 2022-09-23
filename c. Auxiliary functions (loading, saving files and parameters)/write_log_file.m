function write_log_file(path_par, beh_par, pred_par, eval_results)
% Records the parameters used and the prediction numerical results in a txt file.
% 
% Author : Pohl Michel
% Date : January 20th, 2020
% Version : v2.0
% License : 3-clause BSD License

    log_file_complete_filename = sprintf('%s\\%s %s %s', path_par.txt_file_dir, path_par.time_series_dir, pred_par.pred_meth_str, path_par.log_txt_filename);
    fid = fopen(log_file_complete_filename,'wt');
        
        fprintf(fid, 'sequence name : %s \n', path_par.input_seq_dir);
        fprintf(fid, '%s \n',path_par.date_and_time);
 
        % I] Recording the calculation paremeters        

        fprintf(fid, 'Prediction method : %s \n', pred_par.pred_meth_str);
        fprintf(fid, 'Training between t = 1 and t = %d \n', pred_par.tmax_training);
        fprintf(fid, 'Evaluation between t = %d and t = %d \n', pred_par.t_eval_start, pred_par.tmax_pred);
        fprintf(fid, 'Horizon of the prediction h = %d \n', pred_par.horizon);
        if pred_par.NORMALIZE_DATA
            fprintf(fid, 'data normalized before prediction\n');
        else
            fprintf(fid, 'data not normalized before prediction \n');
        end

        fprintf(fid, 'Signal history length k = %d \n', pred_par.SHL);
        fprintf(fid, 'Nb of neurons in the hidden layer q = %d \n', pred_par.rnn_state_space_dim);
        fprintf(fid, 'Learning rate eta = %g \n', pred_par.learn_rate);
        fprintf(fid, 'Synaptic weights standard deviation (initialization) sg = %g \n', pred_par.Winit_std_dev);
        fprintf(fid, 'Number of runs due to random weights initialization nb_runs = %d \n', pred_par.nb_runs);
        if (pred_par.update_meth == 2) % gradient clipping
            fprintf(fid, 'Gradient clipping with threshold grd_tshld = %f \n', pred_par.grad_threshold);
        else
            fprintf(fid, 'No gradient clipping');
        end
        if beh_par.GPU_COMPUTING 
            fprintf(fid, 'Computation with the GPU \n');
        else
            fprintf(fid, 'Computation with the CPU \n');
        end


        fprintf(fid, '\n');
        
        % II] Recording the evaluation results
        if beh_par.EVALUATE_PREDICTION
        
            fprintf(fid, 'Calculation time \n');
            fprintf(fid, 'Average time for predicting the position at t+%d given the data until t : %e s\n', pred_par.horizon, eval_results.mean_pt_pos_pred_time);
            fprintf(fid, '\n');

            fprintf(fid, 'Evaluation results \n');
            fprintf(fid, 'nb of prediction runs with numerical error (gradient explosion) : %d \n', eval_results.nb_xplosion);
            fprintf(fid, 'mean prediction error : %f mm \n', eval_results.mean_mean_err);
            fprintf(fid, 'mean prediction error 95%% confidence half range : %f (mm) \n', eval_results.confidence_half_range_mean_err);            
            fprintf(fid, '(mean of the) rms error : %f mm \n', eval_results.mean_rms_err);
            fprintf(fid, 'rms prediction error 95%% confidence half range : %f (mm) \n', eval_results.confidence_half_range_rms_err);  
            fprintf(fid, '(mean of the) max prediction error : %f mm \n', eval_results.mean_max_err);
            fprintf(fid, 'max prediction error 95%% confidence half range : %f (mm) \n', eval_results.confidence_half_range_max_err);            
            fprintf(fid, '(mean of the) jitter : %f mm \n', eval_results.mean_jitter);
            fprintf(fid, 'jitter 95%% confidence half range : %f (mm) \n', eval_results.confidence_half_range_jitter);    
            fprintf(fid, '(mean of the) NRMSE : %f \n', eval_results.mean_nrmse);
            fprintf(fid, 'NRMSE 95%% confidence half range : %f \n', eval_results.confidence_half_range_nrmse);                      
            
        end 

        fprintf(fid, '\n');
        
    fclose(fid);

end