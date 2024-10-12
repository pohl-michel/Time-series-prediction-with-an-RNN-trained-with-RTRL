Update (10th October 2024): An extension of this repository is available here: https://github.com/pohl-michel/2D-MR-image-prediction.
That new repository focuses on video forecasting but the ["Time_series_forecasting" folder](https://github.com/pohl-michel/2D-MR-image-prediction/tree/main/Time_series_forecasting) has several new functionalities, including a script to perform hyper-parameter optimization, the addition of other online algorithms for RNNs (unbiased online recurrent optimization, decoupled neural interfaces, sparse 1-step approximation, and a simpler implementation of real-time recurrent learning) and other evaluation metrics for general time series (not necessarily representing the motion of 3D objects).

----------------------------------------------
This repository is the second of a series of three repositories containing code that we used in the research corresponding to the following article: 

Michel Pohl, Mitsuru Uesaka, Kazuyuki Demachi, Ritu Bhusal Chhatkuli, "Prediction of the motion of chest internal points using a recurrent neural network trained with real-time recurrent learning for latency compensation in lung cancer radiotherapy", Computerized Medical Imaging and Graphics, Volume 91, 2021, 101941, ISSN 0895-6111

You can access it with the following links:
 - https://doi.org/10.1016/j.compmedimag.2021.101941 (journal version with restricted access)
 - https://doi.org/10.48550/arXiv.2207.05951 (accepted manuscript version, openly available)

The code in this repository predicts multidimensional time-series data using a recurrent neural network (RNN) trained by real-time recurrent learning (RTRL) with gradient clipping. The two other repositories corresponding to the article mentioned above are the following:
 - Lucas-Kanade pyramidal optical flow for 3D image sequences: https://github.com/pohl-michel/Lucas-Kanade-pyramidal-optical-flow-for-3D-image-sequences
 - 3D image warping using Nadaraya-Watson non-linear regression: https://github.com/pohl-michel/Time-series-prediction-with-an-RNN-trained-with-RTRL

Please kindly consider citing our published article if you use this code in your research. Also, please do not hesitate to look at the other two repositories mentioned above.

Please also have a look at the following repository, which extends the current repository by including training of RNNs with Unbiased Online Recurrent Optimization (UORO) and hyper-parameter optimization, as well as the associated research and blog articles:
 - repository: https://github.com/pohl-michel/time-series-forecasting-with-UORO-RTRL-LMS-and-linear-regression 
 - research article (journal version with restricted access): https://doi.org/10.1016/j.cmpb.2022.106908
 - research article (accepted manuscript version, openly available): https://doi.org/10.48550/arXiv.2106.01100
 - blog article (Medium): https://medium.com/towards-data-science/forecasting-respiratory-motion-using-online-learning-of-rnns-for-safe-radiotherapy-bdf4947ad22f
 - blog article (personal blog): https://pohl-michel.github.io/blog/articles/predicting-respiratory-motion-online-learning-rnn/article.html

The figure below gives an example of prediction 400ms in advance with RTRL (the sampling rate is 2.5Hz). 
![alt text](prediction_RTRL.png "prediction with RTRL a horizon of 400ms")

Our implementation is based on the chapter 15 ("Dynamically Driven Recurrent Networks") of the following book :
Haykin, Simon S. "Neural networks and learning machines/Simon Haykin." (2009).

The main script to execute is "prediction_main.m".
The data to be predicted is in the directory named "1. Input time series sequences", 
and represents the 3D motion of 3 implanted metal markers in the chest during lung cancer radiotherapy.
The markers move because of the respiratory motion and their position is sampled at approximately 2.5Hz.

The parameters concerning prediction and display should be set manually in the files "pred_par.xlsx" and "disp_par.xlsx", and the "load_pred_par.m" function.
For instance, one can choose to use training with gradient clipping or not by setting manually the field `update_meth` of the `pred_par` structure (line 28). 
The behavior of the program is controlled by the structure `beh_par` defined in "load_behavior_parameters.m".
In particular, one can choose to perform computations on the GPU by setting `beh_par.GPU_COMPUTING = true`.
The fields of that structure can be set manually.

The figures showing prediction on the test data and the loss function are saved in the following folders: 
"1. Prediction results (figures)" and "2. Prediction results (images)". 
The RNN results are saved in the folder "3. RNN variables (temp)".
The parameters used and the numerical RNN evaluation results are also recorded in a txt file located in the folder "4. Log txt files".

Note : prediction is made without any specific assumption on the nature of the temporal data,
but the evaluation of the RNN uses the assumption that the data represents the 3D position of several objects.
