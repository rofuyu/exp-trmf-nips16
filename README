This is a README for an experimental code of the paper 

Temporal Regularized Matrix Factorization for High Dimensional Time Series Prediction.


Requirements
============
	g++: note that clang++ shipped by Mac is not supported. Please ``brew install g++'' using Mac Homebrew on a Mac machine. 
	matlab/octave: Please ``brew install octave'' using Mac Brew if you are on a Mac machine.

Install
=======
	on a matlab/octave shell, type 
	
	> install


Data Set
========
	See data/ folder for traffic and electricity datasets


Usage of trmf_train.m
=====================
function model = trmf_train(siz, Ycoo, lag_idx, k, lambdas, maxit, missing, do_lasso, model0);

	siz = (n, T) : the dimension of the observation matrix
	Ycoo         : size(Ycoo) = a matrix of |\Omega|-by-3 for all the observations (coordinate format)
	lag_idx      : an array of lag indices included in L
	k            : the latent dimension
	lambdas      : an array of length 3, lambdas = [lambdaF, lambdaX, lambdaw]
	maxit        : max number of iterations
	missing      : whether there are missing values in Y. In our experiments, you can always use 1
	do_lasso     : impost L1-regularization on w or not, you can always use 0
	model0       : the initial model to start with.


	Output model:
		model.F = F;
		model.X = X;
		model.lag_idx = lag_idx;
		model.lag_val = lag_val;
	
Usage of trmf_forcast.m
=======================
function [Ynew, Xnew] = trmf_forecast(model, len)

	model         : the model obtained by trmf_train
	len           : forecast the next `len` values

	Output: 
		Ynew      : n-by-len array, observation forecast
		Xnew      : k-by-len array, temporal embedding forecast 


Usage of rolling_forecast_trmf.m
================================
function [ret] = rolling_foreast_trmf(Ycoo, lag_idx, k, lambdas, maxit, do_lasso, window_size, nr_windows, missing)
		
	window_size   : the size of forecast period
	nr_windows    : number of rolling forecast periods
	other parameters are the same as trmf_train


Usage of grid_forecast_trmf
===========================
function [cv_result, best] = grid_forecast_trmf(Ycoo, kset, lambda1, lambda2, lambda3, window_size, nr_window, lag_set)

	Ycoo            : size(Y) = a matrix of |\Omega|-by-3 for all the observations (coordinate format)
	kset            : an array of possible values for `k` 
	lamdda1         : an array of possible values for `lambdaF`
	lamdda2         : an array of possible values for `lambdaX`
	lamdda3         : an array of possible values for `lambdaw`


Usage of imputation_trmf
========================
function [ret] = imputation_trmf(Ymat, observed, lag_idx, k, lambdas, maxit)

	Ymat            : n-by-T array, the entire observation Y
	observed        : a sparse indicator matrix for the observation in Y

	ret.model       : the TRMF model obtained by running the TRMF algorithm on the observed entries




Citation
========
Please acknowledge the use of the code with a citation.
@InProceedings{   HFY16a,
  title={Temporal Regularized Matrix Factoriztion for High-dimensional Time Series Prediction},
  author={Yu, Hsiang-Fu and Rao, Nikhil and Dhillon, Inderjit S.},
  booktitle = {Advances in Neural Information Processing Systems 28},
  year={2016}
}


If you have any questions regarding the code, feel free to contact Hsiang-Fu Yu (rofuyu at cs utexas edu). 

