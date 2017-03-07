function [cv_result, best] = grid_forecast_trmf(Ycoo, kset, lambda1, lambda2, lambda3, window_size, nr_window, lag_set)
if nargin < 2
	kset = [5 10 20 40];
end
if nargin < 3
	lambda1 = [50, 5, 0.5, 0.05];
end
if nargin < 4
	lambda2 = [50, 5, 0.5, 0.05];
end
if nargin < 5
	lambda3 = [50, 5, 0.5, 0.05];
end
if nargin < 6
	window_size=6;
end
if nargin < 7
	nr_window = 9;
end
if nargin < 8
	lag_set = [1:10, 50:56];
end
maxit=5;
do_lasso = 0;

cv_result = [];
best.nrmse = 100000;
best.nd = 100000;
best.nd_idx = 0;
best.nrmse_idx = 0;
for k = kset
	for l1 = lambda1
		for l2 = lambda2
			for l3 = lambda3
				lambdas = [l1, l2, l3];
				ret = rolling_forecast_trmf(Ycoo, lag_set, k, lambdas, maxit, do_lasso, window_size, nr_window);
				cv_result = [cv_result; k, l1, l2, l3, ret.nd, ret.nrmse, ret.nd, ret.mape];
				if ret.nrmse < best.nrmse
					fprintf(1,'Best nrmse: %g (%d %g %g %g)\n', ret.nrmse, k, l1, l2, l3);
					best.nrmse = ret.nrmse;
					best.nrmse_idx = size(cv_result, 1);
				end
				if ret.nd < best.nd
					fprintf(1,'Best nd   : %g (%d %g %g %g)\n',ret.nd, k, l1, l2, l3);
					best.nd = ret.nd;
					best.nd_idx = size(cv_result, 1);
				end
			end
		end
	end
end
