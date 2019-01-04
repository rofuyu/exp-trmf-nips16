function [cv_result, best] = grid_imputaion_trmf(Ymat, observed, kset, lambda1, lambda2, lambda3, lag_set)

maxit=10;
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
				ret = imputation_trmf(Ymat, observed, lag_set, k, lambdas, maxit);
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

