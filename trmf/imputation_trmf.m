function [ret] = imputation_trmf(Ymat, observed, lag_idx, k, lambdas, maxit)
	idx = ~(observed); % idx to missing entries

	observed = find(observed > 0);
	[n, T] = size(Ymat);
	Y = zeros(size(Ymat));
	Y(observed) = Ymat(observed)+1;
	[i j v] = find(Y);
	Ycoo = [i j v-1];

	missing = 1;
	do_lasso = 0;

	model = trmf_train([n,T], Ycoo, lag_idx, k, lambdas, maxit, missing, do_lasso);
	Ynew =model.F *model.X;
	trueY = Ymat;

	mse = norm(Ynew(idx) - trueY(idx),'fro')^2;
	mape = sum(sum(abs(Ynew(idx)-trueY(idx))./trueY(idx)));
	nd = sum(sum(abs(Ynew(idx)-trueY(idx))));
	nd_den = sum(sum(abs(trueY(idx))));
	cnt = nnz(idx);

	ret.mse = mse/cnt; 
	ret.nrmse = sqrt(ret.mse)/(nd_den/cnt);
	ret.mape = mape/cnt;
	ret.nd = nd/nd_den;
	ret.model = model;
end


