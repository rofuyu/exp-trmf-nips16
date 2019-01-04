function [ret] = rolling_foreast_trmf(Y, lag_idx, k, lambdas, maxit, do_lasso, window_size, nr_windows, missing, logtransform)
	if nargin <= 8
		missing = 1;
		threshold = 0; % assume all Y values are nonnegative
	end
	if nargin <= 9
		logtransform = 0;
	end

	n = max(Y(:,1));
	T = max(Y(:,2));

	if missing
		threshold = min(Y(:,3));
	end
	origY = Y;
	if logtransform
		Y(:,3)=log2(Y(:,3)); 
	end
	if do_lasso
		%fprintf(1,'perform lasso %d on w\n', do_lasso);
		;
	end

	trn_start = 1;
	trn_end = T-nr_windows*window_size;
	mse = 0;
	mape = 0;
	nd = 0;  % normalized deviation
	nd_den = 0;  % denominator for normalized deviation
	forecastY = [];
	cnt = 0;
	for i = 1:(nr_windows),
		trn_end = T - (nr_windows-i+1)*window_size;
		trn_idx = trn_start:trn_end;
		idx = find(Y(:,2)<= trn_end);
		if i == 1
			m = trmf_train([n,trn_idx(end)], Y(idx, :), lag_idx, k, lambdas, maxit, missing, do_lasso);
		else 
			m = trmf_train([n,trn_idx(end)], Y(idx, :), lag_idx, k, lambdas, maxit, missing, do_lasso, m);
		end
		models{i} = m;
		test_idx = (trn_end+1):(trn_end+window_size);
		[Ynew Xnew] = trmf_forecast(m, window_size);
		if logtransform 
			Ynew = 2.^Ynew;
		end
		Ynew(Ynew<threshold) = threshold; % for real-value;
		forecastY = [forecastY Ynew];
		idx = find(Y(:,2)>=test_idx(1) & Y(:,2)<=test_idx(end));
		trueY = sparse(origY(idx,1), origY(idx,2)-test_idx(1)+1, origY(idx,3), n, length(test_idx));
		tmp = sparse(origY(idx,1), origY(idx,2)-test_idx(1)+1, ones(length(idx),1), n, length(test_idx));
		idx = find(tmp~=0);
		if missing % Y contains missing values
			mse = mse + norm(Ynew(idx) - trueY(idx),'fro')^2;
			mape = mape + sum(sum(abs(Ynew(idx)-trueY(idx))./trueY(idx)));
			nd = nd + sum(sum(abs(Ynew(idx)-trueY(idx))));
			nd_den = nd_den + sum(sum(abs(trueY(idx))));
			cnt = cnt + nnz(tmp);
		else % fully observation
			mse = mse + norm(Ynew-trueY, 'fro')^2;
			mape = mape + sum(sum(abs(Ynew-trueY)./abs(trueY)));
			nd = nd + sum(sum(abs(Ynew-trueY)));
			nd_den = nd_den + sum(sum(abs(trueY)));
			cnt = cnt + prod(size(trueY));
		end
	end
	ret.mse = mse/cnt; 
	ret.nrmse = sqrt(ret.mse)/(nd_den/cnt);
	ret.mape = mape/cnt;
	ret.nd = nd/nd_den;
	ret.forecastY = forecastY;
	ret.models = models;
end
