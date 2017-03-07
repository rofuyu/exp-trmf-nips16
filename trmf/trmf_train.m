function model = trmf_train(siz, Y, lag_idx, k, lambdas, maxit, missing, do_lasso, model0)
	lang = 'matlab';
	m_path = mfilename('fullpath');
	[m_path, fname, ext] = fileparts(m_path);
	addpath([m_path '/trmf-core/' lang '/']);
	addpath([m_path '/trmf-core/SpaRSA/']);

	n = siz(1); T = siz(2);
	nz_lags = length(lag_idx);
	midx = max(lag_idx);
	lambdaF = lambdas(1);
	lambdaX = lambdas(2);
	lambdaw = lambdas(3);
	rand('seed', 0);
	F = rand(n, k);
	X = rand(k, T);

	randn('seed',0);
	lag_val = randn(k, nz_lags);
	%lag_val = zeros(k, nz_lags);
	%lag_val(:,1) = 1;
	%lag_val = zeros(k, nz_lags);
	%lag_val = ones(k, nz_lags) ./ sqrt(nz_lags); 

	if nargin < 8
		do_lasso = 0;
	end

	if missing
		solver = 31;
	else
		solver = 30;
	end

	% warm start
	if nargin >= 9
		model = model0;
		lag_idx = model0.lag_idx;
		lag_val = model0.lag_val;
		F = model.F;
		X(:,1:size(model0.X,2)) = model0.X;
	end

	%Y = sparse(Y);

	for it=1:maxit,
		% update F and X using Mex interface for the situation with missing entries
		[F, X] = arr_mf_train(Y, [], lag_idx, lag_val, F', X,...
		sprintf('-s %d -k %d -li %g -la %g -T 1 -g 10 -t 2 -n 16', solver, k, lambdaF, lambdaX));
		F = F';

		% update lag_val for AR(lag_idx,lag_val)
		for r=1:k,
			singleX = X(r,:)';
			idx = (midx+1):T;
			tmpY = singleX(idx);
			tmpX = zeros(length(idx), length(lag_idx));
			for i=1:length(lag_idx),
				lag = lag_idx(i);
				tmpX(:,i) = singleX(idx-lag);
			end
			if do_lasso == 0,
				% ridge
				lag_val(r,:) = ((tmpX'*tmpX + lambdaw*eye(size(lag_val,2)))\(tmpX'*tmpY))';
			elseif do_lasso == 1,
				% lasso
				[lv,lvd,~,~,~,~]=SpaRSA(tmpY,tmpX,lambdaw,...
					'Debias',1, ...
					'StopCriterion',1,...
					'Monotone',1,...
					'Continuation',1,...
					'MaxiterA',20, ...
					'ToleranceA',0.00001,...
					'Verbose',0 ...
					); 
				if isempty(lvd)
					lag_val(r,:) = lv;
				else
					lag_val(r,:) = lvd;
				end
				%tmp = lasso(tmpX, tmpY, 'Lambda', lambdaw, 'Standardize', false);
				%lag_val(r,:) = tmp';
			end
		end
		%fprintf(1,'it %d loss %g\n', it, 0.5*norm(Y-F*X,'fro')^2);
	end
	model.F = F;
	model.X = X;
	model.lag_idx = lag_idx;
	model.lag_val = lag_val;
end



