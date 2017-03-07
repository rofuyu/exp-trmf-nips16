function [Ynew, Xnew] = trmf_forecast(model, len)
	lag_idx = model.lag_idx;
	lag_val = model.lag_val;
	F = model.F;
	X = model.X;
	T = size(X,2);
	Xnew = [X zeros(size(X,1), len)];
	for t = (T+1):(T+len),
		Xnew(:, t) = sum(Xnew(:, t - lag_idx) .* lag_val, 2);
	end
	Xnew = Xnew(:, (T+1):end);
	Ynew = F*Xnew;
end
