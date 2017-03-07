% Fixed Point Continuation (FPC) for l1 Regularized Least Squares
%
%--------------------------------------------------------------------------
% GENERAL DESCRIPTION & INPUTS
%-------------------------------------------------------------------------- 
%
% Out = fpc(n,A,b,mu,M,opts,varargin)
%
% Solves
%
%   min ||x||_1 + (mu/2)*||Ax - b||_M^2.
%
% A may be an explicit m x n matrix, or a function handle that implements 
% A*x and A'*x.  If the latter, this function must have the form
%
%   function y = name(trans,m,n,x,inds,varargin)
%
% where
%
%   trans    - if false then y = A(:,inds)*x, if true then y = A(:,inds)'*x 
%   m, n     - A is m x n
%   x        - length(inds) x 1 vector if ~trans; m x 1 vector if trans
%   inds     - vector of indices; if empty the function should return A*x 
%              or A'*x, as appropriate
%   varargin - placeholder for additional parameters
%
% b must be an m x 1 vector.
%
% M can be any m x m positive definite matrix, or the empty matrix.  If M
% is empty, fpc assumes M = I, which reduces the second term of the 
% objective to (mu/2)*||Ax - b||_2^2 (standard least squares). 
%
% This function assumes that the maximum eigenvalue of A^T M A is less
% than or equal to 1.  If your initial problem does not satisfy this
% condition, an equivalent problem may be constructed by setting:
%   sigma^2 = max eigenvalue of A^T M A
%   mu = mu*sigma^2
%   A = A/sigma
%   b = b/sigma
% Also see getM_mu.m.
%
% fpc_opts.m describes the available options.  If opts is passed as empty,
% fpc_opts will be called to obtain the default values.
%
% All variables in varargin are passed to A if A is a function handle.
%
%--------------------------------------------------------------------------
% OUTPUTS
%--------------------------------------------------------------------------
%
%   Out.x    - x at last iteration
%   Out.f    - vector of function values
%   Out.lam  - vector of ||x||_1
%   Out.step - vector of norm(x - xp)
%   Out.mus  - vector of mu values for each outer iteration
%   Out.itr  - number of iterations to convergence (or Inf if reach mxitr)
%   Out.itrs - vector of number of inner iterations completed during each
%              outer iteration
%   Out.tau  - value of tau
%   Out.n2re - if opts.xs exists, is vector of norm(x - xs)/norm(xs).
%              starts with 0th iteration.
%--------------------------------------------------------------------------

function Out = fpc(n,A,b,mu,M,opts,varargin)

% problem dimension
m = length(b);

% implicit or explicit A
imp = isa(A,'function_handle');
% calculate AtMb
if imp
    if isempty(M), AtMb = A(true,m,n,b,[],varargin{:});
    else AtMb = A(true,m,n,M*b,[],varargin{:}); end
else
    if isempty(M), AtMb = A'*b;
    else AtMb = A'*(M*b); end
end

% check for 0 solution
if mu <= 1/norm(AtMb,'inf'); 
    Out.x = zeros(n,1); Out.itr = 0; Out.itrs = 0; 
    Out.tau = 0; Out.mus = mu; Out.lam = 0; Out.step = [];
    if isempty(M), Out.f = (mu/2)*(b'*b); 
    else Out.f = (mu/2)*(b'*(M*b)); end
    if isfield(opts,'xs'), Out.n2re = 1; end
    return
end

% get opts
if isempty(opts), opts = fpc_opts([]); end

% initialize x, nu, tau, mu
muf = mu;                       % final value of mu
[x,nu,tau,mu] = fpc_init(n,m,b,AtMb,M,opts);
if mu > muf, mu = muf; nu = tau/mu; end
Out.mus = mu; Out.tau = tau;

% initialize Out.n2re
if isfield(opts,'xs'), xs = opts.xs; else xs = []; end
if ~isempty(xs), Out.n2re = norm(x - xs)/norm(xs); end

xtol = opts.xtol;
gtol = opts.gtol;

% prepare for iterations
Out.step = []; Out.itrs = []; Out.f = []; Out.lam = [];
Out.itr = Inf; oitr = 0; Ax = [];

% main loop
for i = 1:opts.mxitr
    
    % store old point
    xp = x;
    
    % get gradient at x and store objective function
	[g,f,lam] = get_g(x,imp,m,n,mu,A,b,M,AtMb,varargin{:});
    Out.f = [Out.f; f]; Out.lam = [Out.lam; lam];
    
    % take fixed-point step
    y = x - tau*g; 
    x = sign(y).*max(0,abs(y)-nu);
    
    nrmxxp = norm(x - xp);
    Out.step = [Out.step; nrmxxp]; 
    
    if ~isempty(xs), Out.n2re = [Out.n2re; norm(x - xs)/norm(xs)]; end
    
    crit1 = nrmxxp/max(norm(xp),1);
    crit2 = mu*norm(g,'inf') - 1;
    
    if (crit1 < xtol*sqrt(muf/mu)) && (crit2 < gtol)
        
        oitr = oitr + 1;
        
        if isempty(Out.itrs), Out.itrs = i;
        else Out.itrs = [Out.itrs; i - sum(Out.itrs)]; end
        
        % stop if reached muf
        if mu == muf
            Out.x = x; Out.itr = i;
            [g,f,lam] = get_g(x,imp,m,n,mu,A,b,M,AtMb,varargin{:});
            Out.f = [Out.f; f]; Out.lam = [Out.lam; lam];
            return 
        end
        
        % update mu
        if opts.fullMu
            phi = sqrt((2/mu)*(f - lam));
            mu = getNextMu(n,phi,opts.eta,opts.kappa);
        else
            mu = opts.eta*mu;
        end
        mu = min(mu,muf); nu = tau/mu;          
        Out.mus = [Out.mus; mu];
    end
end

% did not converge within opts.mxitr
Out.x = x;
if isempty(Out.itrs), Out.itrs = i;
else Out.itrs = [Out.itrs; i - sum(Out.itrs)]; end

end % fpc

%--------------------------------------------------------------------------
% SUBFUNCTION FOR INITIALIZATION
%--------------------------------------------------------------------------
%
% OUTPUTS -----------------------------------------------------------------
% x   - initialized based on opts.x0 and opts.init.  if opts.x0 exists, 
%       x = opts.x0, otherwise, opts.init determines x:
%           0 - x = zeros(n,1)
%           1 - x = tau*||AtMb||_Inf * ones(n,1)
%           2 - x = tau*AtMb
% nu  - equals tau/mu
% tau - equals opts.tau if exists, otherwise min(1.999,-1.665*m/n+2.665)
% mu  - set based on x = 0, mu = 1/norm(AtMb,inf), and getNextMu
%--------------------------------------------------------------------------

function [x,nu,tau,mu] = fpc_init(n,m,b,AtMb,M,opts)

% initialize tau
if isfield(opts,'tau'), tau = opts.tau;
else tau = min(1.999,-1.665*m/n + 2.665); end

% initialize x
if isfield(opts,'x0')
    x = opts.x0;
    if length(x) ~= n, error('User supplied x0 is wrong size.'); end
else
    switch opts.init
        case 0, x = zeros(n,1);
        case 1, x = tau*norm(AtMb,inf)*ones(n,1);
        case 2, x = tau*AtMb;
    end
end

% initialize mu
if opts.fullMu
    if isempty(M), phi = norm(b);       % phi = ||Ax - b||_M
    else phi = sqrt(b'*(M*b)); end
    mu = getNextMu(n,phi,opts.eta,opts.kappa);
else
    mu = opts.eta/norm(AtMb,inf);
end

% initialize nu
nu = tau/mu;

end % fpc_init

%--------------------------------------------------------------------------
% SUBFUNCTION FOR CALCULATING NEXT mu
%--------------------------------------------------------------------------
%
% Calculates the next value of mu based on taking a predictor step along
% the pareto curve phi(lam).  The derivative of this curve is derived in 
% 
% van den Berg, E. and M. Friedlander.  In pursuit of a root.  Preprint,
%   2007.
%
% The steplength is chosen so that phi(current)/phi(next) \approx eta.
% Mu is chosen so that the true phi(next) is guaranteed to be <= the 
% predicted value.
%
% INPUTS ------------------------------------------------------------------
% n     - length of x
% phi   - ||Ax - b||_M
% g     - A'M(Ax - b)
% eta   - parameter for choosing how much phi should decrease during next
%         outer iteration.  getNextMu is approximately the same as choosing
%         mu = eta*mu.
% kap   - condition number of A'MA.  See getM_mu.
%--------------------------------------------------------------------------

function mu = getNextMu(n,phi,eta,kap)

mu = eta*sqrt(n*kap)/phi;

end % getNextMu

%--------------------------------------------------------------------------
% SUBFUNCTION FOR CALCULATING g
%--------------------------------------------------------------------------

function [g,f,lam] = get_g(x,imp,m,n,mu,A,b,M,AtMb,varargin)

% get A*x
if imp
	Ax = A(false,m,n,x,[],varargin{:});
else
    Ax = A*x;
end

% calc g
if imp
    if isempty(M)
        g = A(true,m,n,Ax,[],varargin{:})-AtMb;
    else
        g = A(true,m,n,M*Ax,[],varargin{:})-AtMb;
    end
elseif isempty(M)
    g = A'*Ax - AtMb;
else
    g = A'*(M*Ax) - AtMb;
end

% calc f
r = Ax - b; lam = sum(abs(x));
if isempty(M)
    f = 0.5*mu*norm(r)^2 + lam;
else
    f = 0.5*mu*r'*(M*r) + lam;
end

end % get_g

% Copyright (c) 2007.  Elaine Hale, Wotao Yin, and Yin Zhang
%
% Last modified 28 August 2007.