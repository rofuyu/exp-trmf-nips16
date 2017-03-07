function [x,x_debias,objective,times,debias_start,mses,taus]= ...
    SpaRSA(y,A,tau,varargin)

% SpaRSA version 2.0, December 31, 2007
% 
% This function solves the convex problem 
%
% arg min_x = 0.5*|| y - A x ||_2^2 + tau phi(x)
%
% using the SpaRSA algorithm, which is described in "Sparse Reconstruction
% by Separable Approximation" by S. Wright, R. Nowak, M. Figueiredo, 
% IEEE Transactions on Signal Processing, 2009 (to appear).
%
% The algorithm is related GPSR (Figueiredo, Nowak, Wright) but does not
% rely on the conversion to QP form of the l1 norm, because it is not
% limited to being used with l1 regularization. Instead it forms a separable
% approximation to the first term of the objective, which has the form
%
%  d'*A'*(A x - y) + 0.5*alpha*d'*d 
%
% where alpha is obtained from a BB formula. In a monotone variant, alpha is
% increased until we see a decreasein the original objective function over
% this step. 
%
% -----------------------------------------------------------------------
% Copyright (2007): Mario Figueiredo, Robert Nowak, Stephen Wright
%
% GPSR is distributed under the terms
% of the GNU General Public License 2.0.
% 
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ----------------------------------------------------------------------
% 
% Please check for the latest version of the code and paper at
% www.lx.it.pt/~mtf/SpaRSA
%
%  ===== Required inputs =============
%
%  y: 1D vector or 2D array (image) of observations
%     
%  A: if y and x are both 1D vectors, A can be a 
%     k*n (where k is the size of y and n the size of x)
%     matrix or a handle to a function that computes
%     products of the form A*v, for some vector v.
%     In any other case (if y and/or x are 2D arrays), 
%     A has to be passed as a handle to a function which computes 
%     products of the form A*x; another handle to a function 
%     AT which computes products of the form A'*x is also required 
%     in this case. The size of x is determined as the size
%     of the result of applying AT.
%
%  tau: regularization parameter (scalar)
%
%  ===== Optional inputs =============
%
%  
%  'AT'    = function handle for the function that implements
%            the multiplication by the conjugate of A, when A
%            is a function handle. If A is an array, AT is ignored.
%
%  'Psi'   = handle to the denoising function, that is, to a function
%            that computes the solution of the densoing probelm 
%            corresponding to the desired regularizer. That is, 
%            Psi(y,tau) = arg min_x (1/2)*(x - y)^2 + tau phi(x).
%            Default: in the absence of any Phi given by the user,
%            it is assumed that phi(x) = ||x||_1 thus 
%            Psi(y,tau) = soft(y,tau)
%            Important: if Psi is given, phi must also be given,
%                       so that the algorithm may also compute
%                       the objective function.
%
%  'StopCriterion' = type of stopping criterion to use
%                    0 = algorithm stops when the relative 
%                        change in the number of non-zero 
%                        components of the estimate falls 
%                        below 'ToleranceA'
%                    1 = stop when the relative 
%                       change in the objective function 
%                       falls below 'ToleranceA'
%                    2 = stop when relative duality gap 
%                       falls below 'ToleranceA'
%                    3 = stop when LCP estimate of relative
%                       distance to solution falls below ToleranceA
%                    4 = stop when the objective function 
%                        becomes equal or less than toleranceA.
%                    5 = stop when the norm of the difference between 
%                        two consecutive estimates, divided by the norm
%                        of one of them falls below toleranceA
%                    Default = 2
%
%  'ToleranceA' = stopping threshold; Default = 0.01
% 
%  'Debias'     = debiasing option: 1 = yes, 0 = no.
%                 Default = 0.
%
%  'ToleranceD' = stopping threshold for the debiasing phase:
%                 Default = 0.0001.
%                 If no debiasing takes place, this parameter,
%                 if present, is ignored.
%
%  'MaxiterA' = maximum number of iterations allowed in the
%               main phase of the algorithm.
%               Default = 1000
%
%  'MiniterA' = minimum number of iterations performed in the
%               main phase of the algorithm.
%               Default = 5
%
%  'MaxiterD' = maximum number of iterations allowed in the
%               debising phase of the algorithm.
%               Default = 200
%
%  'MiniterD' = minimum number of iterations to perform in the
%               debiasing phase of the algorithm.
%               Default = 5
%
%  'Initialization' must be one of {0,1,2,array}
%               0 -> Initialization at zero. 
%               1 -> Random initialization.
%               2 -> initialization with A'*y.
%           array -> initialization provided by the user.
%               Default = 0;
%
%  'BB_variant' specifies which variant of Barzila-Borwein to use, or not.
%               0 -> don't use a BB rule - instead pick the starting alpha
%               based on the successful value at the previous iteration
%               1 -> standard BB choice  s'r/s's
%               2 -> inverse BB variant r'r/r's
%               Default = 1
%
%  'BB_cycle' specifies the cycle length  - the number of iterations between
%             recalculation of alpha. Requires integer value at least
%             1. Relevant only if a **nonmonotone BB rule** is used 
%             (BB_variant = 1 or 2 and Monotone=0).
%             Default = 1
%
%  'Monotone' =  enforce monotonic decrease in f, or not? 
%               any nonzero -> enforce monotonicity (overrides 'Safeguard')
%               0 -> don't enforce monotonicity.
%               Default = 0;
%
%  'Safeguard' = enforce a "sufficient decrease" over the largest
%               objective value of the past M iterations.
%               any nonzero -> safeguard
%               0 -> don't safeguard
%               Default = 0.
%
%  'M'        = number of steps to look back in the safeguarding process.
%               Ignored if Safeguard=0 or if Monotone is nonzero.
%               (positive integer. Default = 5)
%
%  'sigma'    = sigma value used in Safeguarding test for sufficient 
%               decrease. Ignored unless 'Safeguard' is nonzero. Must be
%               in (0,1). Drfault: .01.
%
%  'Eta'      = factor by which alpha is multiplied within an iteration,
%               until a decrease in the objective function is
%               obtained.
%               Default = 2;
%
%  'Alpha_factor' = factor by which to reduce the successful value of
%                alpha at iteration k, to give the first value of alpha
%                to be tried at iteration k+1.
%                If a Barzilai-Borwein rule is specified (BB_variant > 0), 
%                this parameter is ignored.
%                Default = 0.8;
%
%  'Continuation' = Continuation or not (1 or 0) 
%                   Specifies the choice for a continuation scheme,
%                   in which we start with a large value of tau, and
%                   then decrease tau until the desired value is 
%                   reached. At each value, the solution obtained
%                   with the previous values is used as initialization.
%                   Default = 0
%
% 'ContinuationSteps' = Number of steps in the continuation procedure;
%                       ignored if 'Continuation' equals zero.
%                       If -1, an adaptive continuation procedure is used.
%                       Default = -1.
% 
% 'FirstTauFactor'  = Initial tau value, if using continuation, is
%                     obtained by multiplying the given tau by 
%                     this factor. This parameter is ignored if 
%                     'Continuation' equals zero or 
%                     'ContinuationSteps' equals -1.
%                     Default = 10.
%
%  'True_x' = if the true underlying x is passed in 
%                this argument, MSE plots are generated.
%
%  'AlphaMin' = the alphamin parameter of the BB method.
%               Default = 1e-30;
%
%  'AlphaMax' = the alphamax parameter of the BB method.
%               Default = 1e30;
%
%  'Verbose'  = work silently (0) or verbosely (1)
%
% ===================================================  
% ============ Outputs ==============================
%   x = solution of the main algorithm
%
%   x_debias = solution after the debiasing phase;
%                  if no debiasing phase took place, this
%                  variable is empty, x_debias = [].
%
%   objective = sequence of values of the objective function
%
%   times = CPU time after each iteration
%
%   debias_start = iteration number at which the debiasing 
%                  phase started. If no debiasing took place,
%                  this variable is returned as zero.
%
%   mses = sequence of MSE values, with respect to True_x,
%          if it was given; if it was not given, mses is empty,
%          mses = [].
% ========================================================


% start the clock
t0 = cputime;
times(1) = cputime - t0;
taus = 1;

% test for number of required parametres
if (nargin-length(varargin)) ~= 3
     error('Wrong number of required parameters');
end

% Set the defaults for the optional parameters
maxiterBIG = 200;
stopCriterion = 2;
tolA = 0.01;
tolD = 0.0001;
debias = 0;
maxiter = 10000;
maxiter_debias = 200;
miniter = 5;
miniter_debias = 0;
init = 0;
bbVariant = 1;
bbCycle = 1;
enforceMonotone = 0;
enforceSafeguard = 0;
M = 5;
sigma = .01;
alphamin = 1e-30;
alphamax = 1e30;
compute_mse = 0;
AT = 0;
verbose = 0;
continuation = 0;
cont_steps = -1;
psi_ok = 0;
phi_ok = 0;
% amount by which to increase alpha after an unsuccessful step
eta = 2.0;
% amount by which to decrease alpha between iterations, if a
% Barzilai-Borwein rule is not used to make the initial guess at each
% iteration. 
alphaFactor = 0.8;
phi_l1 = 0;

% Set the defaults for outputs that may not be computed
debias_start = 0;
x_debias = [];
mses = [];

% Read the optional parameters
if (rem(length(varargin),2)==1)
  error('Optional parameters should always go by pairs');
else
  for i=1:2:(length(varargin)-1)
    switch upper(varargin{i})
     case 'PSI'
       psi_function = varargin{i+1};
     case 'PHI'
       phi_function = varargin{i+1};
     case 'STOPCRITERION'
       stopCriterion = varargin{i+1};
     case 'TOLERANCEA'       
       tolA = varargin{i+1};
     case 'TOLERANCED'
       tolD = varargin{i+1};
     case 'DEBIAS'
       debias = varargin{i+1};
     case 'MAXITERA'
       maxiter = varargin{i+1};
     case 'MAXITERD'
       maxiter_debias = varargin{i+1};
     case 'MINITERA'
       miniter = varargin{i+1};
     case 'MINITERD'
       miniter_debias = varargin{i+1};
     case 'INITIALIZATION'
       if prod(size(varargin{i+1})) > 1   % we have an initial x
	 init = 33333;    % some flag to be used below
	 x = varargin{i+1};
       else 
	 init = varargin{i+1};
       end
     case 'BB_VARIANT'
       bbVariant = varargin{i+1};
     case 'BB_CYCLE'
       bbCycle = varargin{i+1};
     case 'MONOTONE'
       enforceMonotone = varargin{i+1};
     case 'SAFEGUARD'
       enforceSafeguard = varargin{i+1};
     case 'M'
       M = varargin{i+1};
     case 'SIGMA'
       sigma = varargin{i+1};
     case 'ETA'
       eta = varargin{i+1};
     case 'ALPHA_FACTOR'
       alphaFactor = varargin{i+1};
     case 'CONTINUATION'
       continuation = varargin{i+1};  
     case 'CONTINUATIONSTEPS' 
       cont_steps = varargin{i+1};
     case 'FIRSTTAUFACTOR'
       firstTauFactor = varargin{i+1};
     case 'TRUE_X'
       compute_mse = 1;
       true = varargin{i+1};
     case 'ALPHAMIN'
       alphamin = varargin{i+1};
     case 'ALPHAMAX'
       alphamax = varargin{i+1};
     case 'AT'
       AT = varargin{i+1};
     case 'VERBOSE'
       verbose = varargin{i+1};
     otherwise
      % Hmmm, something wrong with the parameter string
      error(['Unrecognized option: ''' varargin{i} '''']);
    end;
  end;
end
%%%%%%%%%%%%%%

% it makes no sense to ask for a nonmonotone variant of a non-BB method
if ~enforceMonotone && bbVariant==0
    error(['non-monotone, non-BBmethod requested']);
end

if (sum(stopCriterion == [0 1 2 3 4 5])==0)
   error(['Unknown stopping criterion']);
end

% if A is a function handle, we have to check presence of AT,
if isa(A, 'function_handle') && ~isa(AT,'function_handle')
   error(['The function handle for transpose of A is missing']);
end 

% if A is a matrix, we find out dimensions of y and x,
% and create function handles for multiplication by A and A',
% so that the code below doesn't have to distinguish between
% the handle/not-handle cases
if ~isa(A, 'function_handle')
   AT = @(x) A'*x;
   A = @(x) A*x;
end
% from this point down, A and AT are always function handles.

% Precompute A'*y since it'll be used a lot
Aty = AT(y);

% if phi was given, check to see if it is a handle and that it 
% accepts two arguments
if exist('psi_function','var')
   if isa(psi_function,'function_handle')
      try  % check if phi can be used, using Aty, which we know has 
           % same size as x
            dummy = psi_function(Aty,tau); 
            psi_ok = 1;
      catch
         error(['Something is wrong with function handle for psi'])
      end
   else
      error(['Psi does not seem to be a valid function handle']);
   end
else %if nothing was given, use soft thresholding
   psi_function = @(x,tau) soft(x,tau);
end

% if psi exists, phi must also exist
if (psi_ok == 1)
   if exist('phi_function','var')
%       if isa(phi_function,'function_handle')
%          try  % check if phi can be used, using Aty, which we know has 
%               % same size as x
%               dummy = phi_function(Aty); 
%          catch
%            error(['Something is wrong with function handle for phi'])
%          end
%       else
%         error(['Phi does not seem to be a valid function handle']);
%       end
   else
      error(['If you give Psi you must also give Phi']); 
   end
else  % if no psi and phi were given, simply use the l1 norm.
   phi_function = @(x) sum(abs(x(:))); 
   phi_l1 = 1;
end


% Initialization
switch init
    case 0   % initialize at zero, using AT to find the size of x
       x = AT(zeros(size(y)));
    case 1   % initialize randomly, using AT to find the size of x
       x = randn(size(AT(zeros(size(y)))));
    case 2   % initialize x0 = A'*y
       x = Aty; 
    case 33333
       % initial x was given as a function argument; just check size
       if size(A(x)) ~= size(y)
          error(['Size of initial x is not compatible with A']); 
       end
    otherwise
       error(['Unknown ''Initialization'' option']);
end

% if the true x was given, check its size
if compute_mse && (size(true) ~= size(x))  
  error(['Initial x has incompatible size']); 
end
 
% if tau is large enough, in the case of phi = l1, thus psi = soft,
% the optimal solution is the zero vector
if phi_l1
   aux = AT(y);
   max_tau = max(abs(aux(:)));
   firstTauFactor = 0.8*max_tau / tau;
   if (tau >= max_tau) && (psi_ok==0)
      x = zeros(size(aux));
      if debias
         x_debias = x;
      end
      objective(1) = 0.5*(y(:)'*y(:));
      times(1) = 0;
      if compute_mse
        mses(1) = sum(true(:).^2);
      end
      return
   end
end

% define the indicator vector or matrix of nonzeros in x
nz_x = (x ~= 0.0);
num_nz_x = sum(nz_x(:));

% store given tau, because we're going to change it in the
% continuation procedure
final_tau = tau;
% if we choose to use adaptive continuation, need to reset tau to realmax to
% make things work (don't ask...)
if cont_steps == -1
  tau = realmax;
end

% store given stopping criterion and threshold, because we're going 
% to change them in the continuation procedure
final_stopCriterion = stopCriterion;
final_tolA = tolA;

% set continuation factors

if (continuation && phi_l1 && (cont_steps > 1))
  % If tau is scalar, first check top see if the first factor is 
  % too large (i.e., large enough to make the first 
  % solution all zeros). If so, make it a little smaller than that.
  if prod(size(tau)) == 1
    if firstTauFactor*tau >= max_tau
      firstTauFactor = 0.5 * max_tau / tau;
      if verbose
	fprintf(1,'\n setting parameter FirstTauFactor\n')
      end
    end
    cont_factors = 10.^[log10(firstTauFactor):...
	  log10(1/firstTauFactor)/(cont_steps-1):0];
  end
else
  if ( ~continuation )
    cont_factors = 1;
    cont_steps = 1;
  end
end

keep_continuation = 1;
cont_loop = 1;
iter = 1;
taus = [];

% loop for continuation
contiters = 0;
while keep_continuation 
  contiters = contiters + 1;
  if contiters == 200
      break
  end
  % initialize the count of steps since last update of alpha 
  % (for use in cyclic BB)
  iterThisCycle = 0;
  
  % Compute the initial residual and gradient
  resid =  A(x) - y;
  gradq = AT(resid);
  
  if cont_steps == -1
     
     temp_tau = max(final_tau,0.2*max(abs(gradq(:))));
     
     if temp_tau > tau
        tau = final_tau;    
     else
        tau = temp_tau;
     end
     
     if tau <= final_tau
        stopCriterion = final_stopCriterion;
        tolA = final_tolA;
        keep_continuation = 0;
     else
        stopCriterion = 1;
        tolA = 1e-5;
     end
  else
     tau = final_tau * cont_factors(cont_loop);
     if cont_loop == cont_steps
        stopCriterion = final_stopCriterion;
        tolA = final_tolA;
        keep_continuation = 0;
     else
        stopCriterion = 1;
        tolA = 1e-5;
     end
  end
  
  taus = [taus tau];
  
  if verbose
    fprintf('\n Regularization parameter tau = %10.6e\n',tau)
  end
  
  % compute and store initial value of the objective function 
  % for this tau
  alpha = 1; %1/eps;
  
  f = 0.5*(resid(:)'*resid(:)) + tau * phi_function(x);
  if enforceSafeguard
    f_lastM = f;
  end

  % if we are at the very start of the process, store the initial mses and
  % objective in the plotting arrays
  if cont_loop==1
    objective(1) = f;
    if compute_mse
      mses(1) = (x(:)-true(:))'*(x(:)-true(:));
    end
    if verbose
      fprintf(1,'Initial obj=%10.6e, alpha=%6.2e, nonzeros=%7d\n',...
	  f,alpha,num_nz_x);
    end
  end
 
  % initialization of alpha
  % alpha = 1/max(max(abs(du(:))),max(abs(dv(:))));
  % or just do a dumb initialization 
  %alphas(iter) = alpha;
  
  % control variable for the outer loop and iteration counter
  keep_going = 1;
  
  while keep_going
    
    % compute gradient
    gradq = AT(resid);
    
    % save current values
    prev_x = x;
    prev_f = f;
    prev_resid = resid;
    
    % computation of step
    
    cont_inner = 1;
    while cont_inner
      x = psi_function(prev_x - gradq*(1/alpha),tau/alpha);
      dx = x - prev_x;
      Adx = A(dx);
      resid = prev_resid + Adx;
      f = 0.5*(resid(:)'*resid(:)) + tau * phi_function(x);
      if enforceMonotone
	f_threshold = prev_f;
      elseif enforceSafeguard
	f_threshold = max(f_lastM) - 0.5*sigma*alpha*(dx(:)'*dx(:));
      else
	f_threshold = inf;
      end
       % f_threshold
      
      if f <= f_threshold
	cont_inner=0;
      else
	% not good enough, increase alpha and try again
	alpha = eta*alpha;
	if verbose
	  fprintf(1,' f=%10.6e, increasing alpha to %6.2e\n', f, alpha);
	end
      end
    end   % of while cont_inner

    if enforceSafeguard
      if length(f_lastM)<M+1
	f_lastM = [f_lastM f];
      else
	f_lastM = [f_lastM(2:M+1) f];
      end
    end
    
    % print stuff
    if verbose
      fprintf(1,'t=%4d, obj=%10.6e, alpha=%e  ', iter, f, alpha );
    end
    
    if bbVariant==1
      % standard BB choice of initial alpha for next step
      if iterThisCycle==0 || enforceMonotone==1
	dd  = dx(:)'*dx(:);  
	dGd = Adx(:)'*Adx(:);
	alpha = min(alphamax,max(alphamin,dGd/(realmin+dd)));
      end
    elseif bbVariant==2
      % alternative BB choice of initial alpha for next step
      if iterThisCycle==0 || enforceMonotone==1
	dd  = dx(:)'*dx(:);  
	dGd = Adx(:)'*Adx(:);
	ATAdx=AT(Adx);
	dGGd = ATAdx(:)'*ATAdx(:);
	alpha = min(alphamax,max(alphamin,dGGd/(realmin+dGd)));
      end
    else  
      % reduce current alpha to get initial alpha for next step
      alpha = alpha * alphaFactor;
    end

    % update iteration counts, store results and times
    iter=iter+1; 
    iterThisCycle=mod(iterThisCycle+1,bbCycle);
    objective(iter) = f;
    times(iter) = cputime-t0;
    % alphas(iter) = alpha;
    if compute_mse
      err = true - x;
      mses(iter) = (err(:)'*err(:));
    end
    
    % compute stopping criteria and test for termination
    switch stopCriterion
        case 0,
            % compute the stopping criterion based on the change
            % of the number of non-zero components of the estimate
            nz_x_prev = nz_x;
            nz_x = (abs(x)~=0.0);
            num_nz_x = sum(nz_x(:));
            num_changes_active = (sum(nz_x(:)~=nz_x_prev(:)));
            if num_nz_x >= 1
                criterionActiveSet = num_changes_active / num_nz_x;
                keep_going = (criterionActiveSet > tolA);
            end
            checkme = criterionActiveSet;
            if verbose
                fprintf(1,'Delta nz = %d (target = %e)\n',...
                    criterionActiveSet , tolA)
            end
        case 1,
            % compute the stopping criterion based on the relative
            % variation of the objective function.
            criterionObjective = abs(f-prev_f)/(prev_f);
            keep_going =  (criterionObjective > tolA);
            checkme = criterionObjective;
            if verbose
                fprintf(1,'Delta obj. = %e (target = %e)\n',...
                    criterionObjective , tolA)
            end
        case 2,
            % compute the "duality" stopping criterion - actually based on the
            % iterate PRIOR to the step just taken. Make it relative to the primal
            % function value.
            scaleFactor = norm(gradq(:),inf);
            w = tau*prev_resid(:) / scaleFactor;
            criterionDuality = 0.5* (prev_resid(:)'*prev_resid(:)) + ...
                tau * phi_function(prev_x) + 0.5*w(:)'*w(:) + y(:)'*w(:);
            criterionDuality = criterionDuality / prev_f;
            keep_going = (criterionDuality > tolA);
            checkme = criterionDuality;
            if verbose
                fprintf(1,'Duality = %e (target = %e)\n',...
                    criterionDuality , tolA)
            end
        case 3,
            % compute the "LCP" stopping criterion - again based on the previous
            % iterate. Make it "relative" to the norm of x.
            w = [ min(tau + gradq(:), max(prev_x(:),0.0)); ...
                min(tau - gradq(:), max(-prev_x(:),0.0))];
            criterionLCP = norm(w(:), inf);
            criterionLCP = criterionLCP / max(1.0e-6, norm(prev_x(:),inf));
            checkme = criterionLCP;
            keep_going = (criterionLCP > tolA);
            if verbose
                fprintf(1,'LCP = %e (target = %e)\n',criterionLCP,tolA)
            end
        case 4,
            % continue if not yeat reached target value tolA
            keep_going = (f > tolA);
            checkme = f;
            if verbose
                fprintf(1,'Objective = %e (target = %e)\n',f,tolA)
            end
        case 5,
            % stopping criterion based on relative norm of step taken
            delta_x_criterion = sqrt(dx(:)'*dx(:))/(x(:)'*x(:));
            checkme = delta_x_criterion;
            keep_going = (delta_x_criterion > tolA);
            if verbose
                fprintf(1,'Norm(delta x)/norm(x) = %e (target = %e)\n',...
                    delta_x_criterion,tolA)
            end
        otherwise,
            error(['Unknown stopping criterion']);
    end % end of the stopping criteria switch
    
    % overrule the stopping decision to ensure we take between miniter and
    % maxiter iterations
    if iter<=miniter
      % take no fewer than miniter... 
      keep_going = 1;
    elseif iter > maxiter
        %% THIS IS THE ADD-ON <<nr>>
        if iter < maxiterBIG && checkme >= 1
            keep_going = 1;
        elseif iter < maxiterBIG && checkme < 1
            keep_going = 0;
        elseif iter >=maxiterBIG
            keep_going = 0;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %and no more than maxiter iterations  
      %keep_going = 0;
    end
    
  end % end of the main loop of the GPBB algorithm (while keep_going)
  
  cont_loop = cont_loop + 1;
  
end % end of the continuation loop (while keep_continuation) 

% Print results
if verbose
  fprintf(1,'\nFinished the main algorithm!  Results:\n')
  fprintf(1,'Number of iterations = %d\n',iter)
  fprintf(1,'0.5*||A x - y ||_2^2 = %10.3e\n',0.5*resid(:)'*resid(:))
  fprintf(1,'tau * Penalty = %10.3e\n',tau * phi_function(x))
  fprintf(1,'Objective function = %10.3e\n',f);
  fprintf(1,'Number of non-zero components = %d\n',sum(x(:)~=0));
  fprintf(1,'CPU time so far = %10.3e\n', times(iter));
  fprintf(1,'\n');
end

% If the 'Debias' option is set to 1, we try to
% remove the bias from the l1 penalty, by applying CG to the 
% least-squares problem obtained by omitting the l1 term 
% and fixing the zero coefficients at zero.

if (debias && (sum(x(:)~=0)~=0))
  if verbose
    fprintf(1,'\nStarting the debiasing phase...\n\n')
  end
  
  x_debias = x;
  zeroind = (x_debias~=0); 
  cont_debias_cg = 1;
  debias_start = iter;
  
  % calculate initial residual
  resid = A(x_debias);
  resid = resid-y;
  prev_resid = eps*ones(size(resid));
  
  rvec = AT(resid);
  
  % mask out the zeros
  rvec = rvec .* zeroind;
  rTr_cg = rvec(:)'*rvec(:);
  
  % set convergence threshold for the residual || RW x_debias - y ||_2
  tol_debias = tolD * (rvec(:)'*rvec(:));
  
  % initialize pvec
  pvec = -rvec;
  
  % main loop
  while cont_debias_cg
    
    % calculate A*p = Wt * Rt * R * W * pvec
    RWpvec = A(pvec);      
    Apvec = AT(RWpvec);
    
    % mask out the zero terms
    Apvec = Apvec .* zeroind;
    
    % calculate alpha for CG
    alpha_cg = rTr_cg / (pvec(:)'* Apvec(:));
    
    % take the step
    x_debias = x_debias + alpha_cg * pvec;
    resid = resid + alpha_cg * RWpvec;
    rvec  = rvec  + alpha_cg * Apvec;
    
    rTr_cg_plus = rvec(:)'*rvec(:);
    beta_cg = rTr_cg_plus / rTr_cg;
    pvec = -rvec + beta_cg * pvec;
    
    rTr_cg = rTr_cg_plus;
    
    iter = iter+1;
    
    objective(iter) = 0.5*(resid(:)'*resid(:)) + ...
	                  tau * phi_function(x_debias(:));
    times(iter) = cputime - t0;
    
    if compute_mse
      err = true - x_debias;
      mses(iter) = (err(:)'*err(:));
    end
    
    % in the debiasing CG phase, always use convergence criterion
    % based on the residual (this is standard for CG)
    if verbose
       fprintf(1,'t = %5d, debias resid = %13.8e, convergence = %8.3e\n', ...
	   iter, resid(:)'*resid(:), rTr_cg / tol_debias);
    end
    cont_debias_cg = ...
     	(iter-debias_start <= miniter_debias )| ...
	    ((rTr_cg > tol_debias) & ...
	    (iter-debias_start <= maxiter_debias));
    
  end
  if verbose
  fprintf(1,'\nFinished the debiasing phase! Results:\n')
  fprintf(1,'Final number of iterations = %d\n',iter);
  fprintf(1,'0.5*||A x - y ||_2 = %10.3e\n',0.5*resid(:)'*resid(:))
  fprintf(1,'tau * penalty = %10.3e\n',tau * phi_function(x))
  fprintf(1,'Objective function = %10.3e\n',f);
  fprintf(1,'Number of non-zero components = %d\n',...
          sum((x_debias(:)~=0.0)));
  fprintf(1,'CPU time so far = %10.3e\n', times(iter));
  fprintf(1,'\n');
  end
end

if compute_mse
  mses = mses/length(true(:));
end

end

