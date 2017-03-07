function thetaHATgrp = OverlapLasso_sjw_v1(y,A,Aold,tau,groups,...
                                           group_arr,group_MAT,lam,varargin)

%Nikhil Rao
% 12/17/10
%Sangkyun Lee
%1/6/10
%Nikhil Rao
%1/26/11

% this function solves the overlap group lasso method, with replication of
% columns of A as explained in jacob et.al. It also implements the non
% replicating case, forcing the overlapping coefficients to be the same

%INPUTS : 
% y         = the observed vector
% A         = the measurement matrix
% Aold      = original measurement matrix for debiasing
% tau       = regularization parameter for the overlap lasso penalty
% groups    = row vector indicating groups
% group_MAT = cell array having groups as rows
% lam       = if non zero, it forces replicates to be close to each other

            
  [k,n] = size(Aold);
  yold = y;
  if lam
    mdiff = size(A,1) - size(Aold,1);
    y = [y; zeros(mdiff,1)];
  end
  
  if ~isempty(varargin)
      C = varargin{1};
  else
      C = eye(size(A,2),size(A,2));
  end
  
  % REGULARIZER

  psi = @(x,tau) group_vector_soft_sjw(x,tau,groups,group_arr);
  phi = @(x) group_l2norm_sjw(x,group_arr);
  
  % GROUP LASSO
  
  ttog=cputime;
  [thetaGrp,xg,~,~,~,~,taus]= ...
      SpaRSA(y,A,tau,...
             'Psi',psi,...
             'Phi',phi,...
             'Debias',1,...
             'StopCriterion',1,...
             'Monotone',1,...
             'Continuation',1,...
             'MaxiterA',10000, ...
             'ToleranceA',0.000001,...
             'Verbose',0 ...
             );
  ttog=cputime-ttog;
%  fprintf('\n OGL sparsa call: %8.4f\n',ttog);
  
  
  % RECONSTRUCTION
  if lam
    if ~isempty(xg)
      thetaGrp = xg;
    end
    thetaHATgrp = thetaGrp(1:n);
    
    
  else
    thetaGrp = C*thetaGrp;
    ttog=cputime;
    % THIS IS THE DEBIASING STEP FOR GROUP LASSO WITH
    % REPLICATION
    
    % 1. find groups that have been selected
    g = []; % this indicates the groups that are selected
    keep = find(thetaGrp ~= 0);
    %if is empty, then it causes problems, hence, let keep correspond
    %to all the elements in thetaHAT
    if isempty(keep)
      keep = 1:length(thetaGrp);
    end
    
    %  determine the "group numbers" to select. i.e. the row number of
    %  the group_MAT matrix that corresponds to the non zero element
    %  locations of the group LASSO estimate
    
    g = groups(keep);
    
    g = unique(g);% these are the groups that have been selected
    
    % 2. now that we know the groups, collapse the repetitions , due to
    % overlap
    selgroups = [];
    for j = 1:length(g)
      selgroups = [selgroups group_MAT{g(j)}];
    end             
    
    selcoeff  = unique(selgroups); %these are the coefficients on which to run LS
    
    % 3. run least squares on selcoeff
    lambda = 0;
    Als = Aold(:,selcoeff);
    thetaLS = Als\yold;
%     [thetaLS,~,~,~,~,mses]=SpaRSA(yold,Als,...
%                             lambda,'Debias',0,'StopCriterion',1,'MaxiterA',1000,'ToleranceA',0.0001...
%                                  );
    
    % 4. append this new theta into the estimate, by putting 0s wherever
    % needed
    
    thetaHATgrp = zeros(n,1);
    thetaHATgrp(selcoeff) = thetaLS;
    ttog=cputime-ttog;
%    fprintf('\n time for debiasing %8.4f\n', ttog);
  end
  
end
