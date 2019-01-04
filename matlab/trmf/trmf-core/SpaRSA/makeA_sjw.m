function [A_tilde, groups, group_arr] = makeA_sjw(A,group_MAT,lam)

% Function to form 'A' that will be input to SpaRSA, and associated group
% structure that will be used for the group L2 norm regularizer
  
% INPUTS:
% A         = Original CS(or Blurring) matrix (k X n)
% group_Mat = cell array having rows as group
% lam       = parameter that determines what strategy we are using. 
% lam = 0 ==> replication.
% lam > 0 ==> no replication, but force "replicates" to be the same
  
% OUTPUTS:
% A_tilde   = depending on the strategy, this is either the replicated
% columns of the augmented matrix as explained in the notes.
% group_start = indices of x at which each group starts
% group_len = number of indices in each group.
  
  [k,n] = size(A);
  [l m] = size(group_MAT);
  
  % REPLICATION STEP
  idx = 1:l;
  
  if lam>0
    force = true;
  else
    force = false;
  end
  
  groups = [];
  A_tilde = [];
  group_start = zeros(l,1); group_len = zeros(l,1);
  group_start(1)=1;
  % groups replication, and A_tilde if replication used (by replicating columns of A)
  for i = 1:l
    
    subg = repmat(i,1,length(group_MAT{i}));
    if ~force
      %replicate columns of A if forcing not used
      subc = A(:,group_MAT{i});
      A_tilde = [A_tilde subc];
    end
    groups = [groups subg];
    if (i>1) 
      group_start(i) = group_start(i-1) + group_len(i-1);
    end
    group_len(i)=length(group_MAT{i});
    
  end
  
  % construct group_arr structure, for which row i consists of the
  % indices in group i, padded out with "n+1", which points to a
  % dummy index in x
  group_arr=(length(groups)+1)*ones(l,max(group_len));
  for i=1:l
    group_arr(i,1:group_len(i))=[group_start(i):group_start(i)+group_len(i)-1];
  end
    
  % if forcing is used, we form A_tilde here (Refer to notes1.pdf for the structure of A_tilde)
  if force
    j = length(groups);                
    H = sparse(j,n);    
    J = sparse(j,j);
    for i = 1:max(groups)
      ind = find(groups == i);
      H(ind,i) = 1;
      J(ind,ind) = -1;
    end
    A_tilde = [A sparse(k,j);lam*H lam*J];   
  end
  
  % if forcing is used, we need to make a "dummy" group to account for the
  % group and group_arr matrices
  if force
    vect = i+1*ones(1,n+j-length(groups));
    groups = [groups vect];
    dummy = size(H,1)+1;
    fingrp = dummy+1:length(groups);
    dummy = length(groups)+1;
    M = size(group_arr,1);
    N = size(fingrp,2) - 2;
    group_arr = [group_arr dummy*ones(M,N)];
    group_arr = [group_arr; fingrp];
  end
      
  
end
  