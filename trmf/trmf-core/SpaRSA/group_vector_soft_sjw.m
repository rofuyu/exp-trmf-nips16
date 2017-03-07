 function y = group_vector_soft_sjw(x,tau,groups,group_arr)
% "group_start" and "group_len" are vectors indicating the group
% stucture for the group-vector-soft thresholding operation.
% group_start gives the start index of each group in x, group_len
% gives the length of the group. Hence, group i occupies positions
% group_start[i] through group_start[i]+group_len[i]-1 of the x vector.

  num_groups = size(group_arr,1);
  xp=[x;0];
  ysums=sum(xp(group_arr).^2,2);
  ysums=sqrt(ysums);
  ysums=max(ysums-tau,0);
  ysums=ysums./(ysums+tau);
  % fill out into a big vector of multipliers
  ysums=ysums(groups);
  y=ysums.*x; 
  
% $$$   y = zeros(size(x));
% $$$   for i=1:num_groups
% $$$     thisgroup=[group_start(i):group_start(i)+group_len(i)-1];
% $$$     y(thisgroup) = vector_soft(x(thisgroup),tau);
% $$$   end