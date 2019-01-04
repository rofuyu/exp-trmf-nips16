function y = group_l2norm_sjw(x,group_arr)
% "group_start" and "group_len" are vectors indicating the group
% stucture for the group-vector-soft thresholding operation.
% group_start gives the start index of each group in x, group_len
% gives the length of the group. Hence, group i occupies positions
% group_start[i] through group_start[i]+group_len[i]-1 of the x vector.

 
  num_groups = size(group_arr,1);
  xp=[x;0];
  ysums=sum(xp(group_arr).^2,2);
  ysums=sqrt(ysums);
  y1=sum(ysums);
  
% $$$   y = 0;
% $$$   for i=1:num_groups
% $$$     thisgroup=group_start(i):group_start(i)+group_len(i)-1;
% $$$     y = y + norm(x(thisgroup),2);
% $$$   end
% $$$   
% $$$   fprintf(1,' y=%8.3e, y1=%8.3e\n', y, y1);
  
  y=y1;
