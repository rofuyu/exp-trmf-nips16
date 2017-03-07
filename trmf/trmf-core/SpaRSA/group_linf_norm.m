function y = group_l2norm(x,groups)
% "groups" is vector indicating the group stucture
% for the group-vector-soft thresholding operation
% The size of groups must be the same as x.
% Each set of elements that are mutually equal 
% define a group. 
% Example: if groups = [1 1 1 2 2 2 3 3 2 2 1 1 3 3],
% there are 3 groups and 
% x[1,2,3,11,12] are in group 1,
% x[4,5,6,9,10] are in group 2,
% x[7,8,13,14] are in group 3.
% All elements of groups must be integers in the range 1...num_groups
num_groups = max(groups);
if min(groups)~=1
   error(['Wrong group structure vector'])
end
y = 0;
for i=1:num_groups
    thisgroup = find(groups == i);
    y = y + norm(x(thisgroup),inf);
end


