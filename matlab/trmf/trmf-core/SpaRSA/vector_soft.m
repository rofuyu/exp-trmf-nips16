function y = vector_soft(x,tau)
thenorm = norm(x(:),2);
if thenorm <= tau
   y = zeros(size(x));
else
   y = ((thenorm-tau)/thenorm)*x;
end
