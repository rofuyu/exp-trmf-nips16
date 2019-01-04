function y = soft(x,T)
if sum(abs(T(:)))==0
   y = x;
else
   y = max(abs(x) - T, 0);
   y = y./(y+T) .* x;
end


