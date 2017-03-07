
% n is the original signal length
n = 2^12;

% k is number of observations to make
k = 2^10;

% number of groups with activity 
n_active = 8;

n_groups = 64;
size_groups = n / n_groups;

raux = randperm(n_groups);
actives = raux(1:n_active);

groups = ceil([1:n]'/size_groups);

f = zeros(n,1);

for i=1:n_active
    f(find(groups==actives(i))) = ...
        ones(size(f(find(groups==actives(i)))));
end

% measurement matrix
disp('Creating measurement matrix...');
R = randn(k,n);
% orthonormalize rows
R = orth(R')';
%
disp('Finished creating matrix');

hR = @(x) R*x;
hRt = @(x) R'*x;

% noisy observations
sigma = 0.01;
y = hR(f) + sigma*randn(k,1);

% regularization parameter
tau = 0.1*max(abs(R'*y));

debias = 0;

psi = @(x,tau) group_L2_Linf_shrink(x,tau,groups);
phi = @(x)     group_linf_norm(x,groups);

sparsa_tau = tau*5;
[x_SpaRSA,x_debias_SpaRSA,obj_SpaRSA,...
      times_SpaRSA,debias_start_SpaRSA,mse]= ...
    SpaRSA(y,hR,sparsa_tau,...
    'Psi',psi,...
    'Phi',phi,...
    'Monotone',1,...
    'Debias',0,...
    'AT',hRt,... 
    'Initialization',0,...
    'StopCriterion',1,...
    'ToleranceA',0.001, ...
    'MaxiterA',1000);

[x_SpaRSA2,x_debias_SpaRSA,obj_SpaRSA2,...
      times_SpaRSA2,debias_start_SpaRSA,mse]= ...
    SpaRSA(y,hR,sparsa_tau,...
    'Psi',psi,...
    'Phi',phi,...
    'Monotone',0,...
    'Debias',0,...
    'AT',hRt,... 
    'Initialization',0,...
    'StopCriterion',1,...
    'ToleranceA',0.001, ...
    'MaxiterA',1000);

% ================= Plotting results ==========

figure(1)
plot(obj_SpaRSA,'LineWidth',2)
hold on
plot(obj_SpaRSA2,'g--','LineWidth',2)
hold off
legend('SpaRSA monotone','SpaRSA non-monotone')
set(gca,'FontName','Times','FontSize',16)
xlabel('Iterations')
ylabel('Objective function')
title(sprintf('n=%d, k=%d, tau=%g',n,k,tau))
hold off

figure(2)
plot(times_SpaRSA,obj_SpaRSA,'LineWidth',2)
hold on
plot(times_SpaRSA2,obj_SpaRSA2,'g--','LineWidth',2)
hold off
legend('SpaRSA monotone','SpaRSA non-monotone')
set(gca,'FontName','Times','FontSize',16)
xlabel('CPU time (seconds)')
ylabel('Objective function')
title(sprintf('n=%d, k=%d, tau=%g',n,k,tau))
hold off


% Now try goup l2 norm on this problem
psi = @(x,tau) group_vector_soft(x,tau,groups);
phi = @(x)     group_l2norm(x,groups);

sparsa_tau = tau*5;
[x_SpaRSA_l2,x_debias_SpaRSA_l2,obj_SpaRSA_l2,...
      times_SpaRSA_l2,debias_start_SpaRSA,mse]= ...
    SpaRSA(y,hR,sparsa_tau,...
    'Psi',psi,...
    'Phi',phi,...
    'Monotone',1,...
    'Debias',0,...
    'AT',hRt,... 
    'Initialization',0,...
    'StopCriterion',1,...
    'ToleranceA',0.0001, ...
    'MaxiterA',100);

figure(5)
scrsz = get(0,'ScreenSize');
set(5,'Position',[10 scrsz(4)*0.1 0.9*scrsz(3)/2 3*scrsz(4)/4])

subplot(3,1,1)
plot(f,'LineWidth',1.1)
top = max(f(:));
bottom = min(f(:));
v = [0 n+1 bottom-0.05*(top-bottom)  top+0.05*((top-bottom))];
set(gca,'FontName','Times')
set(gca,'FontSize',14)
title(sprintf('Original (n = %g, number groups = %g, active groups = %g)',n,n_groups,n_active))
axis(v)


subplot(3,1,2)
plot(x_SpaRSA,'LineWidth',1.1)
set(gca,'FontName','Times')
set(gca,'FontSize',14)
axis(v)
title(sprintf('Group-L-infinity (k = %g, tau =%4.2g, MSE = %5.3e)',...
    k,sparsa_tau,(1/n)*norm(x_SpaRSA-f)^2))


subplot(3,1,3)
plot(x_SpaRSA_l2,'LineWidth',1.1)
set(gca,'FontName','Times')
set(gca,'FontSize',14)
top = max(f(:));
bottom = min(f(:));
v = [0 n+1 bottom-0.15*(top-bottom)  top+0.15*((top-bottom))];
axis(v)
title(sprintf(...
 'Group-L2  (k = %g, tau = %5.3g, MSE = %6.4e)',...
      k,sparsa_tau,(1/n)*norm(x_SpaRSA_l2-f)^2))

  


