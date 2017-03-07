
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
        randn(size(f(find(groups==actives(i)))));
end

% measurement matrix
disp('Creating measurement matrix...');
R = randn(k,n);

% orthonormalize rows
R = orth(R')';

disp('Finished creating matrix');

hR = @(x) R*x;
hRt = @(x) R'*x;

% noisy observations
sigma = 0.02;
y = hR(f) + sigma*randn(k,1);

% regularization parameter
tau = 0.1*max(abs(R'*y));

debias = 0;

psi = @(x,tau) group_vector_soft(x,tau,groups);
phi = @(x) group_l2norm(x,groups);

sparsa_tau = tau*3;

[x_SpaRSA,x_debias_SpaRSA,obj_SpaRSA,...
      times_SpaRSA,debias_start_SpaRSA,mse]= ...
    SpaRSA(y,hR,sparsa_tau,...
    'Psi',psi,...
    'Phi',phi,...
    'Monotone',1,...
    'Debias',1,...
    'AT',hRt,... 
    'Initialization',0,...
    'StopCriterion',1,...
    'ToleranceA',0.0001, ...
    'ToleranceD',0.000001, ...
    'MaxiterA',100);
t_SpaRSA = times_SpaRSA(end)


% ================= Plotting results ==========

figure(1)
plot(obj_SpaRSA,'LineWidth',2)
legend('SpaRSA')
set(gca,'FontName','Times','FontSize',16)
xlabel('Iterations')
ylabel('Objective function')
title(sprintf('n=%d, k=%d, tau=%g',n,k,tau))
hold off

figure(2)
plot(times_SpaRSA,obj_SpaRSA,'LineWidth',2)
legend('SpaRSA')
set(gca,'FontName','Times','FontSize',16)
xlabel('CPU time (seconds)')
ylabel('Objective function')
title(sprintf('n=%d, k=%d, tau=%g',n,k,tau))
hold off


% Now we run soft with debiasing
gpsr_tau = tau*2;
[x_BB_mono,x_debias_BB_mono,obj_BB_mono,...
    times_BB_mono,debias_start_BB_mono,mse_BB_mono]= ...
         SpaRSA(y,hR,gpsr_tau,...
         'Debias',1,...
         'AT',hRt,... 
         'True_x',f,...
         'Monotone',1,...
         'Initialization',0,...
         'StopCriterion',3,...
       	 'ToleranceA',0.01,...
         'ToleranceD',0.0001);

% This is figure 1 of the paper.
debias = 1;

figure(5)
scrsz = get(0,'ScreenSize');
set(5,'Position',[10 scrsz(4)*0.1 0.9*scrsz(3)/2 3*scrsz(4)/4])
if debias
    subplot(3,1,1)
else
    subplot(2,1,1)
end
plot(f,'LineWidth',1.1)
top = max(f(:));
bottom = min(f(:));
v = [0 n+1 bottom-0.05*(top-bottom)  top+0.05*((top-bottom))];
set(gca,'FontName','Times')
set(gca,'FontSize',14)
title(sprintf('Original (n = %g, number groups = %g, active groups = %g)',n,n_groups,n_active))
axis(v)

if debias
    subplot(3,1,2)
else
    subplot(2,1,2)
end
plot(x_SpaRSA,'LineWidth',1.1)
set(gca,'FontName','Times')
set(gca,'FontSize',14)
axis(v)
title(sprintf('Block-L2 (k = %g, tau = %5.3g, MSE = %5.3g)',...
    k,sparsa_tau,(1/n)*norm(x_SpaRSA-f)^2))

if debias
    subplot(3,1,3)
    plot(x_debias_BB_mono,'LineWidth',1.1)
    set(gca,'FontName','Times')
    set(gca,'FontSize',14)
    top = max(f(:));
    bottom = min(f(:));
    v = [0 n+1 bottom-0.15*(top-bottom)  top+0.15*((top-bottom))];
    axis(v)
    title(sprintf(...
     'Standard L1 (k = %g, tau = %5.3g, MSE = %0.4g)',...
          k,gpsr_tau,(1/n)*norm(x_debias_BB_mono-f)^2))
end






