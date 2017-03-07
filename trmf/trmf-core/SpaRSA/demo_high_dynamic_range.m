% demo_dyntest_noisefloor.m
%
% adapted from a code by Emmanuel Candès and Justin Romberg,
% September 2008.
%
% sparse spikes of high dynamic range (80dB) over a weak noise floor,
% with partial DCT sensing matrix.
%
%
addpath ./Measurements

n = 65536;
m = 8192;
S = 1024;
Sbig = S/2;
Smid = 0;
Ssmall = S-Sbig-Smid;

% we will have a few spikes at ~0db, a few dynrange db down, and then
% a noise floor  noisefloor db further down
%
noisefloor = 120;
dynrange = 80;
%
xs = zeros(n,1);
xs(1:Sbig) = 1 +.1*rand(Sbig,1);
xs(Sbig+1:S) = 1*10^(-dynrange/20); 
%+ .1*10^(-dynrange/20)*rand(Ssmall,1);
q = randperm(n);
x = xs(q);
x_noisy = x + 10^(-noisefloor/20)*randn(size(x));


% measurement locations
qw = randperm(n);
OMEGA = qw(1:m);
OMEGAC = qw(m+1:n);

% matrix function handles
A = @(z) A_dct(z, OMEGA);
At = @(z) At_dct(z, OMEGA, n);

% measurements
y = A(x_noisy);

% l2 reconstruction
x0 = At(y);

% for 80dB
tau = 5e-6;
% for 75 dB
%tau = 8e-6;
% for 70 dB
%tau = 1e-5;
% for 60 dB
% tau = 5e-5;


% **** Run FPC ****************************************

R = @(trans,mm,nn,xx,inds,OMEGA)  operator4fpc(trans,mm,nn,xx,inds,OMEGA);

% now call fpc
opts=fpc_opts([]);
opts.mxitr=2000;
opts.xtol=1.e-4;

t0 = cputime;
Out = fpc(n,R,y,1/tau,[],opts,OMEGA);
t_fpc = cputime - t0;
x_fpc = Out.x;

% need to scale function value:
obj_fpc = tau*Out.f(end);
iterations_fpc = Out.itr;


% **** SpaRSA test ************************************

% use a duality gap stopping criterion
stopCri=2;
tolA = obj_fpc;   % 1.e-8;
debias = 1;
% parameters for GPSR continuation
% first_tau_factor = 0.8*(max(abs(At(y)))/tau);
% steps = 10;

time_SpaRSA = cputime();
[xg,xg_debias,objective,times,debias_start,mses,totalA_gpsr]=...
    SpaRSA(y,A,tau,...
    'AT',At,...
    'Continuation',1,...
    'Debias',debias,...
    'StopCriterion',2,...
    'ToleranceA',1e-6,...
    'Verbose',0);
time_SpaRSA = cputime() - time_SpaRSA;

time_SALSA = cputime();
[x_S,objective_S,dummy,times_S,mses_S]=...
    SALSA(y,OMEGA,tau,n,...
    'StopCriterion',3,...
    'ToleranceA',objective(end),...
    'Verbose',0);
time_SALSA = cputime() - time_SALSA;

%%%%  PRINTING RESULTS %%%%%%%%%%%%%%%%%%
fprintf(1,'\n ------ RESULTS  ------------------------ \n')
fprintf(1,'SpaRSA; cpu: %6.2f secs (%d iterations)\n',...
        time_SpaRSA,length(objective))
fprintf(1,'final value of the objective function = %6.3e\n',...
          objective(end))
fprintf(1,'number of non-zero estimates = %g\n',sum(xg~=0))
fprintf(1,'number of errors = %g\n\n',sum((x==0)~=(xg==0)))


fprintf(1,'\n ------ RESULTS  ------------------------ \n')
fprintf(1,'FPC; cpu: %6.2f secs (%d iterations)\n',...
        t_fpc,iterations_fpc)
fprintf(1,'final value of the objective function = %6.3e\n',...
          obj_fpc)
fprintf(1,'number of non-zero estimates = %g\n',sum(x_fpc~=0))
fprintf(1,'number of errors = %g\n\n',sum((x==0)~=(x_fpc==0)))


fprintf(1,'\n ------ RESULTS  ------------------------ \n')
fprintf(1,'SALSA; cpu: %6.2f secs (%d iterations)\n',...
        time_SALSA,length(objective_S))
fprintf(1,'final value of the objective function = %6.3e\n',...
          objective_S(end))
fprintf(1,'number of non-zero estimates = %g\n',sum(x_S~=0))
fprintf(1,'number of errors = %g\n\n',sum((x==0)~=(x_S==0)))


% 
figure(1)
scrsz = get(0,'ScreenSize');
set(1,'Position',[10 scrsz(4)*0.1 0.9*scrsz(3) 3*scrsz(4)/4])

subplot(4,1,1)
semilogy(abs(x(1:5000))+10^(-noisefloor/20))
title(sprintf('Abs(original signal) + %g  (showing only first 5000 points)',...
    10^(-noisefloor/20)))
ylim([1.e-8 1.e+2]);

subplot(4,1,2)
semilogy(abs(x_noisy(1:5000)))
title(sprintf('Abs(original signal + noise)  (showing only first 5000 points)'))
ylim([1.e-8 1.e+2]);

subplot(4,1,3)
semilogy(abs(xg(1:5000))+10^(-noisefloor/20))
title(sprintf('Abs(Estimated signal) + %g  (showing only first 5000 points)',...
    10^(-noisefloor/20)))
ylim([1.e-8 1.e+2]);

subplot(4,1,4)
semilogy(abs(xg_debias(1:5000))+10^(-noisefloor/20))
title(sprintf('Abs(Debiased estimated signal) + %g  (showing only first 5000 points)',...
    10^(-noisefloor/20)))
ylim([1.e-8 1.e+2]);

figure(2)
scrsz = get(0,'ScreenSize');
set(1,'Position',[10 scrsz(4)*0.1 0.9*scrsz(3) 3*scrsz(4)/4])

subplot(3,1,1)
semilogy(abs(x(1:5000))+10^(-noisefloor/20))
title(sprintf('Abs(original signal) + %g  (showing only first 5000 points)',...
    10^(-noisefloor/20)))
ylim([1.e-8 1.e+2]);

subplot(3,1,2)
semilogy(abs(x_noisy(1:5000)))
title(sprintf('Abs(original signal + noise)  (showing only first 5000 points)'))
ylim([1.e-8 1.e+2]);

subplot(3,1,3)
semilogy(abs(x_S(1:5000))+10^(-noisefloor/20))
title(sprintf('Abs(Estimated signal) + %g  (showing only first 5000 points)',...
    10^(-noisefloor/20)))
ylim([1.e-8 1.e+2]);




