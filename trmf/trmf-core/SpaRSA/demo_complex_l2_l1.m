% demo_l2_l1 - This demo illustrates  the SpaRSA algorithm in  
% the l2-l1 optimization problem with
%
%     xe = arg min 0.5*||A x-y||^2 + tau ||x||_1
%             x
%  
% which appears, for example, in compressed senseng (CS) 
% applications, with A, x, and y containing *complex* elements

clear all
close all

I=sqrt(-1);

% signal length 
n = 1024;
% observation length 
k = 256;

% number of spikes
n_spikes = 5;
% random +/- 1 signal
x = zeros(n,1);
q = randperm(n/4);
x(q(1:n_spikes)) = sign(randn(n_spikes,1)).* ...
                         exp(I*2*pi*rand(n_spikes,1));
                     
x=[x;conj(x)];

% measurement matrix
disp('Building measurement matrix...');
% define frequencies
W=(1:n)*2*pi/2/n;
T=(1:k)';
R=exp(I*T*W)/sqrt(k) ;

% make [A A*]
R = [R conj(R)];

% % R has been precomputed 
%load CSmatrix
disp('Finished creating matrix');

%TwIST handlers
% Linear operator handlers
hR = @(x) R*x;
hRt = @(x) R'*x;
% define the regularizer and the respective denoising function
% TwIST default
Psi = @(x,th) soft(x,th);   % denoising function
Phi = @(x)    sum(abs(x(:)));     % regularizer

% noise variance
sigma=1e-1;

% observed data; in this case hR(x) is real because of
% the structure of R and x. We use real() to delete any
% spurious imaginary parts thay may have resulted from 
% numerical errors.
y = real(hR(x)) + sigma*randn(k,1);

% regularization parameter 
tau = .2;

% convergence tolerance
tolA = 1e-6;

[x_sparsa,x_debias_sparsa,obj_sparsa,...
      times_sparsa,debias_start_sparsa,mse_sparsa]= ...
    SpaRSA(y,hR,tau,...
    'Debias',0,...
    'AT', hRt, ... 
    'Phi',Phi,...
    'Psi',Psi,...
    'true_x',x,...
    'Monotone',1,...
    'Initialization',1,...
    'StopCriterion',1,...
    'ToleranceA',tolA,...
    'Verbose', 0);

scrsz = get(0,'ScreenSize');
figure(1)
set(1,'Position',[10 scrsz(4)*0.05 scrsz(3)/1.25 0.85*scrsz(4)])

subplot(2,2,1)
plot(times_sparsa,obj_sparsa,'r','LineWidth',2)
st=sprintf('tau = %2.1e, sigma = %2.1e',tau,sigma);
    title(st,'FontName','Times','FontSize',14);
xlabel(sprintf('CPU time (secs.); total iterations %d',length(obj_sparsa)),...
       'FontName','Times','FontSize',14)
ylabel('Objective function','FontSize',14,'FontName','Times')
grid


subplot(2,2,2)
plot(1:2*n,real(x), 'r', 1:2*n, imag(x)+2,'k','LineWidth',1)
axis([1 2*n -1, 3.5]);
legend('Real', 'Imaginary + 2')
title('Original signal','FontName','Times','FontSize',14)
set(gca,'FontName','Times')
set(gca,'FontSize',14)

subplot(2,2,3)
semilogy(times_sparsa,mse_sparsa,'r','LineWidth',2)
st=sprintf('tau = %2.1e, sigma = %2.1e',tau,sigma);
title(st,'FontName','Times','FontSize',14);
xlabel(sprintf('CPU time (secs.); total iterations %d',length(obj_sparsa)),...
       'FontName','Times','FontSize',14)
ylabel('MSE','FontSize',14,'FontName','Times')
grid
set(gca,'FontName','Times')
set(gca,'FontSize',14)

subplot(2,2,4)
plot(1:2*n,real(x_sparsa), 'r', 1:2*n, imag(x_sparsa)+2,'k','LineWidth',1)
axis([1 2*n -1, 3.5]);
legend('Real', 'Imaginary+2')
st=sprintf('Reconstruction: MSE = %2.1e', ...
          ((x_sparsa-x(:))'*(x_sparsa-x(:)))/prod(size(x)));
title(st,'FontName','Times','FontSize',14)
set(gca,'FontName','Times')
set(gca,'FontSize',14)

figure(2)
subplot(3,1,1)
plot(real(hR(x)))
ax = axis;
title('Original signal')

subplot(3,1,2)
plot(y)
axis(ax)
title('Noisy signal')

subplot(3,1,3)
plot(real(hR(x_sparsa)))
axis(ax)
title('Estimated signal')


