% test of log-concavity of Gaussian-Poisson mixture model for calcium fluorescence

addpath nlfuns/
addpath xtras/

% Set up model
nX = 19;  % dimension of stimulus
nT = 5000; % number of time bins
nlfun = @softplus_stable; % set nonlinearity

% Set GLM filter
wfilt = conv2(randn(nX,1),normpdf((1:nX)',nX/2, 2),'same');
wts = [-2.5; 2*wfilt./norm(wfilt)];
nW = length(wts); % should be nX+1 due to const

% Generate stimulus and simulate spikes
Xmat = [ones(nT,1), randn(nT,nX)]; % stimulus
Xproj = Xmat*wts; % projected stimulus
R = nlfun(Xproj); % conditional intensity
Ysps = poissrnd(R); % spike train
maxY = 10; % Max spike count to consider
if max(Ysps) > maxY
    fprintf('WARNING: maximum spike count (%d) is larger than maxY (%d)!\n',max(Ysps),maxY);
end

% Set calcimum model hyperparams
tau = 10; % decay in one time bin
alpha = 50; % gain
signse = 0.5; % stdev of Gaussian noise (in spike train space)
nsevar = signse.^2; % variance of noise

% Generate Ca data
SpNoise = signse*randn(nT,1);
Yobs = filter(alpha,[1, -exp(-1/tau)]',Ysps + SpNoise);

% Plot simulated data
subplot(321);  lw = 2; iit = 1:min(100,nT);
plot(iit, R(iit), 'linewidth', lw); box off; 
title('firing rate'); ylabel('sps/s');
subplot(323); 
stem(iit, Ysps(iit), 'k','linewidth', lw); box off; 
title('spike train'); ylabel('spike count');
subplot(325); 
plot(iit, Yobs(iit),'r', 'linewidth', lw); 
title('simulated fluorescence');  box off; 
xlabel('time (bins)'); ylabel('dF/F');

%% Compute log-likelihood

hprs = [tau,alpha,nsevar]'; % model hyperparams
ygrid = 0:maxY; % grid of spike counts to consider

% Compute timing without grad
tic; negL = neglogli_GLM_GanmorCalciumAR1(wts,Xmat,Yobs,ygrid,hprs,nlfun); toc;

% Compute timing with grad
tic;[negL,grad] = neglogli_GLM_GanmorCalciumAR1(wts,Xmat,Yobs,ygrid,hprs,nlfun); toc;

% Compute timing with Hessian
tic;[negL,grad,H]=neglogli_GLM_GanmorCalciumAR1(wts,Xmat,Yobs,ygrid,hprs,nlfun);toc;

% Check analytic gradient & Hessian
lfun = @(prs)neglogli_GLM_GanmorCalciumAR1(prs,Xmat,Yobs,ygrid,hprs,nlfun);
wts0 = randn(nW,1)*.2;
HessCheck(lfun,wts0);

%% do optimization without gradient or Hessian

% fprintf('\n-----------------\n');
% fprintf('Optimizing without gradient\n-----------------\n');
% opts = optimset('display', 'iter');
% tic;
% what1 = fminunc(lfun,wts0,opts);
% toc;

%% do optimization with gradient

% opts2 = optimoptions('fminunc','algorithm','quasi-newton','SpecifyObjectiveGradient',true,'display','iter');
% fprintf('\n-----------------\n');
% fprintf('Optimizing with gradient\n-----------------\n');
% tic;
% what2 = fminunc(lfun,wts0,opts2);
% toc;

%% do optimization with gradient + Hessian

wts0 = randn(nW,1)*2;


opts3 = optimoptions('fminunc','algorithm','trust-region',...
    'SpecifyObjectiveGradient',true,'HessianFcn','objective', ...
    'display','iter');
fprintf('\n-----------------\n');
fprintf('Optimizing with gradient + Hessian\n-----------------\n');
tic;
what3 = fminunc(lfun,wts0,opts3);
toc;

% Make plot of fitted weights
subplot(222);
plot(1:nW,wts,'k--',1:nW, what3, 'linewidth', lw);
xlabel('coeff #');
ylabel('weight');
title('GLM weights');
legend('true', 'ML','location', 'northwest'); 

Rsqfun = @(w)(1-sum((w-wts).^2)./sum((wts.^2)));
Rsquared = Rsqfun(what3)