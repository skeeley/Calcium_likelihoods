% Demo code illustrating fitting of a GLM under Ganmor Calcium model.
%
% Assumes model hyperparams (tau,alpha, nsevar) are known and just computes
% maximum likelihood estimate for GLM weights.

% Add dirs to path
addpath nlfuns/
addpath xtras/

% Set calcimum model hyperparams
tau = 10; % decay in one time bin
alpha = 50; % gain
signse = 1.5; % stdev of Gaussian noise (in spike train space)
nsevar = signse.^2; % variance of noise
hprs = [tau,alpha,nsevar]'; % model hyperparams

% Set grid of spike counts to consider
maxY = 10; % Max spike count to consider (increase if necessary)
ygrid = 0:maxY; % grid of spike counts to consider

% Set up GLM
nX = 19;  % dimension of stimulus (without DC term)
nXtot = nX+1; % total # of dimensions (with DC)
nT = 50000; % number of time bins
nlfun = @softplus_stable; % set nonlinearity

% Set GLM filter
wfilt = conv2(randn(nX,1),normpdf((1:nX)',nX/2, 2),'same'); % random smooth weights
wDC = -2.5; % DC term
wts = [wDC; 2*wfilt./norm(wfilt)];


%% Generate simulated dataset

% Generate stimulus 
Xmat = [ones(nT,1), randn(nT,nX)]; % stimulus

% Simulate spike response
Xproj = Xmat*wts; % projected stimulus
R = nlfun(Xproj); % conditional intensity
Ysps = poissrnd(R); % spike train
if max(Ysps) > maxY
    fprintf('WARNING: maximum spike count (%d) is larger than maxY (%d)!\n',max(Ysps),maxY);
end

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

% make function handle
lfun = @(prs)neglogli_GLM_GanmorCalciumAR1(prs,Xmat,Yobs,ygrid,hprs,nlfun);

% Set initial weights
wts0 = randn(nXtot,1)*.2;

% % ---  Compute timings ---- 
% tic; negL = lfun(wts0); toc; % Compute timing without gradient
% tic;[negL,grad] = lfun(wts0); toc; % Compute timing with grad
% tic; [negL,grad,H] = lfun(wts0); toc; % Compute timing with grad + Hessian
% 
% % ---- Check analytic gradient & Hessian --- 
% HessCheck(lfun,wts0);

% %% do optimization without numerical gradients
% opts = optimset('display', 'iter');
% fprintf('\n--------------------------------\n');
% fprintf('Optimizing with numerical differencing');
% fprintf('\n--------------------------------\n');
% tic; what1 = fminunc(lfun,wts0,opts); toc;
% 
% %% do optimization with analytic gradient
% opts2 = optimoptions('fminunc','algorithm','quasi-newton','SpecifyObjectiveGradient',true,'display','iter');
% fprintf('\n--------------------------------\n');
% fprintf('Optimizing with gradient only');
% fprintf('\n--------------------------------\n');
% tic; what2 = fminunc(lfun,wts0,opts2); toc;

%% do optimization with gradient + Hessian

opts3=optimoptions('fminunc','algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective','display','iter');
fprintf('Optimizing with gradient + Hessian');
fprintf('\n--------------------------------\n');
tic; what3 = fminunc(lfun,wts0,opts3); toc;

%%  Make plot of fitted weights & compute R^2 of estimated weights
subplot(222);
plot(1:nXtot,wts,'k--',1:nXtot, what3, 'linewidth', lw);
xlabel('coeff #');
ylabel('weight');
title('GLM weights');
legend('true', 'ML','location', 'northwest'); 

Rsqfun = @(w)(1-sum((w-wts).^2)./sum((wts.^2)));
Rsquared = Rsqfun(what3)