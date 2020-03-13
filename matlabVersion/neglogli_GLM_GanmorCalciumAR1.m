function [negL,grad,H] = neglogli_GLM_GanmorCalciumAR1(prs,X,Y,ygrid,hprs,nlfun)
% negL = neglogli_GLM_GanmorCalciumAR1(prs,X,Y,sig,maxcount)
%
% Computes negative log-likelihood for a GLM with likelihood given by the
% Ganmor AR1 mixture model for calcium imaging data
%
% Input:
% -------
%     prs [d  x 1] - vector of GLM regression weights
%       X [nT x d] - design matrix 
%       Y [nT x 1] - calcium fluorescence observations
%   ygrid [1 x nY] - row vector of spike counts to consider
%    hprs [3 x 1] - moadel hyperparameter: logtau, logalpha, lognsevar
%   nlfun [funct] - function handle for GLM nonlinearity
%
% Output:
% -------
%   negL [1 x 1] - negative log-likelihood 
%   grad [d x 1] - gradient
%      H [d x d] - Hessian

% unpack hyperparams
tau = hprs(1); % decay time constant
alpha = hprs(2); % scale factor (photons per spike)
nsevar = hprs(3); % noise variance (in spike count units)

% compute AR(1) diffs
taudecay = exp(-1/tau); % decay factor for one time bin
Ydff = (Y(2:end)-taudecay*Y(1:end-1))/alpha;  

% Grid of Gaussian log-likelihood terms (for each observation and latent spike count)
logPnse = -(Ydff-ygrid).^2/(2*nsevar)-.5*log(2*pi*nsevar); 

% Compute linear filter output and spike-count contribution
Xproj = X(2:end,:)*prs; % linear predictor
poissConst = gammaln(ygrid+1);  % constant from Poisson dist

% Compute the 
switch nargout

    case {0,1} % --- Compute neglogli -----------------------
        
        [f,logf] = nlfun(Xproj); % conditional intensity and its log
        logPcounts = ygrid.*logf - f - poissConst;  % Poisson loglis [nT x nY]
        
        % compute log-likelihood for each time bin
        logli = logsumexp(logPnse+logPcounts,2);
        
        % compute negative log-likelihood
        negL = -sum(logli);
        
        
    case 2 % ---  Compute neglogli & Gradient ----------------
        
        [f,logf,df] = nlfun(Xproj); % cond. intensity, log, and 1st deriv
        logPcounts = ygrid.*logf - f - poissConst; % Poisson loglis [nT x nY]
        
        % compute log-likelihood for each time bin
        logjoint = logPnse+logPcounts;
        logli = logsumexp(logjoint,2); % log-li for each time bin
        negL = -sum(logli); % negative logli
        
        % gradient
        dLpoiss = (ygrid.*df./f)-df; % wts for Poisson log-li gradient
        gwts = sum(exp(logjoint-logli).*dLpoiss,2); % gradientt weights
        grad = -X(2:end,:)'*gwts;
        
    case 3 % --- Compute neglogli, Gradient & Hessian --------

        [f,logf,df,ddf] = nlfun(Xproj);  % cond. intensity, log, & derivs
        logPcounts = ygrid.*logf - f - poissConst; % Poisson loglis [nT x nY]
        
        % compute log-likelihood for each time bin
        logjoint = logPnse+logPcounts;
        logli = logsumexp(logjoint,2); % log-li for each time bin
        negL = -sum(logli); % negative logli
        
        % gradient
        dLpoiss = (ygrid.*df./f)-df; % deriv of Poisson log-li 
        gwts = sum(exp(logjoint-logli).*dLpoiss,2); % gradientt weights
        grad = -X(2:end,:)'*gwts;
        
        % Hessian
        ddLpoiss = ygrid.*(ddf./f-(df./f).^2)-ddf; % 2nd deriv Poisson log-li
        ddL = (ddLpoiss+dLpoiss.^2);
        hwts = sum(exp(logjoint-logli).*ddL,2) - gwts.^2; % gradientt weights
        
        H = -X(2:end,:)'*(X(2:end,:).*hwts); % Hessian
        %H = -X(2:end,:)'*(bsxfun(@times,X(2:end,:),hwts); % OLD (bsxfun)

end


