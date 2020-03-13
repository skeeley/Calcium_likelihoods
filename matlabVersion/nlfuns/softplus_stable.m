function [f,logf,df,ddf] = softplus_stable(x)
% [f,logf,df,ddf] = softplus_stable(x)
%
% Computes the function:
%    f(x) = log(1+exp(x))
% and returns its value, log, and 1st and 2nd derivatives

f = log(1+exp(x));
logf = log(f);

if nargout > 2
    df = exp(x)./(1+exp(x));
end
if nargout > 3
    ddf = exp(x)./(1+exp(x)).^2;
end

% Check for small values to avoid underflow errors
if any(x<-30)
    iix = (x<-30);
    logf(iix) = x(iix);
    f(iix) = exp(x(iix));
    df(iix) = f(iix);
    ddf(iix) = f(iix);
end

% Check for large values to avoid overflow errors
if any(x>30)
    iix = (x>30);
    logf(iix) = log(x(iix));
    f(iix) = x(iix);
    df(iix) = 1;
    ddf(iix) = 0;
end