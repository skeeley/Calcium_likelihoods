function [f,logf,df,ddf] = exp_stable(x)
% [f,logf,df,ddf] = exp_stable(x)
%
% Replacement for 'exp' that returns 4 arguments: log(exp(x), exp(x), 1st & 2nd deriv

f = exp(x);
logf = x;
if nargout > 2
    df = f;
    ddf = f;
end
