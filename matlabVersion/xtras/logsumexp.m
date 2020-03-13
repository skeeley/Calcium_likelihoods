function lse = logsumexp(b,idx)
% lse = logsumexp(b,idx)
% 
% Computes log(sum(x,idx)) in a stable way
% 
% Default is idx = 2 (columns) if unspecified

if nargin < 2
    idx = 2;
end

Bmax = max(b,[],idx); % get max along this index
lse = log(sum(exp(b-Bmax),idx))+Bmax; % compute log-sum-exp
