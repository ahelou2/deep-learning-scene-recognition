function [b, mu, dev] = normalize(a)
[m n] = size(a);
mu = repmat(mean(a), [m 1]);
dev = repmat(std(a), [m 1]);
dev = dev + (dev == 0) ;
b = (a - mu)./dev;