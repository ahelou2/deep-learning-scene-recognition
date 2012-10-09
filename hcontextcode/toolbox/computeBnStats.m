function [prob_bij] = computeBnStats(samples)

% Compute joint probability tables from binary samples.
%
% PARAMETER:
%       samples(m,n) = 2 if image m contains object n, 1 otherwise.
%
% OUTPUT:
%       prob_bij(2*(i-1)+s1, 2*(j-1)+s2) = P(i=s1,j=s2);
%
% Myung Jin Choi, MIT, 2009 October

N = size(samples,1);
M = size(samples,2);

prob_bij = zeros(2*N, 2*N);
for i=1:N
    for j=1:i-1
        sample_pairs = 2*samples(i,:) + samples(j,:);
        
        p00 = length(find(sample_pairs==3)); % P(bi=0,bj=0)
        p01 = length(find(sample_pairs==4)); % P(bi=0,bj=1)
        p10 = length(find(sample_pairs==5)); % P(bi=1,bj=0)
        p11 = length(find(sample_pairs==6)); % P(bi=1,bj=1)
        
        prob_bij([2*(i-1)+1,2*(i-1)+2],[2*(j-1)+1,2*(j-1)+2]) = [p00 p01; p10 p11];
        
        if((abs(p00+p01+p10+p11) - M) > 0)
            fprintf('%d %d\n', i,j);
        end
    end
end

prob_bij = prob_bij + prob_bij';

for i=1:N
    p00 = length(find(samples(i,:)==1)); % P(bi=0)
    p11 = length(find(samples(i,:)==2)); % P(bi=1)
    
    prob_bij([2*(i-1)+1,2*(i-1)+2],[2*(i-1)+1,2*(i-1)+2]) = [p00 0; 0 p11];
end

prob_bij = prob_bij / M;