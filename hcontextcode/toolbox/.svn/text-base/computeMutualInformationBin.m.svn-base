function [mi_b] = computeMutualInformationBin(prob_bij)

% Compute mutual information from binary samples

tiny = 1e-20;
N = size(prob_bij,1)/2;

entropy_bij = zeros(N,N);
for i=1:N
    for j=i:N
        pbij = prob_bij(2*i-1:2*i,2*j-1:2*j);
        ebij = -pbij(:)'*log(pbij(:)+tiny);
        entropy_bij(i,j) = ebij;
    end
end

entropy_bi = diag(entropy_bij);
mi_b = repmat(entropy_bi',N,1)+repmat(entropy_bi,1,N)-entropy_bij-entropy_bij';
