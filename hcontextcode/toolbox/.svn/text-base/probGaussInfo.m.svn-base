function [prob, mu] = probGaussInfo(x,J,h)

K = length(x);
if(length(J)==K) 
    mu = h./J;
    x_mu = x - mu;
    prob = exp(-0.5*x_mu.^2*J')*sqrt(prod(J));
else
    K = K/2;
    detJ = J(1:K).*J(K+1:2*K)-J(2*K+1:end).^2;  % det([a b; b d]) = ad - b^2
    mu = [J(K+1:2*K),J(1:K)].*h-repmat(J(2*K+1:3*K),1,2).*[h(K+1:2*K),h(1:K)];
    mu = mu./repmat(detJ,1,2);
    x_mu = x - mu;
    prob = exp(-0.5*x_mu.^2*J(1:2*K)' - (x_mu(1:K).*x_mu(K+1:2*K))*J(2*K+1:end)');
    prob = prob*sqrt(prod(detJ));
end