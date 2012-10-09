function [mi,mi_b] = computeMutualInformation(prob_bij, sigma_xij, sigma_xij0)

% Compute mutual information from binary and Gaussian samples

tiny = 1e-20;
N = size(prob_bij,1)/2;

% Compute the entropy of Gaussian variables conditioned on binary variables
entropy_xij0 = zeros(N,N);  % H( P(xi,xj | bi=1,bj=0) )  and H( P(xi | bi=1) )
K = size(sigma_xij0,1)/N;
for i=1:N
    for j=1:N
        index_j = K*(j-1)+1:K*j;
        entropy_xij0(i,j) = sum(log(sigma_xij0(i,index_j)+tiny));
    end
end
entropy_xij0 = 0.5*(entropy_xij0 + K*(log(2*pi)+1));

entropy_xij = zeros(N,N);  % H( P(xi,xj | bi=1, bj=1) )
K2 = 2*K;
sppattern = spdiags(ones(K2,K),[-K,0,K],K2,K2);
%sppattern = [eye(K) ones(K); ones(K) eye(K)];
for i=1:N
    index_i = K2*(i-1)+1:K2*i;
    for j=i+1:N
        index_j = K2*(j-1)+1:K2*j;
        entropy_xij(i,j) = log(det(sigma_xij(index_i,index_j).*sppattern)+tiny);
        %entropy_xij(i,j) = log(det(sigma_xij(index_i,index_j))+tiny);
        if(~isreal(entropy_xij(i,j)) || ~isfinite(entropy_xij(i,j)))
            entropy_xij(i,j) = 0;
        end
    end
end
entropy_xij = 0.5*(entropy_xij+K2*(log(2*pi)+1));

% Compute the entropy of binary variables
entropy_bij = zeros(N,N);
for i=1:N
    for j=i:N
        pbij = prob_bij(2*i-1:2*i,2*j-1:2*j);
        ebij = -pbij(:)'*log(pbij(:)+tiny);
        entropy_bij(i,j) = ebij;
    end
end


% Compute the entropy of zi=(bi,xi)
p_bi_1 = diag(prob_bij);
p_bi_1 = p_bi_1(2:2:end);
entropy = diag(entropy_xij0).*p_bi_1 + diag(entropy_bij);  % H( P(bi,xi) )

% Compute the entropy of zi=(bi,xi) and zj=(bj,xj)
joint_entropy = zeros(N,N);
for i=1:N
    for j=i+1:N
        jent = entropy_bij(i,j);
        if (prob_bij(2*i,2*j) > 0 && isfinite(entropy_xij(i,j)))
            jent = jent + entropy_xij(i,j)*prob_bij(2*i,2*j);
        end
        if (prob_bij(2*i,2*j-1) > 0 && isfinite(entropy_xij0(i,j)))
            jent = jent + entropy_xij0(i,j)*prob_bij(2*i,2*j-1);
        end
        if (prob_bij(2*i-1,2*j) > 0 && isfinite(entropy_xij0(j,i)))
            jent = jent + entropy_xij0(j,i)*prob_bij(2*i-1,2*j);
        end
        joint_entropy(i,j) = jent;        
    end
end

%joint_entropy(isnan(joint_entropy)) = 0;
%joint_entropy(isinf(joint_entropy)) = 0;

entropy_bi = diag(entropy_bij);
mi_b = repmat(entropy_bi',N,1)+repmat(entropy_bi,1,N)-entropy_bij-entropy_bij';

joint_entropy = joint_entropy + joint_entropy';
mi = repmat(entropy',N,1)+repmat(entropy,1,N)-joint_entropy;  % mi(i,j) = I( bi,xi; bj,xj)
