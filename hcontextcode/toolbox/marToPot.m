function [locationPot, edge_index] = marToPot(edges,mu_xij,sigma_xij,mu_xij0,sigma_xij0)

% Convert marginal statistics on a tree to J and h for each edge and node.
%
% PARAMETERS:
%       edges(e,:) = [i,j] where i is a parent of j. edges(1,1) is the root.
%       mu_xij(i,[2*K*(j-1)+1:2*K*j]) = mean of (xi, xj) for i < j when (bi=1,bj=1), K = dim(xi)
%       sigma_xij([2*K*(i-1)+1:2*K*i],[2*K*(j-1)+1:2*K*j]) = covariance of (xi,xj) for i < j when (bi=1,bj=1)
%       mu_xij0(i,[K*(j-1)+1:K*j]) = mean of xi when (bi=1,bj=0)
%                                    mean of xi if i=j
%       sigma_xij0(i,[K*(j-1)+1:K*j]) = covariance of xi when (bi=1,bj=0)
%
% OUTPUTS:
%       locationPot = J and h for each edge and node for different values
%       of bi
%       edge_index(i,j) = e where [i,j] = edges(e,:)
%   
% Myung Jin Choi, MIT, 2009 November

Ne = size(edges,1);
N = size(mu_xij0,1);
K = size(mu_xij0,2)/N; % Dimension of locations
K2 = 2*K;
edge_index = sparse(N,N);

var_mar = zeros(N,K); % Marginal variance and mean for each object.
mean_mar = zeros(N,K);
var_condnp = zeros(N,K);  % Variance and mean when the parent node is not present
mean_condnp = zeros(N,K); 

% The root node does not depend on its parent's presence.
root = edges(1,1);
var_mar(root,:) = sigma_xij0(root,K*(root-1)+1:K*root);
mean_mar(root,:) = mu_xij0(root,K*(root-1)+1:K*root);
var_condnp(root,:) = var_mar(root,:);
mean_condnp(root,:) = mean_mar(root,:);

for e=1:Ne
    p = edges(e,1); % parent
    c = edges(e,2); % child
    edge_index(p,c) = e;
    
    indK_c = K*(c-1)+1:K*c;
    var_mar(c,:) = sigma_xij0(c,indK_c);
    mean_mar(c,:) = mu_xij0(c,indK_c);    
    
    indK_p = K*(p-1)+1:K*p;
    var_condnp(c,:) = sigma_xij0(c,indK_p);
    mean_condnp(c,:) = mu_xij0(c,indK_p);
    
    if(sum(var_condnp(c,:)) <=0 || isnan(var_condnp(c,2))) % c always co-occur with p        
        var_condnp(c,:) = var_mar(c,:);
        mean_condnp(c,:) = mean_mar(c,:);
    end
end

% Assume that each element of the location vector is uncorrelated to each
% other.
locationPot.Jmar = 1./var_mar; % Jmar(xc,:) is J of p(xc | bc = 1)
% We assume each J_xc is diagonal and store it as a row vector
locationPot.hmar = locationPot.Jmar.*mean_mar;
locationPot.Jcondnp = 1./var_condnp;
locationPot.hcondnp = locationPot.Jcondnp.*mean_condnp;


% Assume that the joint covariance of xi and xj has the sparsity pattern:
sppattern = spdiags(ones(K2,3),[-K,0,K],K2,K2);
Jcond = zeros(Ne,2*K);
% Jcond(e,[1:K]) = diagonal J elements of xc|xp,
% Jcond(e,[K+1:2K]) = cross J elements of (xp,xc)
hcond = zeros(Ne,K); % h of xc|xp

for e=1:Ne
    p = edges(e,1);
    c = edges(e,2);
    
    ind2K_p = K2*(p-1)+1:K2*p;
    ind2K_c = K2*(c-1)+1:K2*c;
    
    if (p < c)
        cov_pair = sigma_xij(ind2K_p,ind2K_c).*sppattern;
        mu_pair = mu_xij(p,ind2K_c);
    else
        mu_pair = mu_xij(c,ind2K_p);
        mu_pair = [mu_pair(K+1:end), mu_pair(1:K)]; % child comes later in the vector
        cov_pair = sigma_xij(ind2K_c,ind2K_p).*sppattern;
        cov_pair = [cov_pair(K+1:end,K+1:end), cov_pair(1:K,K+1:end); ...
            cov_pair(1:K,K+1:end), cov_pair(1:K,1:K)];
    end

    % For zero or non positive-definite covariance matrix,
    % make the variables independent.
    if(prod(diag(cov_pair))<=0 || min(eig(full(cov_pair)))<=1e-10)
        cov_pair = diag([sigma_xij0(p,K*(p-1)+1:K*p),sigma_xij0(c,K*(c-1)+1:K*c)]);
        mu_pair = [mu_xij0(p,K*(p-1)+1:K*p),mu_xij0(c,K*(c-1)+1:K*c)];
    end

    Jpc = inv(cov_pair);
    Jcond(e,1:K) = diag(Jpc(K+1:end,K+1:end)); % J of xc|xp (diagonal)
    Jcond(e,K+1:end) = diag(Jpc(1:K,K+1:K2)); % Cross J elements between xp and xc
    hpc = mu_pair*Jpc;
    hcond(e,:) = hpc(K+1:end);    
end

edge_index = edge_index + edge_index';    

locationPot.Jcond = Jcond; 
locationPot.hcond = hcond; 
