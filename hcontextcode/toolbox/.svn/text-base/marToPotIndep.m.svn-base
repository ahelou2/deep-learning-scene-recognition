function [locationPot, edge_index] = marToPotIndep(edges,mu_xij0,sigma_xij0)

% Convert marginal statistics on a tree to J and h for each edge and node
% for an independent model in which all objects are independent.  An
% independent model can be encoded in a much simpler way, but we use this
% parameterization just to be consistent with a tree model.
%
% PARAMETERS:
%       edges(e,:) = [i,j] where i is a parent of j. edges(1,1) is the root.
%       mu_xij0(i,[K*(j-1)+1:K*j]) = mean of xi when (bi=1,bj=0)
%                                    mean of xi if i=j
%       sigma_xij0(i,[K*(j-1)+1:K*j]) = covariance of xi when (bi=1,bj=0)
%
% OUTPUTS:
%       locationPot = J and h for each edge and node for different values
%       of bi, which is same in this independent model.
%       edge_index(i,j) = e where [i,j] = edges(e,:)
%   
% Myung Jin Choi, MIT, 2009 November

Ne = size(edges,1);
N = size(mu_xij0,1);
K = size(mu_xij0,2)/N; % Dimension of locations
edge_index = sparse(N,N);

var_mar = zeros(N,K);  % Variance and mean of each node.
mean_mar = zeros(N,K); 

for i=1:N
    indK_i = K*(i-1)+1:K*i;
    var_mar(i,:) = sigma_xij0(i,indK_i);
    mean_mar(i,:) = mu_xij0(i,indK_i);
end

% Assume that each element of the location vector is uncorrelated to each
% other.
Jmar = 1./var_mar; % Jmar(xc,:) is J of p(xc | bc = 1)
% We assume each J_xc is diagonal and store it as a row vector
hmar = Jmar.*mean_mar;

locationPot.Jmar = Jmar; 
locationPot.hmar = hmar;
locationPot.Jcondnp = locationPot.Jmar;
locationPot.hcondnp = locationPot.hmar;

Jcond = zeros(Ne,2*K);
% Jcond(e,[1:K]) = diagonal J elements of xc|xp,
% Jcond(e,[K+1:2K]) = cross J elements of (xp,xc)
hcond = zeros(Ne,K); % h of xc|xp

for e=1:Ne
    p = edges(e,1);
    c = edges(e,2);
    edge_index(p,c) = e;
    
    Jcond(e,1:K) = Jmar(c,:);
    Jcond(e,K+1:end) = zeros(1,K);
    hcond(e,:) = hmar(c,:);
end

edge_index = edge_index + edge_index';    

locationPot.Jcond = Jcond; 
locationPot.hcond = hcond; 
