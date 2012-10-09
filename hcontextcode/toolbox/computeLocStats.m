function [mu_xij, sigma_xij, mu_xij0, sigma_xij0]...
    = computeLocStats(med_obs_coords, samples)

% Compute the mean and covariance of p(xi,xj | bi=2, bj = 2)
%
% PARAMETERS:
%       med_obs_coords(m,j,:) = median of coordinates of all instances of
%           object i in image m.
%       samples(i,m) = 2 if object i is present in image m, 1 otherwise.
%
% OUTPUTS:
%       mu_xij(i,[2*K*(j-1)+1:2*K*j]) = mean of (xi, xj) for i < j when (bi=1,bj=1), K = dim(xi)
%       sigma_xij([2*K*(i-1)+1:2*K*i],[2*K*(j-1)+1:2*K*j]) = covariance of (xi,xj) when (bi=1,bj=1)
%       mu_xij0(i,[K*(j-1)+1:K*j]) = mean of xi when (bi=1,bj=0)
%                                    mean of xi if i=j
%       sigma_xij0(i,[K*(j-1)+1:K*j]) = covariance of xi when (bi=1,bj=0)
% 
% Myung Jin Choi, MIT, 2009 October

N = size(samples,1); % # variables
K = size(med_obs_coords,3); % dimensions of the location variable
samples = logical(samples-1);
mu_xij = zeros(N,N*2*K);
sigma_xij = zeros(N*2*K,N*2*K);
mu_xij0 = zeros(N,N*K);
sigma_xij0 = zeros(N,N*K);

for i=1:N
    indK_i = K*(i-1)+1:K*i;
    m_bi = samples(i,:);
    xi = squeeze(med_obs_coords(m_bi,i,:));

    mean_xi = mean(xi,1);
    med_xi = median(xi,1);
    mu_xij0(i,indK_i) = med_xi;
    sigma_xij0(i,indK_i) = mean(xi.^2,1) - mean_xi.^2;    
    for j=1:N
        if(j==i)
            continue;
        end
        indK_j = K*(j-1)+1:K*j;
        
        m_bij0 = samples(i,:) & ~samples(j,:);
        xij0 = squeeze(med_obs_coords(m_bij0,i,:));
        
        mean_xij0 = mean(xij0,1);
        med_xij0 = median(xij0,1);
        mu_xij0(i,indK_j) = med_xij0;
        sigma_xij0(i,indK_j) = mean(xij0.^2,1) - mean_xij0.^2;
        
        if (j>i)
            ind2K_i = 2*K*(i-1)+1:2*K*i;
            ind2K_j = 2*K*(j-1)+1:2*K*j;
            m_bij = samples(i,:) & samples(j,:);
            xi = squeeze(med_obs_coords(m_bij,i,:));
            xj = squeeze(med_obs_coords(m_bij,j,:));
            if(size(xi,2)==1)
                xi = xi'; xj = xj';
            elseif(isempty(xi))
                continue;
            end
            mean_xi = mean(xi,1);
            mean_xj = mean(xj,1);
            med_xij = median([xi, xj],1);
            mu_xij(i,ind2K_j) = med_xij;
            xi2 = xi - repmat(mean_xi,size(xi,1),1);
            xj2 = xj - repmat(mean_xj,size(xj,1),1);
            cov_ij = mean(xi2.*xj2,1);
            cov_i = mean(xi2.^2,1);
            cov_j = mean(xj2.^2,1);
            sigma_xij(ind2K_i,ind2K_j) = [diag(cov_i),diag(cov_ij);diag(cov_ij),diag(cov_j)];
        end
    end
end

