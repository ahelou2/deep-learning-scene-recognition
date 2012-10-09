function [J,h] = computeJhPrior(locationPot,b,edges)

% Construct the information matrix J and potential vector h of location
% variables conditioned on b's (presence of each object)
%
% PARAMETERS:
%       locationPot = node and edge potentials for each value of bi
%       b(i) = binary value of object i
%       edges(e,:) = edge (i,j) where i is a parent of j
%
% OUTPUTS:
%       J = information matrix conditioned on b
%       h = potential vector conditioned on b
%
% Myung Jin Choi, MIT, 2009 November

N = length(b);
K = size(locationPot.hmar,2); % dimension of gi
J = sparse(N*K,N*K);
h = zeros(N*K,1);

root = edges(1,1);
indK = K*(root-1)+1:K*root;
J(indK,indK) = diag(locationPot.Jmar(root,:));
h(indK) = locationPot.hmar(root,:)';
for e=1:size(edges,1)
    p = edges(e,1);
    c = edges(e,2);

    indK = K*(c-1)+1:K*c;
    if(b(c)==1)
        J(indK,indK) = J(indK,indK)+diag(locationPot.Jmar(c,:));
        h(indK) = h(indK)+locationPot.hmar(c,:)';       
    elseif(b(p)==1)
        indK = K*(c-1)+1:K*c;
        J(indK,indK) = J(indK,indK)+diag(locationPot.Jcondnp(c,:));
        h(indK) = h(indK)+locationPot.hcondnp(c,:)';
    else % b(c)==2 and b(p)==2
        Jcc = locationPot.Jcond(e,1:K);
        Jpc = locationPot.Jcond(e,K+1:2*K);
        hc = locationPot.hcond(e,:);
        Jpair = [diag(Jpc.^2./Jcc), diag(Jpc); diag(Jpc), diag(Jcc)];
        hpair = [Jpc.*hc./Jcc, hc]';
        indK_pc = [K*(p-1)+1:K*p,K*(c-1)+1:K*c];
        J(indK_pc,indK_pc) = J(indK_pc,indK_pc) + Jpair;
        h(indK_pc) = h(indK_pc) + hpair;       
    end
end
