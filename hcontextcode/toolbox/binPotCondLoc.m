function [edge_pot] = binPotCondLoc(edge_pot,locPot,edges,map_g)

% Computes edge potentials p(bj | bi, Li, Lj)

K = size(map_g,2);

for e=1:size(edges,1)
    p = edges(e,1);
    c = edges(e,2);
    
    w_1 = probGaussInfo(map_g(c,:),locPot.Jmar(c,:),locPot.hmar(c,:)); % P(bj = 0 | bi, Li, Lj)
    w12 = probGaussInfo(map_g(c,:),locPot.Jcondnp(c,:),locPot.hcondnp(c,:)); % P(bj = 1 | bi=0, Li, Lj)
    
    Jcond = locPot.Jcond(e,1:K);
    hcond = locPot.hcond(e,:) - locPot.Jcond(e,K+1:2*K).*map_g(p,:);
    w22 = probGaussInfo(map_g(c,:),Jcond,hcond); % P(bj=1 | bi=1, Li, Lj)

    epot = edge_pot(2*p-1:2*p,2*c-1:2*c).*[w_1, w12; w_1, w22]/(w_1*2+w12+w22);
    edge_pot(2*p-1:2*p,2*c-1:2*c) = epot ./repmat(sum(epot(:)),2,2);
    edge_pot(2*c-1:2*c,2*p-1:2*p) = edge_pot(2*p-1:2*p,2*c-1:2*c)';    
end


    