function [node_pot] = binPotCondLocMeas(node_pot,loc_index,map_g,loc_measurements,detectionWindowLoc)

% Computes p(cik | bi, Li, Wik) and update node potentials for each cik

Ntotal = size(map_g,1);
relativeLoc = loc_measurements - map_g(loc_index,:);

std = 2*sqrt(detectionWindowLoc.varianceCorrect);
prodStd = prod(std,2);

w2 = normpdfYZ(relativeLoc,detectionWindowLoc.meanCorrect(loc_index,:),std(loc_index,:));
% If the window is a correct detection, then p(cik=1 | bi, Li, Wik) is a
% Gaussian distribution with the mean and the variance trained from the
% training set.

w1 = 0.0585*ones(length(loc_index),1)./prodStd(loc_index); 
% If the window is a flase alarm, then p(cik=0 | bi, Li, Wik) is a uniform
% distribution.  The value of the probability 0.0585./prodStd(loc_index) is
% chosen so that if relativeLoc is one standard away from the mean, then
% w1=w2.  (normpdf(1,0,1)^2 = 0.0585)

npot = node_pot(Ntotal+1:end,:).*[w1 w2];
node_pot(Ntotal+1:end,:) = npot./repmat(sum(npot,2),1,2);

function p = normpdfYZ(yz,mean,std)

p = normpdf(yz(:,1),mean(:,1),std(:,1)).*normpdf(yz(:,2),mean(:,2),std(:,2));