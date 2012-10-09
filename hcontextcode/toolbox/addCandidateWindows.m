function [adjmat, node_pot, edge_pot, pw1lscore] = addCandidateWindows(obj,scores,windowScore, prior_adjmat, prior_node_pot, prior_edge_pot, prob_bi1)

% Add candidate windows as measurements to the context model
%
% PARAMETERS: 
%       obj(k) = object index for window k
%       scores(k) = detector score for window k
%       windowScore = measurement model trained from the training set
%       prior_adjmat = prior adjacency matrix
%       prior_node_pot = prior node potential
%       prior_edge_pot = prior edge potential
%       prob_bi1(i) = P(object i present)
%
% OUTPUTS:
%       adjmat = adjacency matrix of the joint (prior + measurement) model
%       node_pot = node potential
%       edge_pot = edge potential
%       pw1lscore(k) = P(window k correct detection | scores(k))
%
% 2010 April, Myung Jin Choi, MIT.

W = length(scores);
Ncategories = length(windowScore.maxCandWindows);

meas_adjmat = sparse(Ncategories, W);
meas_edge_pot = sparse(2*Ncategories,2*W);
meas_node_pot = zeros(W,2);
pw1lscore = zeros(W,1);
numWindows = zeros(Ncategories,1);

for k=1:W   
    i = obj(k);
    numWindows(i) = numWindows(i)+1;
    meas_adjmat(i,k) = 1;
    pw1lb1 = windowScore.pKthCorrectGivenObjectPresent{i}(numWindows(i)); % P(window k correct | object i present)
    meas_edge_pot(2*i-1:2*i,2*k-1:2*k) = [1 0; 1-pw1lb1, pw1lb1];
    pw1lscore(k) = glmval(windowScore.logitCoef{i}, scores(k), 'logit');
    pCorrect = pw1lb1*prob_bi1(i);
    meas_node_pot(k,:) = [(1-pw1lscore(k))/(1-pCorrect) pw1lscore(k)/pCorrect];
end

% To make the number of candidate windows consistent across images, add
% extra windows and set them to false alarms.
for i=1:Ncategories
    prior_node_pot(i,2) = prior_node_pot(i,2)*prod(1-windowScore.pKthCorrectGivenObjectPresent{i}(numWindows(i)+1:end));
end
    
adjmat = [prior_adjmat meas_adjmat; meas_adjmat' sparse(W,W)];
node_pot = [prior_node_pot; meas_node_pot];
edge_pot = [prior_edge_pot meas_edge_pot; meas_edge_pot' sparse(2*W,2*W)];
node_pot = node_pot./repmat(sum(node_pot,2),1,2);