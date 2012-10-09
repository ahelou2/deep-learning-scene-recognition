clear all

configure

disp('loading groundtruth labels of the training set...')
load(groundTruth, 'Dtraining');   
load(objectCategories) % Load object names and heights and the index of the root node     

%% Learn binary prior Model

% Learn the strucure of the tree and parameters for the binary variables
learn_binary_prior

% Show the structure of the tree
subtrees = {'sky','floor','wall','building','tree','sink'}; 
% drawHierarchy(prior_adjmat, names, edge_weight, names, subtrees,0);


%% Learn location prior model

learn_loc_params

save(priorModel,'prior_adjmat','prior_node_pot', 'prior_edge_pot','prob_bi1',...
    'root','Ncategories','edge_weight','subtrees','locationPot','K','prior_edges');

%% Save an independent model in a tree format

prob_bi = diag(prob_bij);
prior_node_pot = prob_bi;
prior_node_pot = reshape(prior_node_pot,2,Ncategories)';
edges = tree_msg_order(size(tree_msg_order,1)/2+1:end,:);
prior_edge_pot = sparse(2*N,2*N);
for e=1:size(edges,1)
    i = edges(e,1);
    j = edges(e,2);
    prior_edge_pot(2*i-1:2*i,2*j-1:2*j) = [1 1; 1 1];
end
prior_edge_pot = prior_edge_pot + prior_edge_pot';
[locationPot, edge_index] = marToPotIndep(prior_edges,mu_xij0,sigma_xij0);

save(priorModelIndep,'prior_adjmat','prior_node_pot','prior_edge_pot','prob_bi1',...
    'root','Ncategories','locationPot','K','prior_edges');


%% Learn parameters for the measurement model

disp('loading detector outputs...')
load(detectorOutputs,'DdetectorTraining','logitCoef','validcategories','MaxNumDetections') % Load detector outputs on the training set

learn_binary_measurement_model

% Save 'DdetectorTraining' again if it does not contain a field showing 
% whether each window is a correct detection or not.
if(~isfield(DdetectorTraining(1).annotation.object(1),'detection'))
    save(detectorOutputs,'DdetectorTraining','-APPEND')
end

learn_location_measurement_model

save(measurementModel,'windowScore','detectionWindowLoc')

%% Train gist

disp('loading groundtruth labels of the test set...')
load(groundTruth, 'Dtest')     

train_gist

save(gistPredictions,'p_b_gist_test','class_training')
