% Collect binary samples values from the training set

disp('Collecting binary sample values...')
N = length(names);
M = length(Dtraining);
b_samples = ones(N, M); % 1: not present, 2: present
for m=1:M
    
    if(isfield(Dtraining(m).annotation, 'object'))
        objects = Dtraining(m).annotation.object;
    end
    for o = 1:length(objects)
        obj_label = objects(o).name;
        [tf, var_index] =ismember(obj_label, names);
        if (tf)
           b_samples(var_index,m) = 2; 
        end
    end    
end
num_samples = sum(b_samples-1,2);
% % Exclude objects appearing in less than 10 images
% obj_indx = (num_samples>=5);
% b_samples = b_samples(obj_indx,:);
% names = names(obj_indx);

%% Learn a Chow-Liu tree

disp('Learning the prior tree model...')
prob_bij = computeBnStats(b_samples);
mi = computeMutualInformationBin(prob_bij);
adjmat = ChowLiu(mi,names);
prob_bi = diag(prob_bij);
prob_bi = reshape(prob_bi',2,N)';
prob_bi1 = prob_bi(:,2);
prior_adjmat = sparse(adjmat);

tree_msg_order = treeMsgOrder(prior_adjmat, root);
[prior_node_pot, prior_edge_pot] = marToPotBin(prob_bij, tree_msg_order);
Ncategories = N;

% Set edge weights for illustration
edge_weight = adjmat.*mi/max(mi(:));
for i=1:N
    for j=i+1:N
        p01 = prob_bij(2*i-1,2*j);
        p10 = prob_bij(2*i,2*j-1);
        p11 = prob_bij(2*i,2*j);
        if(p11 < (p01+p11)*(p10+p11))  % Negative relationship if p(bi=1,bj=1) < p(bi=1)*p(bj=1)
            edge_weight(i,j) = -edge_weight(i,j);
            edge_weight(j,i) = -edge_weight(j,i);
        end
    end
end





