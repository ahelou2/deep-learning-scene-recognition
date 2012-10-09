function map_est = maxProductBin(adjmat,node_potential,edge_potential,tree_msg_order)

% Sum-product on a tree with binary variables to compute node and edge marginals.
%
% PARAMETERS:
%       adjmat = adjacency matrix
%       node_potential(i,:) = node potential at node i (may include
%       evidence)
%       edge_potential(2*i-1:2*i,2*j-1:2*j) = edge potential at edge (i,j)
%       tree_msg_order = message order on a tree
%
% OUTPUTS:
%       map_est(i) = MAP estimate at node i
%
% Myung Jin Choi, MIT, 2009 October

adjmat = logical(adjmat);
N = size(node_potential,1);
msg = sparse(N,2*N);  % msg(i,2*j-1:2*j) = message from i to j
states = sparse(N,2*N); % states(i,2*j-1:2*j) = maximizing state of node i for each value of node j
in_msg_prod = zeros(N,2);  % in_msg_prod(i,:) = product of incoming messages except for the current target node

% Upward pass
for n=1:size(tree_msg_order,1)/2
    i = tree_msg_order(n,1);
    j = tree_msg_order(n,2);
    
    neighbors = adjmat(i,:);
    neighbors(j) = 0;
    in_msg_prod(i,:) = prod(msg(neighbors,2*i-1:2*i),1);
    msg_ij = node_potential(i,:).*in_msg_prod(i,:);
    msg_ij = repmat(msg_ij',1,2).*edge_potential(2*i-1:2*i,2*j-1:2*j);
    [msg(i,2*j-1:2*j), states(i,2*j-1:2*j)] = max(msg_ij,[],1);
end

map_est = zeros(N,1);
% Determine the maximizing state at the root
neighbors = adjmat(j,:);
in_msg_prod(j,:) = prod(msg(neighbors,2*j-1:2*j),1);
root_max_marginals = node_potential(j,:).*in_msg_prod(j,:);
[temp, map_est(j)] = max(root_max_marginals);

% Downward pass
for n=size(tree_msg_order,1)/2+1:size(tree_msg_order,1)
    i = tree_msg_order(n,1);
    j = tree_msg_order(n,2);
    if(map_est(i) == 0)
        fprintf('Error in node %d\n',i);
    end
    map_est(j) = states(j,2*(i-1)+map_est(i));
end        
