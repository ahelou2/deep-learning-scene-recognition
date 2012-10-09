function drawHierarchy(adjmat, names, edge_weight, labels, root_names, save_figures)

% Draw a tree graph

if(nargin < 3)
    edge_weight = 0.1*adjmat;
end

if(nargin < 4)
    labels = names;
end

if(nargin < 5)
    root_names = names{1};
end

if(nargin < 6)
    save_figures = 0;
end

[tf, roots] = ismember(root_names, names);
% Draw a separate tree for each subroot
for i=1:length(roots)
    adjmat_n = adjmat;
    root = roots(i);
    neighbors = find(adjmat(root,:));
    
    tree_nodes = [root, neighbors];
    adjmat_n(root,:) = 0;
    adjmat_n(:,root) = 0;    
    
    root_neighbors = intersect(neighbors, roots);
    
    % Disconnect all neighbors of other subroots
    adjmat_n(root_neighbors,:) = 0;
    adjmat_n(:,root_neighbors) = 0;

    while(~isempty(neighbors))
        new_neighbors = find(sum(adjmat_n(neighbors,:),1));
        adjmat_n(neighbors,:) = 0;
        adjmat_n(:,neighbors) = 0;
        neighbors = new_neighbors;
        tree_nodes = union(tree_nodes, neighbors);
    end

    [tf, root_index] = ismember(roots(1), tree_nodes);
    
    [tf, box_index] = ismember(roots, tree_nodes);
    box_index = box_index(find(box_index));

    if(root_index == 0)
        root_index = box_index(1);
    end
    
    node_box = zeros(length(tree_nodes),1);
    node_box(box_index) = 1;
    
    figure;
    [x,y,h] = drawWeightedGraph(adjmat(tree_nodes,tree_nodes),...
        labels(tree_nodes),root_index, edge_weight(tree_nodes, tree_nodes),node_box);
    
    if(save_figures)
        saveas(gcf, ['../figures/tree_',names{root}],'png');        
    end
end


