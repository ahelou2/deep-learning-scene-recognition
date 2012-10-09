function [x,y] = treeLayout(adj,root,edge_weight)

% Similar to make_layout but specialized for a tree.

if nargin < 2
    root = 1;
end

if nargin < 3
    edge_weight = adj;
end

N = size(adj,1);
level = poset(adj,root)'-1;  

y = (level+1)./(max(level)+2);
y = 1-y;

% neighbors = find(adj(root,:));
% [temp1, sorted_index] = sort(edge_weight(root,neighbors),'descend');
% neighbors = neighbors(sorted_index);
% if(length(neighbors) > 20)
%     neighbors1 = neighbors(1:2:end);
%     neighbors2 = neighbors(2:2:end);
%     y(neighbors1) = y(neighbors1)+0.03;
%     y(neighbors2) = y(neighbors2)-0.03;
% end

x = zeros(size(y));

for i=0:max(level),
  idx = find(level==i);
  if(i<1) 
      x(idx) = 0.5; 
      child_order = (1:length(idx));
  else
      offset=0;
      pidx = find(level==i-1);
      [v, ind] = sort(child_order);
      pidx = pidx(ind);
      child_order = zeros(length(idx),1);
      for j=1:length(pidx)
          [tf,child_nodes] = ismember(find(adj(pidx(j),:)),idx);
          child_nodes = child_nodes(tf);
          % Sort child with edge weights
          child_edge_weights = edge_weight(pidx(j),idx(child_nodes));
          [temp1, edge_weight_order] = sort(child_edge_weights,'descend');
          [temp2, siblings_order] = sort(edge_weight_order, 'ascend');
          child_order(child_nodes) = siblings_order+offset;
          offset = offset + length(child_nodes);
      end      
      x(idx) = child_order./(length(idx)+1);  
      
      if(length(idx)>20)
          [tmp,co_sorted] = sort(child_order,'ascend');
          idx_co = idx(co_sorted);
          y(idx_co(1:2:end)) = y(idx_co(1:2:end))+0.03;
          y(idx_co(2:2:end)) = y(idx_co(2:2:end))-0.03;
      end      
  end
end;

%%%%%%%

function [depth] = poset(adj, root)
% POSET		Identify a partial ordering among the nodes of a graph
% 
%  [DEPTH] = POSET(ADJ,ROOT)
% 
% Inputs :
%    ADJ : Adjacency Matrix
%    ROOT : Node to start with
% 
% Outputs :
%    DEPTH : Depth of the Node
% 
% Usage Example : [depth] = poset(adj,12);
% 
% 
% Note     : All Nodes must be connected
% See also 

% Uses :

% Change History :
% Date		Time		Prog	Note
% 17-Jun-1998	12:01 PM	ATC	Created under MATLAB 5.1.0.421

% ATC = Ali Taylan Cemgil,
% SNN - University of Nijmegen, Department of Medical Physics and Biophysics
% e-mail : cemgil@mbfys.kun.nl 

adj = adj+adj';

N = size(adj,1);
depth = zeros(N,1);
depth(root) = 1;
queue = root;

while 1,
  if isempty(queue),
    if all(depth), break; 
    else
      root = find(depth==0); 
      root = root(1);
      depth(root) = 1;
      queue = root;
    end;
  end;
  r = queue(1); queue(1) = [];
  idx = find(adj(r,:));
  idx2 = find(~depth(idx));
  idx = idx(idx2);
  queue = [queue idx];
  depth(idx) = depth(r)+1;
end;
