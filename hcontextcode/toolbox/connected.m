function tf = connected(adj,i,j)

% Returns true of i and j are connected on the graph.

N = size(adj,1);
neighbors = find(adj(i,:));
adj(i,:) = 0;
adj(:,i) = 0;
tf = false;

while(~isempty(neighbors))
    if(ismember(j, neighbors))
        tf = true;
        break;
    end
    new_neighbors = find(sum(adj(neighbors,:),1));
    adj(neighbors,:) = 0;
    adj(:,neighbors) = 0;
    neighbors = new_neighbors;
end
