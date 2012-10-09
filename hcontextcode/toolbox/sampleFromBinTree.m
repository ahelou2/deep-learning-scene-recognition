function [smp,logLikelihood] = sampleFromBinTree(adjmat, node_potential,edge_potential,tree_msg_order)

% Sample from a binary tree conditioned on measurements
% Perform sum-product and sample in the downward pass.

adjmat = logical(adjmat);
N = size(node_potential,1);
msg = ones(N,2*N);  % msg(i,2*j-1:2*j) = message from i to j
in_msg_prod = ones(N,2);  % in_msg_prod(i,:) = product of incoming messages except for the current target node
smp = ones(N,1);
logLikelihood = 0;

for n=1:size(tree_msg_order,1)
    i = tree_msg_order(n,1);
    j = tree_msg_order(n,2);
    
    neighbors = adjmat(i,:);
    neighbors(j) = 0;
    in_msg_prod(i,:) = prod(msg(neighbors,2*i-1:2*i),1);
    msg_ij = node_potential(i,:).*in_msg_prod(i,:);
    msg_ij = repmat(msg_ij',1,2).*edge_potential(2*i-1:2*i,2*j-1:2*j);
    if(sum(msg_ij(:)) > 0)
        msg(i,2*j-1:2*j) = sum(msg_ij,1)/sum(msg_ij(:));
    end
    
   
    if (n>=N) % Sample a child node conditioned on the parent node        
        emar = in_msg_prod(i,:)'*in_msg_prod(j,:);  % in_msg_prod(j) has product of messages from its children
        emar = emar.*edge_potential(2*i-1:2*i,2*j-1:2*j);
        emar = emar.*(node_potential(i,:)'*node_potential(j,:)); % edge marginal of (i,j)
        if(n==N) % Sample from the root
            nmar = sum(emar,2);
            nmar = nmar/sum(nmar);
            smp(i) = (rand > nmar(1))+1;
            logLikelihood = logLikelihood + log(nmar(smp(i)));
        end        
        cond_prob = emar(smp(i),:);
        cond_prob = cond_prob/sum(cond_prob);
        smp(j) = (rand > cond_prob(1))+1;
        logLikelihood = logLikelihood + log(cond_prob(smp(j)));
    end
end        
