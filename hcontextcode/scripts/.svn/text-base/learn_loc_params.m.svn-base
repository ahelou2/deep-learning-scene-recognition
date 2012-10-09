% Compute the median and average locations of all the instances for each
% object category in each image

disp('Learning the parameters for the location prior model...')
K = 2;  % dimension of the locaion 
image_size = zeros(M,2);
med_obs_coords = zeros(M,N,K);
avg_obs_coords = zeros(M,N,K);
for m=1:M
    objects = {Dtraining(m).annotation.object.name};
    [foo,obj] = ismember(objects, names); obj = obj';
    
    % find valid objects
    valid = find(obj>0);   
    obj = obj(valid);
    unique_obj = unique(obj);    
    
    image_size(m,1) = Dtraining(m).annotation.imagesize.ncols;
    image_size(m,2) = Dtraining(m).annotation.imagesize.nrows; 
    [loc_index,loc_measurements,loc_img_coords] = getWindowLoc(Dtraining(m).annotation.object(valid),names,image_size(m,:),heights);
    
    for oi=1:length(unique_obj)
        o = unique_obj(oi);
        avg_obs_coords(m,o,:) = mean(loc_measurements(obj==o,:),1);
        med_obs_coords(m,o,:) = median(loc_measurements(obj==o,:),1); 
    end  
end

% Learn parameters for the location variables
[mu_xij, sigma_xij, mu_xij0, sigma_xij0] = computeLocStats(med_obs_coords, b_samples);
% mu_xij(i,[2*K*(j-1)+1:2*K*j]) = mean of (xi, xj) for i < j when (bi=1,bj=1), K = dim(xi)
% sigma_xij([2*K*(i-1)+1:2*K*i],[2*K*(j-1)+1:2*K*j]) = covariance of (xi,xj) when (bi=1,bj=1)
% mu_xij0(i,[K*(j-1)+1:K*j]) = mean of xi when (bi=1,bj=0), mean of xi if i=j
% sigma_xij0([K*(i-1)+1:K*i],[K*(j-1)+1:K*j]) = covariance of xi when
% (bi=1,bj=0)

tree_msg_order = treeMsgOrder(prior_adjmat,root);
prior_edges = tree_msg_order(size(tree_msg_order,1)/2+1:end,:);
[locationPot, edge_index] = marToPot(prior_edges,mu_xij,sigma_xij,mu_xij0,sigma_xij0);   
