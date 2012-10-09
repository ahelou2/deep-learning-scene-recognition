function features = cvis_features(model)

[height, width, num_filters numfeat_maps] = size(model.W) ;

W = reshape(model.W, height*width, num_filters) ;
[data_dim, ~] = size(W) ;
W_sqr = W.^2 ;
W_sqr_sum = sqrt(sum(W_sqr)) ;
W_sqr_sum = repmat(W_sqr_sum, data_dim,1) ;
features = W./W_sqr_sum ;
features = features' ; %num_hid*data_dim

 %figure(1); 
dispims(features(1:num_filters,:)',height,width);
drawnow