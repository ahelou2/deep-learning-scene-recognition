function [features feat_full_size]= cvis_features4D(model)

[height, width, num_filters numfeat_maps] = size(model.W) ;


W = reshape(model.W, height*width, num_filters, numfeat_maps) ;
[data_dim, ~, ~] = size(W) ;
W_sqr = W.^2 ;
W_sqr_sum = sqrt(sum(W_sqr)) ;
W_sqr_sum = repmat(W_sqr_sum, data_dim,1) ;
features = W./W_sqr_sum ;
features = permute(features, [2 1 3]) ; %num_filters*filter_dim*numfeat_maps

full_size = model.N_V2D ;
padding = (model.N_V2D - model.N_W2D - 1)/2 ;
feat_full_size = reshape(features, [num_filters model.N_W2D numfeat_maps]) ;
%[0 padding 0]
%feat_full_size = padarray(feat_full_size, [0 padding 0]) ;
feat_full_size = repmat(feat_full_size, 2,2) ;
%feat_full_size(num_filters,full_size(1),full_size(2),numfeat_maps) = 0 ;
feat_full_size = reshape(feat_full_size, [num_filters full_size(1)*full_size(2) numfeat_maps]) ;
%figure(1); 
%  size(features)
%  size(features(2,:,1:numfeat_maps))
%  size(squeeze(features(2,:,1:numfeat_maps)))
 
%dispims(squeeze(features(1:num_filters,:,:)'),height,width);
dispims(squeeze(features(2,:,1:numfeat_maps)),height,width);
drawnow