% function features = vis_features(W, height, width)
% 
% %INPUT: Learned weight matrix (data_dim*num_hid).
% %OUTPUT: features (num_hid*data_dim) which maximally activate each neuron
% % NOTE: To determine which image maximally activates a unit, we used the
% % formula in the autoencoder lecture p.17. There each unit was stochastic
% % computing the sigmoiding function on input that could be continuous.
% 
% %TODO: Modify function so it can visualize features for units belonger to
% %any layer.
% 
% [data_dim, num_hid] = size(W) ;
% W_sqr = W.^2 ;
% W_sqr_sum = sqrt(sum(W_sqr)) ;
% W_sqr_sum = repmat(W_sqr_sum, data_dim,1) ;
% features = W./W_sqr_sum ;
% features = features' ; %num_hid*data_dim
% 
%  figure(1); 
% dispims(features(1:20,:)',height,width);
% drawnow

function features = vis_features(model, hid_layer, height, width, varargin)

%INPUT: Learned weight matrix (data_dim*num_hid).
%       hid_layer is the hidden layer you are interested in seeing the
%       learned features from.
%OUTPUT: features (num_hid*data_dim) which maximally activate each neuron
% NOTE: To determine which image maximally activates a unit, we used the
% formula in the autoencoder lecture p.17. There each unit was stochastic
% computing the sigmoiding function on input that could be continuous.

%TODO: Modify function so it can visualize features for units belonger to
%any layer.

%Process options
%if args are just passed through in calls they become cells
if (isstruct(varargin)) 
    args= prepareArgs(varargin{1});
else
    args= prepareArgs(varargin);
end
[   pca        ...
    ] = process_options(args    , ...
    'pca'             ,  []);

num_hid = length(model) ;

W = model{hid_layer}.W;

[data_dim, num_hid] = size(W);
W_sqr = W.^2;
W_sqr_sum = sqrt(sum(W_sqr));
W_sqr_sum = repmat(W_sqr_sum, data_dim,1);
features = W./W_sqr_sum;
features = features'; %num_hid*data_dim

sampled_data = features;
[numcases numdims] = size(sampled_data);
if (hid_layer > 1)
    if model{hid_layer}.type == 'BB'
        sampled_data = sampled_data > rand(numcases, numdims);
    end
    for i=hid_layer-1:-1:1 
        phstates = sampled_data;
        [numcases numdims] = size(sampled_data);
        W = model{i}.W;
        c = model{i}.c;
        b = model{i}.b;
        numhid = size(W, 2);
        if model{i}.type == 'BB'
            sampled_data = logistic(phstates*W' + repmat(c,numcases,1));
            % WARNING: In Ruslan's science rbm.m he doesn't sample but in his
            % DBM rbm.m he samples!
            sampled_data = sampled_data > rand(numcases, numdims);
        elseif model{i}.type == 'CB'
            sampled_data = phstates*W' + repmat(c,numcases,1); 
        end
    end
end

if (length(pca) > 0)
    sampled_data = sampled_data * pca;
end


dispims(sampled_data(1:2,:)',height,width);
