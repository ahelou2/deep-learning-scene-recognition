function [err features] = unsuperv_offline_eval(X, model, height, width, varargin)

%Process options
%if args are just passed through in calls they become cells
if (isstruct(varargin)) 
    args= prepareArgs(varargin{1});
else
    args= prepareArgs(varargin);
end
[   pca        ...
    feat_probs ...
    ] = process_options(args    , ...
    'pca'             ,  []     , ...
    'feat_probs'             ,  false);

% Evaluates squared reconstruction error of DBN

%NOTE: Feeding probabilistic hidden states to the next layer because
    %learning is sped this way *apparrently*.

H = length(model) ;
data = X ;

% UP-pass
features = [];
for i=1:H
    [numcases numdims] = size(data) ;
    W = model{i}.W ;
    c = model{i}.c ;
    b = model{i}.b ;
    numhid = size(W, 2) ;
    ph = logistic(data*W + repmat(b,numcases,1)) ;
    phstates = ph > rand(numcases,numhid);
    if ~feat_probs
        features = [features phstates];
    else
        features = [features ph];
    end

    %data = phstates ;
    data = ph ;
end

% DOWN-pass
sampled_data = data ;
for i=H:-1:1
    phstates = sampled_data ;
    [numcases numdims] = size(sampled_data) ;
    W = model{i}.W ;
    c = model{i}.c ;
    b = model{i}.b ;
    numhid = size(W, 2) ;
    if model{i}.type == 'BB'
        sampled_data = logistic(phstates*W' + repmat(c,numcases,1));
        % WARNING: In Ruslan's science rbm.m he doesn't sample but in his
        % DBM rbm.m he samples!
        sampled_data = sampled_data > rand(size(sampled_data)) ;
    elseif model{i}.type == 'CB'
       sampled_data = phstates*W' + repmat(c,numcases,1) ; 
    end
end

err= sum(sum( (X-sampled_data).^2 ));

if (length(pca) > 0)
    sampled_data = sampled_data * pca;
end

% WARNING: I'm not suer if height or width should come first.
% WARNING: dispims forces the image to be square
%dispims(sampled_data(1:20,:)',height,width);
%drawnow
