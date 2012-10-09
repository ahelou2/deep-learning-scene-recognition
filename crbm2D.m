function [model] = crbm2D(X, N_V2D, N_W2D, K, C2D, varargin)

%WARNING: This is machine specific
if (matlabpool('size')) > 0
    matlabpool close
end
matlabpool(2)
% Learn convolutional rbm with Bernoulli hidden and visible variables as
% well as gaussian continuous visible variables.

%Process options
%if args are just passed through in calls they become cells
if (isstruct(varargin)) 
    args= prepareArgs(varargin{1});
else
    args= prepareArgs(varargin);
end
[   method        ...
    eta           ...
    momentum      ...
    maxepoch      ...
    avglast       ...
    penalty       ...
    batchsize     ...
    verbose       ...
    anneal        ...
    gaussian_vis  ...
    vis_fantasy   ...
    vis_fantasy_height ...
    vis_fantasy_width ...
    lambda        ...
    p             ...
    batch_perm    ...
    vis_features  ...
    simple_sparsification ...
    sparse_bias_val   ...
    ] = process_options(args    , ...
    'method'        ,  'CD'     , ...
    'eta'           ,  0.1      , ...
    'momentum'      ,  0.5      , ...
    'maxepoch'      ,  50       , ...
    'avglast'       ,  5        , ...
    'penalty'       , 2e-4      , ...
    'batchsize'     , 100       , ...
    'verbose'       , false     , ...
    'anneal'        , false     , ...
    'gaussian_vis'  , false     , ...
    'vis_fantasy'   , false     , ...
    'vis_fantasy_height' , 0    , ...
    'vis_fantasy_width'  , 0    , ...
    'lambda'        ,  0        , ...
    'p'             ,  0.1      , ...
    'batch_perm'    ,  true     , ...
    'vis_features'  ,  false    , ...
    'simple_sparsification', false, ...
    'sparse_bias_val',  -3);
avgstart = maxepoch - avglast;
oldpenalty= penalty;
[N,d, numfeat_maps]=size(X);
% WARNING: Hack for making sure that all batches have the same size
X(1:(mod(N, batchsize)), :,:) = [] ;
[N,d, numfeat_maps]=size(X) ;

if (vis_fantasy == true && d ~= vis_fantasy_height*vis_fantasy_width) 
    error ('If vis_fantasy = true then we must have (data dimension) = vis_fantasy_height * vis_fantasy_width.') ;
end

if (verbose) 
    fprintf('Preprocessing data...\n');
end

%Create batches
numcases=N ;
numdims=d;
numbatches= ceil(N/batchsize);
groups= repmat(1:numbatches, 1, batchsize);
groups= groups(1:N);
if batch_perm
    perm=randperm(N);
    groups = groups(perm);
end
for i=1:numbatches
    batchdata{i}= X(groups==i,:,:);
end

% Train cRBM

% Parameters initializations
N_H2D = N_V2D - N_W2D + 1 ;
fprintf('N_H2D = [%i %i] \n', N_H2D(1), N_H2D(2)) ;
if sum(mod(N_H2D./C2D, [1 1]) ~= [0 0])
    N_H2D
    C2D
   error('N_H2D must be divisble by C2D') ; 
end
% P is pooling layer but I don't think I need it in this function
P = zeros(N_H2D./C2D) ;
pooldims = N_H2D./C2D ;
ph = zeros(batchsize, N_H2D(1)*N_H2D(2),K) ;
nh = zeros(batchsize, N_H2D(1)*N_H2D(2),K) ;
phstates = zeros(batchsize, N_H2D(1)*N_H2D(2),K) ;
W = 0.001*randn([N_W2D K numfeat_maps]);
c = zeros(1, numfeat_maps) ;
b = zeros(1, K) ;
if simple_sparsification
    b = sparse_bias_val*ones(1, K) ;
end
Winc = zeros([N_W2D K numfeat_maps]) ;
binc = zeros(1, K) ;
cinc = zeros(1, numfeat_maps) ;
dW = zeros([N_W2D K numfeat_maps]) ;
db = zeros(1, K) ;
Wavg = W;
bavg = b;
cavg = c;
t = 1;
errors=zeros(1,maxepoch);

for epoch = 1:maxepoch
    tic
    errsum=0;
    negdata_mean_sum = 0  ;
    %weights_gradient_mean_sum = zeros(1, numfeat_maps) ;
    weights_gradient_mean_sum = 0 ;
    if (anneal)
        %apply linear weight penalty decay
        penalty= oldpenalty - 0.9*epoch/maxepoch*oldpenalty;
    end
    
    for batch = 1:numbatches
		[numcases numdims numfeat_maps]=size(batchdata{batch});
		data = batchdata{batch};
        
        %go up
        % WARNING: I am fairly confident I can get rid of the following
        % FOR-loop!
        %for i=1:1:K
        parfor i=1:1:K
            %W_conv_data = zeros([N_V2D numcases] - N_W2D + 1) ;
            W_conv_data = zeros([(N_V2D - N_W2D + 1) numcases]) ;
            for j=1:1:numfeat_maps
                data_for_conv = reshape(data(:,:,j)', [N_V2D numcases]) ;
                 W_conv_data = W_conv_data + ...
                     convn(data_for_conv, flipud(fliplr(W(:,:,i,j))), 'valid') ;
            end
            
            W_conv_data = ...
                reshape(W_conv_data, [N_H2D(1)*N_H2D(2) numcases]) ;
            W_conv_data = W_conv_data' ;
            [ph(:,:,i), z, phstates(:,:,i)] = multinomial_exp(W_conv_data + repmat(b(1,i), [numcases N_H2D(1)*N_H2D(2)]), C2D, N_H2D) ;
            %phstates(:,:,i) = multinomial_exp_sample(ph(:,:,i), C2D, N_H2D) ;
        end
        
        if (isequal(method,'SML'))
            if (epoch == 1 && batch == 1)
                nhstates = phstates;
            end
        elseif (isequal(method,'CD'))
            nhstates = phstates;
        end
        
        
        %go down
        % WARNING: I am having doubt over how I implemented thw down pass:
        % am I supposed to enforce that the data generated is such that
        % at most one hidden unit is active in every N_W*N_W block?
        j=1 ; % Used to allow for parallel processing
        %for j=1:1:numfeat_maps
        N_V2D1 = N_V2D(1) ;
        N_V2D2 = N_V2D(2) ;
        negdata_mean = zeros(numfeat_maps,1) ;
        parfor z=1:numfeat_maps
        %for z=1:numfeat_maps
            %W_conv_hid_sum = zeros(numcases, N_V2D1*N_V2D2) ;
            parW = W(:,:,:,z) ;
            parW_conv_hid = zeros(numcases, N_V2D1*N_V2D2, K) ;
            for i=1:1:K
                hid_for_conv = reshape(nhstates(:,:,i)', [N_H2D numcases]) ;
                W_conv_hid = convn(hid_for_conv, parW(:,:,i)) ;
                W_conv_hid = ...
                    reshape(W_conv_hid, [N_V2D1*N_V2D2 numcases]) ;
                parW_conv_hid(:,:,i) = W_conv_hid' ;
                %W_conv_hid_sum = W_conv_hid_sum + W_conv_hid ;
            end
            W_conv_hid_sum = sum(parW_conv_hid,3) ;
            if (~gaussian_vis) % binary visible units
%                 size(W_conv_hid_sum)
%                 size(repmat(c(1,z),numcases, N_V2D1*N_V2D2))
                negdata(:,:,z) = logistic(W_conv_hid_sum + repmat(c(1,z),numcases, N_V2D1*N_V2D2));
                negdatastates(:,:,z) = negdata(:,:,z) > rand(numcases,N_V2D1*N_V2D2);
            else % gaussian linear units
                negdata(:,:,z) = W_conv_hid_sum + repmat(c(1,z),numcases, N_V2D1*N_V2D2) ;
                if verbose
                    negdata_mean(z,1) = mean(mean(negdata(:,:,z))) ;
                    %negdata_mean_sum = negdata_mean_sum + negdata_mean ; 
                    %fprintf('The mean of the generated data is %f \n', negdata_mean) ;
                    if abs(negdata_mean(z,1)) > 0.5
                        warning('The absolute mean of the generated data is greater than 0.5. It is %f', negdata_mean(z,1)) ;
                    end
                end
                negdatastates(:,:,z) = negdata(:,:,z) ;
            end
        end
        
        negdata_mean_sum = negdata_mean_sum + mean(negdata_mean) ;
        
        
        %go up one more time
        j=1 ; % Used to allow for parallel processing
        %for i=1:1:K
        parfor i=1:1:K
            %W_conv_data = zeros([N_V2D numcases] - N_W2D + 1) ;
            W_conv_data = zeros([(N_V2D - N_W2D + 1) numcases]) ;
            for j=1:1:numfeat_maps
                data_for_conv = reshape(negdatastates(:,:,j)', [N_V2D numcases]) ;
                W_conv_data = W_conv_data + ...
                    convn(data_for_conv, flipud(fliplr(W(:,:,i,j))), 'valid') ;
            end
             W_conv_data = ...
                reshape(W_conv_data, [N_H2D(1)*N_H2D(2) numcases]) ;
            W_conv_data = W_conv_data' ;
            [nh(:,:,i), z, nhstates(:,:,i)] = multinomial_exp(W_conv_data + repmat(b(1,i), [numcases N_H2D(1)*N_H2D(2)]), C2D, N_H2D) ;
            %nhstates(:,:,i) = multinomial_exp_sample(nh(:,:,i), C2D, N_H2D) ;
        end
        
        
        %update weights and biases
        %NOTE: Using prob(data)*binary(hidden) - prob(data)*prob(hidden).
        % However, if using gaussian linear units then:
        % prob(data)*binary(hidden) - (data)*prob(hidden).
        %See "A practical guide for training RBM"
        %BUT, it seemed like learning was faster when using:
        %prob(data)*binary(hidden) - prob(data)*binary(hidden).
        weights_gradient_mean = zeros(1,numfeat_maps) ;
        data_conv_phstates = zeros([N_W2D, K, numfeat_maps]) ;
        negdata_conv_nh = zeros([N_W2D, K, numfeat_maps]) ;
        for i=1:1:K
            % According to "Stacks of 
            % Convolutional Restricted Boltzmann Machines for Shift-Invariant 
            % Feature Learning", I simply need to convolve the data with
            % the hidden states.
            % NOTE: There is a "deconv" function that may be useful
            phstates_for_conv = reshape(phstates(:,:,i)', [N_H2D numcases]) ;
            nh_for_conv = reshape(nh(:,:,i)', [N_H2D numcases]) ;
%             data_conv_phstates = zeros([N_W2D, K, numfeat_maps]) ;
%             negdata_conv_nh = zeros([N_W2D, K, numfeat_maps]) ;
            %for j=1:1:numfeat_maps
            parfor j=1:1:numfeat_maps
                data_for_conv = reshape(data(:,:,j)', [N_V2D numcases]) ;
                negdata_for_conv = reshape(negdata(:,:,j)', [N_V2D numcases]) ;
                tmp1 = zeros([N_W2D numcases]) ;
                tmp2 = zeros([N_W2D numcases]) ;
                for k=1:1:numcases
                    tmp1(:,:,k) = ...
                        conv2(data_for_conv(:,:,k), flipud(fliplr(phstates_for_conv(:,:,k))), 'valid') ;
                    tmp2(:,:,k) = ...
                        conv2(negdata_for_conv(:,:,k), flipud(fliplr(nh_for_conv(:,:,k))), 'valid') ;
                end
%                 data_conv_phstates(:,:,i,j) = data_conv_phstates(:,:,i,j) + sum(tmp1,3) ; 
%                 negdata_conv_nh(:,:,i,j) = negdata_conv_nh(:,:,i,j) + sum(tmp2,3) ;
                data_conv_phstates(:,:,i,j) = sum(tmp1,3) ;
                negdata_conv_nh(:,:,i,j) = sum(tmp2,3) ;
            end
        end
        dW = data_conv_phstates - negdata_conv_nh ;
        db = squeeze(sum(sum(phstates))/(N_H2D(1)*N_H2D(2)) - sum(sum(nh))/(N_H2D(1)*N_H2D(2)))' ;
        Winc = momentum*Winc + eta*(dW/numcases - penalty*W);
        binc = momentum*binc + eta*(db/numcases);
        W = W + Winc;
        
        
        if ~simple_sparsification
            b = b + binc ; 
        end
        
        if verbose
           %fprintf('average dW over all filters is %f \n', weights_gradient_mean) ;
           weights_gradient_mean = mean(mean(mean(mean(dW/numcases)))) ;
           weights_gradient_mean_sum = weights_gradient_mean_sum + weights_gradient_mean ;
           if weights_gradient_mean > 10^2
               warning('average dW over all filters might be exploding. It is %f', weights_gradient_mean) ;
           end
        end
        dc = squeeze(sum(sum(data))/(N_V2D(1)*N_V2D(2)) - sum(sum(negdata))/(N_V2D(1)*N_V2D(2)))' ;
        cinc = momentum*cinc + eta*(dc/numcases);
        c = c + cinc;
        if (epoch > avgstart)
            %apply averaging
            %for i=1:1:K
                Wavg = Wavg - (1/t)*(Wavg - W);
                bavg = bavg - (1/t)*(bavg - b);
            %end
            cavg = cavg - (1/t)*(cavg - c);
            t = t+1;
        else
            %for i=1:1:K
                Wavg = W;
                bavg = b;
            %end
            cavg = c;
        end
        %accumulate reconstruction error
        err= sum(sum(sum( (data-negdata).^2 )));
		errsum = err + errsum;
%         clear ph ;
%         clear phstates ;
%         clear nh ;
%         clear nhstates ;
    end
    errors(epoch)=errsum;
    if (verbose) 
        fprintf('Ended epoch %i/%i. Reconstruction error is %f\n', ...
            epoch, maxepoch, errsum);
        fprintf('The mean of the generated data over all batches is %f \n', ...
            negdata_mean_sum/numbatches) ;
        fprintf('average dW over all filters and batches is %f \n', ...
            weights_gradient_mean_sum/numbatches) ;
    end
    if (vis_fantasy == true)
        figure(1); 
        dispims(negdatastates(1:10,:,1)',vis_fantasy_height,vis_fantasy_width);
        drawnow
    end
    if (vis_features)
        figure(2)
        model.W = Wavg ;
        cvis_features(model) ;
    end
    toc
end

if gaussian_vis
    model.type= 'CB';
else
    model.type= 'BB';
end

numcases = size(X,1) ;
for i =1:1:K
    W_conv_data = zeros([(N_V2D - N_W2D + 1) numcases]) ;
    for j=1:1:numfeat_maps
        data_for_conv = reshape(X(:,:,j)', [N_V2D numcases]) ;
        W_conv_data = W_conv_data + ...
            convn(data_for_conv, flipud(fliplr(Wavg(:,:,i,j))), 'valid') ;
    end
    W_conv_data = ...
        reshape(W_conv_data, [N_H2D(1)*N_H2D(2) numcases]) ;
    W_conv_data = W_conv_data' ;
    [model.hid_top(:,:,i) model.pool_top(:,:,i)] = multinomial_exp(W_conv_data + repmat(b(1,i), ...
        [numcases N_H2D(1)*N_H2D(2)]), C2D, N_H2D) ;
end
%model.hid_top = interleave(hid_top) ;
%model.pool_top = interleave(pool_top) ;
model.W= Wavg;
model.b= bavg;
model.c= cavg;
model.num_filters = K ;
model.N_V2D = N_V2D ;
model.N_V2D_next = N_H2D./C2D ;
model.N_W2D = N_W2D ;
model.C2D = C2D ;
model.N_H2D = N_H2D ;
end

function isnan_isinf(X, name)
    if sum(sum(isnan(X))) || sum(sum(isinf(X)))
        X
       error('%s', name) ; 
    end
end