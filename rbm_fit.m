function [model, errors] = rbm_fit(X, numhid, y, varargin)
%TAGS: WARNING
%Learn RBM with Bernoulli hidden and visible units
%This is not meant to be applied to image data
%code by Andrej Karpathy
%based on implementation of Kevin Swersky and Ruslan Salakhutdinov

%INPUTS: 
%X              ... data. should be binary, or in [0,1] to be interpreted 
%               ... as probabilities
%numhid         ... number of hidden layers

%additional inputs (specified as name value pairs or in struct)
%nclasses       ... number of classes
%method         ... CD or SML 
%eta            ... learning rate
%momentum       ... momentum for smoothness amd to prevent overfitting
%               ... NOTE: momentum is not recommended with SML
%maxepoch       ... # of epochs: each is a full pass through train data
%avglast        ... how many epochs before maxepoch to start averaging
%               ... before. Procedure suggested for faster convergence by
%               ... Kevin Swersky in his MSc thesis
%penalty        ... weight decay factor
%batchsize      ... The number of training instances per batch
%verbose        ... For printing progress
%anneal         ... Flag. If set true, the penalty is annealed linearly
%               ... through epochs to 10% of its original value
%gaussian_vis   ... default is false meaning that we are using binary
%               ... visible units. true means that we are using linear
%               ... gaussian visible units. NOTE: Make sure data is
%               ... normalized to have 0 mean and variance of 1 before
%               ... setting this to true.
%vis_fantasy    ... Specifies whether the sample data in the negative phase
%               ... should be visualized. The default is false. If vis_fantasyis set to
%               ... true then we must have (data dimension) =
%               ... vis_fantasy_height * vis_fantasy_width.
%vis_fantasy_height ... height of the sample data image.
%vis_fantasy_width  ... width of the sample data image.
%lambda         ... controls the relative importance of the data likelihood
%               ... term and the sparsity term. Default is 0. Note:
%               ... experiments show that for cross-validation increasing
%               ... this term on a exponential scale is recommended. Based
%               ..  on paper "Sparse deep belief net model for visual area
%               ... V2"
%p              ... 0<= p <= 1, controls how sparse the hidden units
%               ... activity is. default is 0.1

%OUTPUTS:
%model.type     ... Type of RBM (i.e. type of its visible and hidden units)
%model.W        ... The weights of the connections
%model.b        ... The biases of the hidden layer
%model.c        ... The biases of the visible layer
%model.top      ... The activity of the top layer, to be used when training
%               ... DBN's
%errors         ... The errors in reconstruction at every epoch

%Process options
%if args are just passed through in calls they become cells
if (isstruct(varargin)) 
    args= prepareArgs(varargin{1});
else
    args= prepareArgs(varargin);
end
[   nclasses      ...
    method        ...
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
    batch_perm    ...
    p             ...
    ] = process_options(args    , ...
    'nclasses'      , nunique(y), ...
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
    'batch_perm'    ,  true     , ...
    'p'             ,  0.1);
avgstart = maxepoch - avglast;
oldpenalty= penalty;
[N,d]=size(X);

if (vis_fantasy == true && d ~= vis_fantasy_height*vis_fantasy_width) 
    error ('If vis_fantasy = true then we must have (data dimension) = vis_fantasy_height * vis_fantasy_width.') ;
end

if (verbose) 
    fprintf('Preprocessing data...\n');
end

%Create targets: 1-of-k encodings for each discrete label
u= unique(y);
targets= zeros(N, nclasses);
for i=1:length(u)
    targets(y==u(i),i)=1;
end

%Create batches
numcases=N;
numdims=d;
numbatches= ceil(N/batchsize);
groups= repmat(1:numbatches, 1, batchsize);
groups= groups(1:N);
if batch_perm
    perm=randperm(N);
    groups = groups(perm);
end
for i=1:numbatches
    batchdata{i}= X(groups==i,:);
    batchtargets{i}= targets(groups==i,:);
end

%fit RBM
numclasses= length(u);
W = 0.001*randn(numdims,numhid);
c = zeros(1,numdims);
b = zeros(1,numhid);
Wc = 0.001*randn(numclasses,numhid);
cc = zeros(1,numclasses);
ph = zeros(numcases,numhid);
nh = zeros(numcases,numhid);
phstates = zeros(numcases,numhid);
nhstates = zeros(numcases,numhid);
negdata = zeros(numcases,numdims);
negdatastates = zeros(numcases,numdims);
Winc  = zeros(numdims,numhid);
binc = zeros(1,numhid);
cinc = zeros(1,numdims);
Wcinc = zeros(numclasses,numhid);
ccinc = zeros(1,numclasses);
Wavg = W;
bavg = b;
cavg = c;
Wcavg = Wc;
ccavg = cc;
t = 1;
errors=zeros(1,maxepoch);

for epoch = 1:maxepoch
    negdata_mean_sum = 0 ;
    weights_gradient_mean_sum = 0 ;
	errsum=0;
    if (anneal)
        %apply linear weight penalty decay
        penalty= oldpenalty - 0.9*epoch/maxepoch*oldpenalty;
    end
    
    for batch = 1:numbatches
		[numcases numdims]=size(batchdata{batch});
		data = batchdata{batch};
        classes = batchtargets{batch};
        
        %go up
		ph = logistic(data*W + repmat(b,numcases,1));
		phstates = ph > rand(numcases,numhid);
        if (isequal(method,'SML'))
            if (epoch == 1 && batch == 1)
                nhstates = phstates;
            end
        elseif (isequal(method,'CD'))
            nhstates = phstates;
        end
		
        %go down
        if (~gaussian_vis) % binary visible units
            negdata = logistic(nhstates*W' + repmat(c,numcases,1));
            negdatastates = negdata > rand(numcases,numdims);
        else % gaussian linear units
            negdata = nhstates*W' + repmat(c,numcases,1) ;
            negdatastates = negdata ;
            if verbose
                negdata_mean = mean(mean(negdata)) ;
                negdata_mean_sum = negdata_mean_sum + negdata_mean ;
                if abs(negdata_mean) > 0.5
                    warning('The absolute mean of the generated data is greater than 0.5. It is %f', negdata_mean) ;
                end
            end
        end
        negclasses = softmaxPmtk(nhstates*Wc' + repmat(cc,numcases,1));
		negclassesstates = softmax_sample(negclasses);
        
        %go up one more time
        nh = logistic(negdatastates*W + negclassesstates*Wc + ... 
            repmat(b,numcases,1));
		nhstates = nh > rand(numcases,numhid);
		
        %update weights and biases
        %dW = (data'*ph - negdatastates'*nh);
        %NOTE: Using prob(data)*binary(hidden) - prob(data)*prob(hidden).
        % However, if using gaussian linear units then:
        % prob(data)*binary(hidden) - (data)*prob(hidden).
        %See "A practical guide for training RBM"
        dW = (data'*phstates - negdata'*nh);
        if verbose
           weights_gradient_mean = (mean(mean(dW/numcases))) ;
           weights_gradient_mean_sum = weights_gradient_mean_sum + weights_gradient_mean ;
           if weights_gradient_mean > 10^2
               warning('average dW over all filters might be exploding. It is %f', weights_gradient_mean) ;
           end
        end   
        %dc = sum(data) - sum(negdatastates);
        dc = sum(data) - sum(negdata);
        %db = sum(ph) - sum(nh);
        db = sum(phstates) - sum(nh);
        
        %sparsification
        A = 1/numhid*phstates.*logistic(data*W + repmat(b,numcases,1));
        B = p - sum(A') ;
        D = -1/numhid*phstates.*logistic(data*W + repmat(b,numcases,1)).*...
            (1 - logistic(data*W + repmat(b,numcases,1))); 
        db = db - 2*lambda*B*D ;

        dWc = (classes'*ph - negclassesstates'*nh);
        dcc = sum(classes) - sum(negclassesstates);
		Winc = momentum*Winc + eta*(dW/numcases - penalty*W);
		binc = momentum*binc + eta*(db/numcases);
		cinc = momentum*cinc + eta*(dc/numcases);
        Wcinc = momentum*Wcinc + eta*(dWc/numcases - penalty*Wc);
		ccinc = momentum*ccinc + eta*(dcc/numcases);
		W = W + Winc;
		b = b + binc;
		c = c + cinc;
        Wc = Wc + Wcinc;
		cc = cc + ccinc;
        
        if (epoch > avgstart)
            %apply averaging
			Wavg = Wavg - (1/t)*(Wavg - W);
			cavg = cavg - (1/t)*(cavg - c);
			bavg = bavg - (1/t)*(bavg - b);
            Wcavg = Wcavg - (1/t)*(Wcavg - Wc);
			ccavg = ccavg - (1/t)*(ccavg - cc);
			t = t+1;
		else
			Wavg = W;
			bavg = b;
			cavg = c;
            Wcavg = Wc;
			ccavg = cc;
        end
        
        %accumulate reconstruction error
        err= sum(sum( (data-negdata).^2 ));
		errsum = err + errsum;
    end
    
    errors(epoch)=errsum;
    if (verbose) 
        fprintf('Ended epoch %i/%i. Reconstruction error is %f\n', ...
            epoch, maxepoch, errsum);
        fprintf('The mean of the generated data over all batches is %f \n', ...
            mean(negdata_mean_sum)/numbatches) ;
        fprintf('average dW over all filters and batches is %f \n', ...
            weights_gradient_mean_sum/numbatches) ;
    end
    if (vis_fantasy == true)
        dispims(negdata',vis_fantasy_height,vis_fantasy_width);
        drawnow
    end
end

if gaussian_vis
    model.type= 'CB';
else
    model.type= 'BB';
end
model.top= logistic(X*Wavg + repmat(bavg,N,1));
model.W= Wavg;
model.b= bavg;
model.c= cavg;
model.Wc= Wcavg;
model.cc= ccavg;
model.labels= u;
