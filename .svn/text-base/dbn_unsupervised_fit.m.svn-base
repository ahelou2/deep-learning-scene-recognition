function model = dbn_unsupervised_fit(X, numhid, varargin)
%fit a DBN to bianry (or continuous) data in X. NOTE: If using continuous
%data then make sure to normalize data component-wise to have 0 mean and 1
%variance.

%INPUTS: 
%X              ... data. should be binary, or in [0,1] interpreted as
%               ... probabilities
%numhid         ... list of numbers of hidden units

%OUTPUTS:
%model          ... A cell array containing models from all RBM's

%varargin may contain options for the RBM's of this DBN, in row one by one
%for example:
%dbnFit(X, [500,400], opt1, opt2) uses opt1 for 500 and opt2 for 400
%dbnFit(X, [500,400], opt1) uses opt1 only for 500, and defaults for 400

%TODO: Should have the option to input weights and biases at any level in
%order to prevent computational waste.

numopts=length(varargin);
H=length(numhid);
model=cell(H,1);

    
%train the first RBM on data
if(numopts>=1)
    model{1}= rbm(X, numhid(1),varargin{1});
else
    model{1}= rbm(X, numhid(1));
end
if H >= 2
    %train all other RBM's on top of each other
    for i=2:H
        if(numopts>=i)
            model{i}=rbm(model{i-1}.top, numhid(i), varargin{i});
        else
            model{i}=rbm(model{i-1}.top, numhid(i));
        end

    end
end
    
