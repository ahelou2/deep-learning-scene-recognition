clear all

configure

%model = 'indep';
model = 'cltree';
loc = true;
gist = true;
useSamples = true;
debugmode = false;

load_tree
% Dtest = ground truth for test set
% DdetectorTest = output of the detector on the set

Nimages = length(DdetectorTest);
DdetectorTestContext = DdetectorTest;
presence_score = zeros([Ncategories, Nimages]);
presence_score_c = zeros([Ncategories, Nimages]);
presence_truth = zeros([Ncategories, Nimages]);

% Apply the hierarchical context model
apply_hcontext

% Evaluate the performance of the baseline and the context model.
eval_performance
