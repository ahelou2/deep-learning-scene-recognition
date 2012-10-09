function [prediction acc] = dbnPredict(models, testdata, testlabels)
%Use DBN to predict discrete label for testdata

%INPUTS:
%m          ... is the model from dbnFit()
%testdata   ... binary, or in [0,1] interpreted as probabilities

%OUTPUTS:
%prediction ... the discrete labels for every class

%map input all the way to the top
for i=1:length(models)-1
    testdata= rbmVtoH(models{i}, testdata);
end

%and predict on the last layer
prediction= rbmPredict(models{length(models)}, testdata);

acc = sum(prediction == testlabels)/length(testlabels) ;
