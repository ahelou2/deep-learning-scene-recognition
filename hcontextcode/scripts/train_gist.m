Ntraining = length(Dtraining);
Ntest = length(Dtest);
Nobjects = length(names);

class_training = -ones([Ntraining Nobjects]);
class_test  = -ones([Ntest Nobjects]);

for c = 1:Nobjects
    [foo, j] = LMquery(Dtraining, 'object.name', names{c}, 'exact');
    class_training(j,c) = 1;
    [foo, j] = LMquery(Dtest, 'object.name', names{c}, 'exact');
    class_test(j,c) = 1;
end

% gist Parameters:
clear param
param.imageSize = 256;
param.imageSize = [256 256];
param.orientationsPerScale = [8 8 8 8];
param.numberBlocks = 4;
param.fc_prefilt = 4;

% compute gist
gist_training = LMgist(Dtraining, HOMEIMAGES, param);
gist_test = LMgist(Dtest, HOMEIMAGES, param);

% training
% svm parameters
lambda = 0.2;
lambda = 1;
opt.iter_max_Newton = 200;
opt.cg = 1;

% building kernel
global K
svm.type = 'rbf';
svm.sig = .6;

% train and test
K = kernel(gist_training, gist_training, svm);
p_b_gist_test = zeros(Ntest, Nobjects);

figure
for c = 1:Nobjects
    % train
    Y = class_training(:,c);
    [beta,b]=primal_svm(0, Y, lambda, opt);
    
    % fit p(y|gist)
    s = K*beta+b;
    [logitCoef,dev] = glmfit(s, Y==1, 'binomial', 'link', 'logit');
    
    % test  p(b=1 | gist score)
    Kt = kernel(gist_test, gist_training, svm);
    st = Kt*beta+b;
    
    %[logitCoef,dev] = glmfit(st, class_test(:,c)==1, 'binomial', 'link', 'logit');
    p_b_gist_test(:,c) = glmval(logitCoef, st, 'logit');
    %p_b_gist_test(:,c) = st;
    
    % evaluation single class
    %subplot(5,5,c)
    %au(c)=areaROC(p_b_gist_test(:,c), class_test(:,c)==1, 'r');
    [recall, precision, th, averagePrecision(c)]=precisionRecall(p_b_gist_test(:,c)', class_test(:,c)'==1, 'r');
    title(averagePrecision(c))
    drawnow
end

