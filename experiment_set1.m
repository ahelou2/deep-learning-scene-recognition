load experiment_set1.mat;
air_motor_train_norm = normalize(air_motor_train) ;
air_motor_test_norm = normalize(air_motor_test);
[pca,score,latent,tsquare] = princomp(air_motor_train_norm, 'econ');
pca = pca';
train_data = air_motor_train_norm * pca';
test_data = air_motor_test_norm * pca';
% Experiment 1 with sparsity

fprintf('\n \n Experiment 1 \n \n')

op_vis.gaussian_vis = 1;
op_vis.verbose = 1 ;
op_vis.batchsize = 25 ;
op_vis.lambda = 1000 ;
op_vis.maxepoch = 200 ;
op_vis.eta = 0.005 ;

op_hid1 = op_vis ;
op_hid1.gaussian_vis = 0 ;

op_hid2 = op_hid1 ;

model_1= dbnFit(train_data, [500 500 1000], y_air_motor_train, op_vis, op_hid1, op_hid2) ;
result_1.err = unsuperv_offline_eval(test_data, model_1, 50, 75, pca);
[result_1.prediction result_1.acc] = dbnPredict(model_1, test_data, y_air_motor_test); 

fprintf('Reconstruction error on test set %f\n', result_1.err);
fprintf('Accuracy on test set %f\n', result_1.acc);

% Results:
%   - reconstruction err on test data = 171909.165414
%   - classification accuracy on test data = 0.9870
%   - features learned from the first layer all look like motorcycles. I
%     was hoping for edges to be learned at this stage.


% Experiment 2 smaller lambda

fprintf('\n \n Experiment 2 \n \n')
op_vis.lambda = 100 ;
op_hid1.lambda = 100 ;
op_hid2.lambda = 100 ;

model_2= dbnFit(train_data, [500 500 1000], y_air_motor_train, ...
    op_vis, op_hid1, op_hid2) ;
result_2.err = unsuperv_offline_eval(test_data, model_2, 50, 75, pca);
[result_2.prediction result_2.acc] = dbnPredict(model_2, test_data, y_air_motor_test); 

fprintf('Reconstruction error on test set %f\n', result_2.err);
fprintf('Accuracy on test set %f\n', result_2.acc);

% Results:
%   - reconstruction err on test data = 171169.451791/235097.464153, 
%   first fig looks like
%   white airplanes, motorcycles superimposed on top
%   second figure looks like motorcycles with no airplanes superimposed
%   - classification accuracy on test data = 0.9740
%   - features learned from the first layer from both feature figures 
%      all look like motorcycles. 

% Experiment 3 no sparsity

fprintf('\n \n Experiment 3 \n \n')
op_vis.lambda = 0 ;
op_hid1.lambda = 0 ;
op_hid2.lambda = 0 ;

model_3= dbnFit(train_data, [500 500 1000], y_air_motor_train, ...
    op_vis, op_hid1, op_hid2) ;
result_3.err = unsuperv_offline_eval(test_data, model_3, 50, 75, pca);
[result_3.prediction result_3.acc] = dbnPredict(model_3, test_data, y_air_motor_test); 

fprintf('Reconstruction error on test set %f\n', result_3.err);
fprintf('Accuracy on test set %f\n', result_3.acc);

% Results:
%   - reconstruction err on test data = 173392.841839/236580.533989,  
%   first fig looks like
%   white airplanes, motorcycles superimposed on top
%   second figure looks like motorcycles with no airplanes superimposed
%   - classification accuracy on test data = 0.9805
%   - features learned from the first layer from both feature figures 
%      all look like motorcycles. 


% Experiment 4 no sparsity, 2 hidden layers

fprintf('\n \n Experiment 4 \n \n')
model_4= dbnFit(train_data, [500 500], y_air_motor_train, op_vis, op_hid1) ;
result_4.err = unsuperv_offline_eval(test_data, model_4, 50, 75, pca);
[result_4.prediction result_4.acc] = dbnPredict(model_4, test_data, y_air_motor_test); 

fprintf('Reconstruction error on test set %f\n', result_4.err);
fprintf('Accuracy on test set %f\n', result_4.acc);
% Results:
%   - reconstruction err on test data = 156772.316664/208249.937224, 
%   first fig looks like
%   white airplanes, motorcycles superimposed on top
%   second figure looks like motorcycles with no airplanes superimposed
%   - classification accuracy on test data = 0.9740
%   - again, features learned from the first layer from both feature figures 
%      all look like motorcycles. 

% Experiment 5 no sparsity, one hidden layer

fprintf('\n \n Experiment 5 \n \n')
model_5= dbnFit(train_data, 500, y_air_motor_train, op_vis) ;
result_5.err = unsuperv_offline_eval(test_data, model_5, 50, 75, pca);
[result_5.prediction result_5.acc] = dbnPredict(model_5, test_data, y_air_motor_test); 

fprintf('Reconstruction error on test set %f\n', result_5.err);
fprintf('Accuracy on test set %f\n', result_5.acc);
% Results:
%   - reconstruction err on test data = 138907.878556/138632.967580, 
%   has the same problem as previous experiments 
%      first fig looks like
%      white airplanes, motorcycles superimposed on top
%   second figure looks like motorcycles with no airplanes superimposed
%   - classification accuracy on test data = 0.9805
%   - again, features learned from the first layer from both feature figures 
%      all look like motorcycles. 

save workspace_exp_1;
clear all;
