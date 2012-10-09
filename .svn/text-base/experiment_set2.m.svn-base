load experiment_set2.mat;
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

model_1= dbnFit(train_data, [1000 1000 2000], y_air_motor_train, ...
    op_vis, op_hid1, op_hid2) ;
result_1.err = unsuperv_offline_eval(test_data, model_1, 50, 75, pca);
[result_1.prediction result_1.acc] = dbnPredict(model_1, test_data, y_air_motor_test); 

fprintf('Reconstruction error on test set %f\n', result_1.err);
fprintf('Accuracy on test set %f\n', result_1.acc);
%Results
%  - reconstruction error 210813.058149
%    reconstruction figures still has the problem of airplanes and
%    motorcycles superimposed.
%  - accuracy 0.9805
%  - features in layer 3 seems to be better

% Experiment 2 make more sparse

fprintf('\n \n Experiment 2 \n \n')
op_vis.p = 0.05 ;
op_hid1.p = 0.05 ;
op_hid1.p = 0.05 ;

model_2= dbnFit(train_data, [1000 1000 2000], y_air_motor_train, ...
    op_vis, op_hid1, op_hid2) ;
result_2.err = unsuperv_offline_eval(test_data, model_2, 50, 75, pca);
[result_2.prediction result_2.acc] = dbnPredict(model_2, test_data, y_air_motor_test); 

fprintf('Reconstruction error on test set %f\n', result_2.err);
fprintf('Accuracy on test set %f\n', result_2.acc);
%Results
%  - reconstruction error 214331.661779
%    reconstruction figures still has the problem of airplanes and
%    motorcycles superimposed.
%  - accuracy 0.9870
%  - features in layer 3 seems to be better

% Experiment 3 no sparsity

fprintf('\n \n Experiment 3 \n \n')
op_vis.lambda = 0 ;
op_hid1.lambda = 0 ;
op_hid2.lambda = 0 ;

model_3= dbnFit(train_data, [1000 1000 2000], y_air_motor_train, ...
    op_vis, op_hid1, op_hid2) ;
result_3.err = unsuperv_offline_eval(test_data, model_3, 50, 75, pca);
[result_3.prediction result_3.acc] = dbnPredict(model_3, test_data, y_air_motor_test); 

fprintf('Reconstruction error on test set %f\n', result_3.err);
fprintf('Accuracy on test set %f\n', result_3.acc);
%Results
%  - reconstruction error 215596.025343
%    reconstruction figures still has the problem of airplanes and
%    motorcycles superimposed.
%  - accuracy 0.9740
%  - features in layer 3 much better

% Experiment 4 no sparsity, one hidden layer


fprintf('\n \n Experiment 4 \n \n')
model_4= dbnFit(train_data, 1000, y_air_motor_train, op_vis) ;
result_4.err = unsuperv_offline_eval(test_data, model_4, 50, 75, pca);
[result_4.prediction result_4.acc] = dbnPredict(model_4, test_data, y_air_motor_test); 

fprintf('Reconstruction error on test set %f\n', result_4.err);
fprintf('Accuracy on test set %f\n', result_4.acc);
%Results
%  - reconstruction error 127269.869330
%    reconstruction figures still has the problem of airplanes and
%    motorcycles superimposed.
%  - accuracy 1


save workspace_exp_2;
clear all;
