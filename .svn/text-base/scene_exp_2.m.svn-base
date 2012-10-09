height = 100;
width = 100;
PADSIZE = [height width];
PADVAL = 0;
numTrain = 100;
train_data = [];
train_data_color = [];
y_train = [];
test_data = [];
test_data_color = [];
y_test = [];
for i=1:1:n
     data = [];
     data_color = [];
     m = scene{i}.numImages;
     for j=1:1:m
         scaling = min(height/size(scene{i}.image{j},1), width/size(scene{i}.image{j},2));
         newsize = round([size(scene{i}.image{j},1) size(scene{i}.image{j},2)]*scaling);
         img_color = imresize(scene{i}.image{j}, newsize, 'nearest');
         [nrows, ncols, cols]=size(img_color);
         img = rgb2gray(img_color);
         data_color = [data_color; reshape(img_color, [1 nrows*ncols*cols])];
         data = [data; reshape(img, [1 nrows*ncols])]; 
     end
     train_data_color = [train_data_color; data_color(1: numTrain,:)];
     train_data = [train_data; data(1: numTrain,:)];
     y_train = [y_train; repmat(i, [ numTrain 1])];
     
     test_data_color = [test_data_color; data_color(numTrain+1:m,:)];
     test_data = [test_data; data(numTrain+1:m, :)];
     y_test = [y_test; repmat(i, [m-numTrain 1])];
 end

train_idx = randperm(numTrain*n);
train_data_color = normalize(double(train_data_color(train_idx, :)));
train_data = normalize(double(train_data(train_idx, :)));
y_train = y_train(train_idx, :);
test_data = normalize(double(test_data));
test_data_color = normalize(double(test_data_color));

op_vis.gaussian_vis = 1;
op_vis.batchsize = 25 ;
op_vis.lambda = 0.5 ;
op_vis.maxepoch = 200 ;
op_vis.eta = 0.005 ;

op_hid1 = op_vis ;
op_hid1.gaussian_vis = 0 ;
op_hid1.eta = 0.05;
op_hid2 = op_hid1 ;

[model_1, error_1]= dbnFit(train_data, [500 500 1000], y_train, op_vis, op_hid1, op_hid2) ;
[result_1.err result_1.probs] = unsuperv_offline_eval(train_data, model_1, height, width, 'feat_probs', true);
[result_1.test_err result_1.test_probs] = unsuperv_offline_eval(test_data, model_1, height, width, 'feat_probs', true);
[result_1.prediction result_1.acc] = dbnPredict(model_1, test_data, y_test); 
fprintf('\n \n Experiment 1: Gray Scale\n \n')
fprintf('Reconstruction error on training set %f\n', result_1.err);
fprintf('Reconstruction error on test set %f\n', result_1.test_err);
fprintf('Accuracy on test set %f\n', result_1.acc);
SVMdata(result_1.probs, y_train, result_1.test_probs, y_test, 'exp1_model_1_train', 'exp1_model_1_test');

op_vis.maxepoch = 600 ;
op_hid1.maxepoch = 600 ;
op_hid2.maxepoch = 600 ;
[model_2, error_2] = dbnFit(train_data_color, [500 500 1000], y_train, op_vis, op_hid1, op_hid2) ;
[result_2.err, result_2.probs] = unsuperv_offline_eval(train_data_color, model_2, height, width, 'feat_probs', true);
[result_2.test_err result_2.test_probs] = unsuperv_offline_eval(test_data_color, model_2, height, width, 'feat_probs', true);
[result_2.prediction result_2.acc] = dbnPredict(model_2, test_data_color, y_test); 
fprintf('\n \n Experiment 2: Color \n \n');
fprintf('Reconstruction error on training set %f\n', result_2.err);
fprintf('Reconstruction error on test set %f\n', result_2.test_err);
fprintf('Accuracy on test set %f\n', result_2.acc);
SVMdata(result_2.probs, y_train, result_2.test_probs, y_test, 'exp1_model_2_train', 'exp1_model_2_test');

op_vis.lambda = 0.2 ;
op_hid1.lambda = 0.2 ;
op_hid2.lambda = 0.2 ;

op_vis.maxepoch = 200 ;
op_hid1.maxepoch = 200 ;
op_hid2.maxepoch = 200 ;

[model_3, error_3]= dbnFit(train_data, [500 500 1000], y_train, op_vis, op_hid1, op_hid2) ;
[result_3.err, result_3.probs] = unsuperv_offline_eval(train_data, model_3, height, width, 'feat_probs', true);
[result_3.test_err result_3.test_probs] = unsuperv_offline_eval(test_data, model_3, height, width, 'feat_probs', true);
[result_3.prediction result_3.acc] = dbnPredict(model_3, test_data, y_test); 
fprintf('\n \n Experiment 3: Gray Scale \n \n');
fprintf('Reconstruction error on training set %f\n', result_3.err);
fprintf('Reconstruction error on test set %f\n', result_3.test_err);
fprintf('Accuracy on test set %f\n', result_3.acc);
SVMdata(result_3.probs, y_train, result_3.test_probs, y_test, 'exp1_model_3_train', 'exp1_model_3_test');

op_vis.maxepoch = 600 ;
op_hid1.maxepoch = 600 ;
op_hid2.maxepoch = 600 ;

[model_4, error_4] = dbnFit(train_data_color,[500 500 1000], y_train,op_vis, op_hid1, op_hid2);
[result_4.err, result_4.probs] = unsuperv_offline_eval(train_data_color, model_4, height, width, 'feat_probs', true);
[result_4.test_err result_4.test_probs] = unsuperv_offline_eval(test_data_color, model_4, height, width, 'feat_probs', true);
[result_4.prediction result_4.acc] = dbnPredict(model_4, test_data_color, y_test); 
fprintf('\n \n Experiment 4: Color  \n \n');
fprintf('Reconstruction error on training set %f\n', result_4.err);
fprintf('Reconstruction error on test set %f\n', result_4.test_err);
fprintf('Accuracy on test set %f\n', result_4.acc);
SVMdata(result_4.probs, y_train, result_4.test_probs, y_test,'exp1_model_4_train', 'exp1_model_4_test');
op_vis.lambda = 0.1 ;
op_hid1.lambda = 0.1 ;
op_hid2.lambda = 0.1 ;

op_vis.maxepoch = 200 ;
op_hid1.maxepoch = 200 ;
op_hid2.maxepoch = 200 ;


[model_5, error_5] = dbnFit(train_data, [500 500 1000], y_train, op_vis, op_hid1, op_hid2) ;
[result_5.err, result_5.probs] = unsuperv_offline_eval(train_data, model_5, height, width, 'feat_probs', true);
[result_5.prediction result_5.acc] = dbnPredict(model_5, test_data, y_test); 
[result_5.test_err result_5.test_probs] = unsuperv_offline_eval(test_data, model_5, height, width, 'feat_probs', true);
fprintf('\n \n Experiment 5: Gray scale \n \n');
fprintf('Reconstruction error on training set %f\n', result_5.err);
fprintf('Reconstruction error on test set %f\n', result_5.test_err);
fprintf('Accuracy on test set %f\n', result_5.acc);
SVMdata(result_5.probs, y_train, result_5.test_probs, y_test, 'exp1_model_5_train', 'exp1_model_5_test');

op_vis.maxepoch = 600 ;
op_hid1.maxepoch = 600 ;
op_hid2.maxepoch = 600 ;

[model_6, error_6] = dbnFit(train_data_color, [500 500 1000], y_train, op_vis, op_hid1, op_hid2) ;
[result_6.err, result_6.probs] = unsuperv_offline_eval(train_data_color, model_6, height, width, 'feat_probs', true);
[result_6.test_err result_6.test_probs] = unsuperv_offline_eval(test_data_color, model_6, height, width, 'feat_probs', true);
[result_6.prediction result_6.acc] = dbnPredict(model_6, test_data_color, y_test); 
fprintf('\n \n Experiment 6: Color  \n \n');
fprintf('Reconstruction error on training set %f\n', result_6.err);
fprintf('Reconstruction error on test set %f\n', result_6.test_err);
fprintf('Accuracy on test set %f\n', result_6.acc);
SVMdata(result_6.probs, y_train, result_6.test_probs, y_test, 'exp1_model_6_train', 'exp1_model_6_test');
save scene_experiment_1; 
