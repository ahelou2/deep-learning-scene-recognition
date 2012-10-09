%assume gray scale for now
function [data, target, traindata, y_train, testdata, y_test, mu, sigma, idx]=read_outdoor_scene(height, width, num_test_per_class)
load scene.mat;
n = length(scene);
PADSIZE = [height width];
PADVAL = 0;
data = [];
%data_color = [];
target = [];
sz = zeros(n,1);
for i=1:1:n
     sz(i) = scene{i}.numImages;    
     for j=1:sz(i)
         scaling = min(height/size(scene{i}.image{j},1), width/size(scene{i}.image{j},2));
         newsize = round([size(scene{i}.image{j},1) size(scene{i}.image{j},2)]*scaling);
         img_color = imresize(scene{i}.image{j}, newsize, 'nearest');
         [nrows, ncols, cols]=size(img_color);
         img = rgb2gray(img_color);
%         data_color = [data_color; reshape(img_color, [1 nrows*ncols*cols])];
         data = [data; reshape(img, [1 nrows*ncols])]; 
     end
     target = [target; repmat(i, [sz(i) 1])];
end
clear PADVAL PADSIZE scaling newsize scene;
clear img img_color nrows ncols cols;

%data_color = double(data_color);
data = double(data);
num_classes = n;
[traindata, y_train, testdata, y_test, mu, sigma, idx] = getbatches(data, target, sz, num_classes, num_test_per_class);
err = unit_test_batches_data(data, target, sz, idx, traindata, y_train, testdata, y_test, num_classes, num_test_per_class);

