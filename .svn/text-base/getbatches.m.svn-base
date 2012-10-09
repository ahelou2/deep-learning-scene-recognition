function [traindata, y_train, testdata, y_test, mu, sigma, idx] = getbatches(data, target, sz, num_classes, num_test_per_class)
n = min(sz);
m = max(sz);
idx = randperm(size(target,1));
data = data(idx, :);
target = target(idx, :);
[data, mu, sigma] = normalize(double(data));
% group the data into num_classes groups
for j=1:num_classes
   d{j}.images = data(target==j,:);
   d{j}.values = target(target==j);
end
%get all num_test_per_class testing images for each class
[num_samples, num_dims] = size(data);
testdata = zeros(num_test_per_class*num_classes, num_dims);
y_test = zeros(num_test_per_class*num_classes,1);
id = 0;
for i=1:num_test_per_class
    for j=1:num_classes
        id = id + 1;
        testdata(id,:) = d{j}.images(i,:);
        y_test(id) = d{j}.values(i);
    end
end

leftover = num_samples - num_test_per_class*num_classes;

traindata = zeros(leftover, num_dims);
y_train = zeros(leftover, 1);
id = 0;
for i=num_test_per_class+1:n
    for j=1:num_classes
        id = id + 1;
        traindata(id,:) = d{j}.images(i,:);
        y_train(id) = d{j}.values(i);
    end
end

v=1:num_classes;
i=n+1;
while size(v,2) > 0
    v(sz(v)<i) = [];
    for k=1:size(v,2)
        j = v(k);
        id = id + 1;
        traindata(id,:) = d{j}.images(i,:);
        y_train(id) = d{j}.values(i);
    end
    i=i+1;
end

