function err = unit_test_batches_data(data, target, sz, idx, traindata, y_train, testdata, y_test, num_classes, num_test_per_classes)
n = min(sz)-num_test_per_classes;
err = 0;
for i=1:num_test_per_classes
    j=(1:num_classes)';
    err = err + power(sum(y_test((i-1)*num_classes+1:i*num_classes) - j), 2);
end

for i=1:n
    j=(1:num_classes)';
    err = err + power(sum(y_train((i-1)*num_classes+1:i*num_classes) - j), 2);
end



[permuted_data, a, b] = normalize(double(data(idx,:)));
permuted_target = double(target(idx));

classes = zeros(num_classes, size(target,1));
for i=1:num_classes
    class_inx = 1:size(data,1);
    class_inx(permuted_target~=i) = [];
    test_inx = i:num_classes:num_test_per_classes*num_classes;
    train_inx = i:num_classes:n*num_classes;
    err = err + sum(permuted_target(class_inx(1:num_test_per_classes)) - y_test(test_inx))
    err = err + sum(sum(permuted_data(class_inx(1:num_test_per_classes)) - testdata(test_inx)))
    err = err + sum(permuted_target(class_inx(num_test_per_classes+1:n+num_test_per_classes)) - y_train(train_inx))
    err = err + sum(sum(permuted_data(class_inx(num_test_per_classes+1:n+num_test_per_classes)) - traindata(train_inx)))
end
