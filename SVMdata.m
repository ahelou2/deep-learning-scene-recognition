function SVMdata(traindata, trainlabels, testdata, testlabels, trainfile, testfile)
[numSamples numFeatures] = size(traindata);
numTests = size(testdata,1);
train = double([reshape(trainlabels, numSamples, 1) traindata]);
save(trainfile, 'train', '-ascii');
test = double([reshape(testlabels, numTests, 1) testdata]);
save(testfile, 'test', '-ascii');
%file = fopen(trainfile, 'w');
%for i=1:numSamples
%    fprintf(file, '%d', trainlabels(i));
%    for j=1:numFeatures
%        fprintf(file, ' %d:%f', j, traindata(i,j));
%    end
%    fprintf(file, '\n');
%end
%fclose(file);
%file = fopen(testfile, 'w');
%for i=1:numTests
%    fprintf(file,'%d', testlabels(i));
%    for j=1:numFeatures
%        fprintf(file, ' %d:%f', j, testdata(i,j));
%    end
%    fprintf(file, '\n');
%end
%fclose(file);
