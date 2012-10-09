% Learn p(c_{ik} | b) = probability of correct detection when object is
% present/not present
% Learn p(c | s) = probability of correct detection based on the detector
% score (logistic function)

disp('Training window scores and probabilities of correct detections')
Nimages = length(DdetectorTraining);
Nobjects = length(names);

countsDetections = zeros(Nobjects);
countsCorrect = zeros(Nobjects);
countsImgTrueObj = zeros(Nobjects);
countsDetectionsInImgTrueObj = zeros(Nobjects);
countsCorrectInImgTrueObj = zeros(Nobjects);
countsKthCorrect = cell(Nobjects,1);
maxDetections = MaxNumDetections*ones(Nobjects,1);
for i=1:Nobjects
    countsKthCorrect{i} = zeros(1,maxDetections(i));
end

for n=1:Nimages    
    if(mod(n,100)==0)
        disp(n)
    end
    objects = {Dtraining(n).annotation.object.name};
    existObj = ismember(names,objects);
    countsImgTrueObj(existObj)  = countsImgTrueObj(existObj) + 1;
    
    objects = {DdetectorTraining(n).annotation.object.name};
    detections = {DdetectorTraining(n).annotation.object.detection};
    [foo,obj] = ismember(objects, names); obj = obj'; 
    [foo,correct] = ismember(detections,'correct'); 
    
    valid = find(obj>0);
    obj = obj(valid);
    correct = correct(valid);
    unique_obj = unique(obj);
    
    for oi=1:length(unique_obj);
        o = unique_obj(oi);
        windows = find(obj==o);
        numDetections = length(windows);
        if(numDetections > maxDetections(o))
            fprintf('  Warning: the maximum number of candidate windows is %d, which is larger than the expected value %d\n',numDetections,maxDetections(o));
            countsKthCorrect{o} = [countsKthCorrect{o} zeros(1,numDetections-maxDetections(o))];
            maxDetections(o) = numDetections;
        end
        countsDetections(o) = countsDetections(o)+numDetections;
        correctDetections = correct(windows);
        countsKthCorrect{o}(1:numDetections) = countsKthCorrect{o}(1:numDetections) + correctDetections;
        countsCorrect(o) = countsCorrect(o) + sum(correctDetections);
        if(existObj(o))
            countsCorrectInImgTrueObj(o) = countsCorrectInImgTrueObj(o) + sum(correctDetections);
            countsDetectionsInImgTrueObj(o) = countsDetectionsInImgTrueObj(o) + numDetections;
        end
    end
end

clear windowScore

windowScore.name = names;
windowScore.countsCorrectDetections = countsCorrect;
windowScore.countsFalseAlarms = countsDetections - countsCorrect;
windowScore.maxCandWindows = maxDetections;

windowScore.pCorrect = countsCorrect ./ countsDetections;
windowScore.pCorrectGivenObjectPresent = countsCorrectInImgTrueObj./ countsDetectionsInImgTrueObj;

for i=1:Nobjects
    windowScore.pKthCorrectGivenObjectPresent{i} = (countsKthCorrect{i}+1)/(countsImgTrueObj(i)+maxDetections(i));
    [tf,j] = ismember(names{i},validcategories);
    if(tf)
        windowScore.logitCoef{i} = logitCoef{j};
    else
        ferror('Cannot find the logistic coefficients for %s\n',names{i});
    end 
end