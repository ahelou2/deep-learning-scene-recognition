
% Script that reads the files with output of the detector and creates the
% labelme structs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
configure
HOMERESULTS = '/csail/vision-torralba/projects/Hcontext/Boxes';
objHeightsFile = 'objectHeights';

% load structs
load(groundTruth)

% Number of objects
Nobjects = length(categories.names);
MaxNumDetections = 10;

% add image size
if(~isfield(Dtest(1).annotation,'imagesize'))
    Dtest = addimagesize(Dtest, HOMEIMAGES);
    Dtraining = addimagesize(Dtraining, HOMEIMAGES);
end

% count how many times each object appears on the test set
[namestest, countstest] = LMobjectnames(Dtest);

% get list of valid categories
validcategories = []; bad = [];
for i = 1:Nobjects
        a = dir(fullfile(HOMERESULTS, 'test', categories.names{i}, '*.mat'));
        j = strmatch(categories.names{i}, namestest, 'exact');
        if length({a.name})==length(Dtest) && ~isempty(j) && (countstest(j)>15)
             validcategories = [validcategories categories.names(i)];
        else
            bad = [bad categories.names(i)];
        end
end
validcategories = sort(validcategories);
%%
% look on the training set what is the smallest detectable object for each
% class (in percentage with respect to image size).
[objectnames, instancecounts, areacounts, minarea] = LMobjectstats(Dtraining, HOMEIMAGES);

Nobjects = length(validcategories);

clear mn
for i = 1:Nobjects
    c = strmatch(validcategories{i}, objectnames, 'exact');
    j = find(instancecounts(c, :)>0);
    mn(i) = min(full(minarea(c,j)));
end
mn = max(0.0001, mn);

disp('Loading the detector outputs of the test set')
DdetectorTest = detectors2LMstruct(Dtest, [HOMERESULTS '/test'], validcategories, MaxNumDetections, mn);

disp('Loading the detector outputs of the training set')
DdetectorTraining = detectors2LMstruct(Dtraining, [HOMERESULTS '/train'], validcategories, MaxNumDetections, mn);


% Insert 'detection' field indicating whether the window is a correct
% detection or not 
countCorrectDetections = zeros(length(validcategories),1);
logitCoef = cell(length(validcategories),1);
for n = 1:length(validcategories)
    fprintf('%d:%s\n',n,validcategories{n})
    [recall, precision, DdetectorTraining, threholds, score, correct, ap] = LMrecallPrecision(Dtraining, DdetectorTraining, validcategories{n}, 'nomisses');
    countCorrectDetections(n) = sum(correct==1);

    % Train a logistic function for p(correct detection | detector score)
    if length(score)>5
        num_trials = 1;
        score = [-50*num_trials score 50*num_trials];
        correct = [0 correct 1];        
        [logitCoef{n},dev] = glmfit(score', [correct'==1], 'binomial', 'link', 'logit');
        while(sum(abs(logitCoef{n})) > 1e8)
            num_trials = num_trials+1;
            if(num_trials > 10)
                fprintf('  Failed to train %d:%s...\n',n,validcategories{n})
                break;
            end
            fprintf('  Numerical issues in logistic coefficienst. Number of trials %d..\n',num_trials)
            score = [-50*num_trials score 50*num_trials];
            correct = [0 correct 1];        
            [logitCoef{n},dev] = glmfit(score', [correct'==1], 'binomial', 'link', 'logit');
        end
    end
    [recall, precision, DdetectorTest, threholds, score, correct, ap] = LMrecallPrecision(Dtest, DdetectorTest, validcategories{n}, 'nomisses');
end

% remove classes with less than 4 correct detections
wrong = (countCorrectDetections<4);

wrongClasses = validcategories(wrong);
D1 = DdetectorTest;
D2 = DdetectorTraining;
for n = 1:length(wrongClasses)
    fprintf('Removing object category %s\n',wrongClasses{n})
    D1 = LMvalidobjects(D1, ['-' wrongClasses{n}], 'exact');
    D2 = LMvalidobjects(D2, ['-' wrongClasses{n}], 'exact');
end
DdetectorTest = D1;
DdetectorTraining = D2;

validcategories = validcategories(~wrong);
logitCoef = logitCoef(~wrong);

save(detectorOutputs,'DdetectorTest','DdetectorTraining','validcategories','logitCoef','MaxNumDetections')

% Save the list of validcategories and physical heights
names = validcategories;
load(objHeightsFile);
heights = zeros(length(names),1);
for n=1:length(names)
    [tf,ind] = ismember(names{n},physicalSize.categories);
    if(tf)
        heights(n) = physicalSize.heights(ind);
    else
        fprintf('Need to add a height for %d:%s\n',n,names{n});
        heights(n) = 1;
    end
end

[tf, root] = ismember('sky',names); % Root of the tree
if(~tf)
    ferror('The root of the tree is not defined\n');
end

save(objectCategories,'names','heights','root');

    