% CREATE TRAINING and TEST splits
clear all
close all

configure

tagsFile = 'tagsSUN.txt';
sceneFolder = 'static_sun09_database';
objectFolder = 'static_sun_objects';

% 1) build struct
D = LMdatabase(HOMEANNOTATIONS);
Doutofcontext =  LMquery(D, 'folder', 'out_of_context');
D = LMquery(D, 'folder', '-out_of_context');

Ds = LMquery(D, 'folder', sceneFolder);
[foo, mt] = LMquery(Ds, 'object.polygon.username', 'mt_');

% 2) create two structs: scenes and objects
DtrainingObjects = LMquery(D, 'folder', objectFolder);
test = deleteObjectsOutsideFrame(DtrainingObjects, HOMEIMAGES);
DtrainingObjects = LMvalidobjects(test);
DtrainingObjects = [DtrainingObjects Ds(mt)];

% script that does all the replacementes of object names for MT
DtrainingObjects = cleanMTlabels(DtrainingObjects);

Dscenes = Ds(setdiff(1:length(Ds), mt));

% 3) images with less than 90% area labeled only for training objects. 
relativearea = LMlabeledarea(Dscenes);
good = find(relativearea>.9);

DtrainingObjects = [DtrainingObjects Dscenes(setdiff(1:length(Dscenes), good))];
Dscenes = Dscenes(good);

% 4) remove crop words and combine synonyms
DtrainingObjects = addcroplabel(DtrainingObjects);
DtrainingObjects = LMaddtags(DtrainingObjects, tagsFile, 'unmatched');

Dscenes = addcroplabel(Dscenes);
Dscenes = LMaddtags(Dscenes, tagsFile, 'unmatched');

Doutofcontext = addcroplabel(Doutofcontext);
Doutofcontext = LMaddtags(Doutofcontext, tagsFile, 'unmatched');

% 5) create training/test splits
% split folders equaly
Ntraining = .5; % percentage images used for training
scenes = folder2class(Dscenes);
train = []; test = []; validation = [];
for i = 1:max(scenes)
    j = find(scenes==i);
    N = length(j);
    j = j(randperm(N));
    
    train = [train j(1:fix(N*Ntraining+rand))];
end
test = setdiff(1:length(Dscenes), train);

DtrainingScenes = Dscenes(train);    % training scenes
DtestScenes = Dscenes(test);         % test scenes

counts = LMcountobject(DtrainingObjects);
j = find(counts>0);
DtrainingObjects = DtrainingObjects(j);

%[names, counts] = LMobjectnames(LMquery([DtrainingObjects DtrainingScenes], 'object.name', '-unmatched'));
[names, counts] = LMobjectnames(LMquery([DtrainingScenes], 'object.name', '-unmatched'));
[foo,j] = sort(counts, 'descend');
categories.names = names(j);
categories.counts = counts(j);

DtestScenes = addimagesize(DtestScenes, HOMEIMAGES);
DtrainingScenes = addimagesize(DtrainingScenes, HOMEIMAGES);
Doutofcontext = addimagesize(Doutofcontext, HOMEIMAGES);

Dtest = DtestScenes;
Dtraining = DtrainingScenes;

%save(groundTruth,'DtrainingObjects','Dtraining','Dtest','Doutofcontext','categories')

%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP 3: Show statistics of objects in DtrainingObjects and
% DtrainingScenes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% remove unmatched object class:
DtrainingObjects = LMquery(DtrainingObjects, 'object.name', '-unmatched');
DtrainingScenes = LMquery(DtrainingScenes, 'object.name', '-unmatched');

% collect stats
[names_allobjects, counts_allobjects] = LMobjectnames([DtrainingObjects DtrainingScenes]);
[counts_allobjects, j] = sort(counts_allobjects, 'descend');
names_allobjects = names_allobjects(j);

[names_objects, counts_objects] = LMobjectnames(DtrainingObjects);
[counts_objects, j] = sort(counts_objects, 'descend');
names_objects = names_objects(j);

[names_mtobjects, counts_mtobjects] = LMobjectnames(LMquery(DtrainingObjects, 'folder', 'static_sun_objects'));
[counts_mtobjects, j] = sort(counts_mtobjects, 'descend');
names_mtobjects = names_mtobjects(j);

[names_scenes, counts_scenes] = LMobjectnames(DtrainingScenes);
[counts_scenes, j] = sort(counts_scenes, 'descend');
names_scenes = names_scenes(j);

figure
loglog(counts_scenes, 'g')
hold on
loglog(counts_mtobjects, 'r');
loglog(counts_allobjects, 'k');
axis('tight')
