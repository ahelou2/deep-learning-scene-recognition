% Load pascal dataset
clear all

% File name to save the pascal dataset structure
groundTruth = '/csail/vision-torralba/projects/Hcontext/codes/cvpr10Public/data/pascal_groundTruth';

% Folders with the PASCAL images and annotations
HOME = '/csail/vision-torralba/datasets/PASCAL/VOCdevkit';
HOMEANNOTATIONS = fullfile(HOME, 'VOC2007/Annotations');
HOMEIMAGES = fullfile(HOME, 'VOC2007/JPEGImages');

addpath(fullfile(HOME, 'VOCcode'))

files = dir(HOMEANNOTATIONS);
files = {files(:).name};
files = setdiff(files, {'.', '..'});

clear D
for i = 1:length(files)
    i
    res = VOCreadxml(fullfile(HOMEANNOTATIONS, files{i}));
    for n = 1:length(res.annotation.object)
        bb = res.annotation.object(n).bndbox;
        x = str2num([bb.xmin ' ' bb.xmax ' ' bb.xmax ' ' bb.xmin ' ' bb.xmin]);
        y = str2num([bb.ymin ' ' bb.ymin ' ' bb.ymax ' ' bb.ymax ' ' bb.ymin]);
        
        res.annotation.object(n).polygon = setLMpolygon(x',y');
        res.annotation.object(n).deleted = '0';
    end
    D(i) = res;
    D(i).annotation.folder = '';
end


% load test and training indices
train = textread(fullfile(HOME, 'VOC2007/ImageSets/Main/train.txt'), '%d');
test = textread(fullfile(HOME, 'VOC2007/ImageSets/Main/test.txt'), '%d');
val = textread(fullfile(HOME, 'VOC2007/ImageSets/Main/val.txt'), '%d');


classes2007={...
        'aeroplane'
        'bicycle'
        'bird'
        'boat'
        'bottle'
        'bus'
        'car'
        'cat'
        'chair'
        'cow'
        'diningtable'
        'dog'
        'horse'
        'motorbike'
        'person'
        'pottedplant'
        'sheep'
        'sofa'
        'train'
        'tvmonitor'};

validcategories = cell(1,length(classes2007));
for n = 1:length(classes2007)
    validcategories{n} = ['pascal_' classes2007{n}];
    D = LMreplaceobjectname(D, classes2007{n}, validcategories{n});
end

% create splits and save
Dtraining = D(train);
Dtest = D(test);
Dvalidation = D(val);

 save(groundTruth,'HOMEIMAGES','HOMEANNOTATIONS','classes2007','validcategories','Dtraining','Dtest','Dvalidation')
 
 LMstats(D, HOMEIMAGES)
 
 