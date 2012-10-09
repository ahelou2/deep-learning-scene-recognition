function getObjects(database, objects, HOMEIMAGES, HOMEANNOTATIONS, height, width, prefix)
% Input:
% database: labelme database from LMdatabase
% objects: the cells of object names
% height: number of rows in the output image
% width: number of columns in the output image
% prefix: output directory
% Output
%    - for each object class, it creates a directory, save the all
%      images contain that class of object.
numObjects = size(objects,1);
for i=1:1:numObjects
    mkdir([prefix '/' objects{i}]);
    [obj, idx] = LMquery(database, 'object.name', objects{i});
    num = size(obj, 2);
    for j=1:1:num
        images.img = imread([HOMEIMAGES '/' obj(j).annotation.folder '/' obj(j).annotation.filename]);
        if size(images.img) > 2
            images.img = grb2gray(images.img);
        end
        [obj(j).annotation, images.img] = LMimpad(obj(j).annotation, images.img, [height width]);
        img = double(images.img);
        [mask, class] = LMobjectmask(obj(j).annotation, HOMEIMAGES, objects{i});
        n = size(class,2);
        images.img = rgb2gray(imread([HOMEIMAGES '/' obj(j).annotation.folder '/' obj(j).annotation.filename]));
        n = size(obj(j).annotation.object,2);
        for k=1:1:n
            images.objects{k}.name = obj(j).annotation.object(k).name;
            images.objects{k}.mask = unit8(img.*mask(:,:,k));
        end
        filename = char(strcat(prefix, '/', objects{i}, '/', obj(j).annotation.filename, '.mat'));
        save(filename, '-struct', 'images');
    end
end
