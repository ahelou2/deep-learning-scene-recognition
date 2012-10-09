function [loc_index,obj_coords,location,valid] = getWindowLoc(objects,names,image_size,heights)

N = length(objects);
valid = false(N,1);
loc_index = zeros(N,1);
location = zeros(N,4);
for o=1:length(objects)
    obj_label = objects(o).name;
    [tf, var_index] = ismember(obj_label, names);
    if (tf)
        valid(o) = true;
        loc_index(o) = var_index;
        [X,Y] = getLMpolygon(objects(o).polygon);
        cx = (min(X)+max(X))/2; cy = (min(Y)+max(Y))/2;
        sx = max(X)-min(X); sy = max(Y)-min(Y);
        location(o,:) = [cx cy sx sy];
    end
end

loc_index(~valid) = [];
location(~valid,:) = [];
obj_coords = image2WorldCoords(location(:,[1,2,4]),image_size,heights(loc_index));
obj_coords = obj_coords(:,2:3);