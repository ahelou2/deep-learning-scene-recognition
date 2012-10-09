function [med_obs_coords,avg_obs_coords,instance_var, num_instances] = locTransform(N, locations, loc_indcs, heights, image_size)

% Transforms the location and scale samples to observer centric
% coordinates.  For an object that appears multiple times in an image, take
% the average of all instances.  Computes the physical height of each
% object by taking the average of all instances.
%
% PARAMETERS:
%       N = number of objct categories
%       locations{m} = [cx,cy,sx,sy] of a bounding box for each instance
%       loc_indcs{m} = Object index corresponding to each location sample
%       heights(i) = real height of object i
%       image_size(m,:) = [width, height] of each image
%
% OUTPUTS:
%       med_obs_coords{m}(i,1:2) = observer centric coordinates (y,z) of
%           the i'th object (sorted by index) image m.  Median of all
%           instances.
%       instance_var(i,:) = variance of y,z of multiple instances
%           relative to its mean for each object i.
%       
% Myung Jin Choi, MIT, 2009 October

M = length(locations); % # samples
K = 2; % dimensions of the location variable

med_obs_coords = zeros(M,N,K);
avg_obs_coords = zeros(M,N,K);
num_instances = zeros(N,1);
instance_var = zeros(N,K);

for m=1:M
    loc_index_list = loc_indcs{m};  % Index of objects in this image
    if(isempty(loc_index_list))
        continue;
    end
    loc_index = unique(loc_index_list);    
    location = locations{m}; 
    
    obs_coords = image2WorldCoords(location(:,[1,2,4]),image_size(m,:),heights(loc_index_list));
    obs_coords(:,1) = [];  % Drop the x-coordinates
    for o=1:length(loc_index)
        obj = loc_index(o);
        obj_instances = (loc_index_list==obj);
        avg_obs_coords_m = mean(obs_coords(obj_instances,:),1);
        avg_obs_coords(m,obj,:) = avg_obs_coords_m;
        med_obs_coords(m,obj,:) = median(obs_coords(obj_instances,:),1);
        diff_sq = (obs_coords(obj_instances,:)-repmat(avg_obs_coords_m,sum(obj_instances),1)).^2;
        num_instances(obj) = num_instances(obj) + sum(obj_instances);
        instance_var(obj,:) = instance_var(obj,:) + sum(diff_sq,1);
    end  
end

instance_var = instance_var ./ repmat(num_instances,1,K);