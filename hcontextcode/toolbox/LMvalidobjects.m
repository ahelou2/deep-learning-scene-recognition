function D = LMvalidobjects(D, objectlist, method)
%
% Remove from the index struct all the objects that are deleted and not in
% the list of objects.
%
% D = LMvalidobjects(D, objectlist); 
%
% D = LMvalidobjects(D, objectlist, 'exact'); 
%
% Example:
%    1) Removing labels
%    Dcp = LMvalidobjects(D, 'car,person');
%
%    The output Dcp is a struct in which only the polygons corresponding to
%    cars and people are preserved. This is very similar to using LMquery.
%    The difference is that LMquery will return a struct with the images
%    that contain the objects and it will remove from the struct the images
%    that do not contain any of the objects. On the other hand,
%    LMvalidobjects will only delete the objects not in the list. But the output struct
%    will still contain the pointers to the images not containing any of
%    the objects. 
%    
%    Compare the outputs of
%          D1 = LMvalidobjects(D, 'car,person');
%          D2 = LMquery(D, 'object.name', 'car,person');
%
%    D1 will contain more images than D2.
%      
%    2) Removing deleted files
%    
%    D = LMvalidobjects(D);
%
%    Removes from the dataset all the deleted objects


if nargin == 2;
    method = 'search';
end

if nargin>1
    query = parseContent(objectlist);
end

for i = 1:length(D);
    annotation = D(i).annotation;
        
    if isfield(annotation, 'object')
        if ~isempty(annotation.object)
            if isfield(annotation.object, 'deleted')
                if ischar(annotation.object(1).deleted)
                    j = strmatch('0',{annotation.object.deleted});
                else
                    j = find([annotation.object.deleted]==0);
                end
                % remove object with empty names
                j = intersect(find(~ismember({annotation.object(:).name}, {''})), j);
            else
                %j = 1:length(annotation.object);
                % select only objects with non-empty names
                j = find(~ismember({annotation.object(:).name}, {''}));
            end
            
            %jc = [];
            %for m = 1:length(annotation.object)
            %    if ~isempty(annotation.object(m).name)
            %        jc = [jc m];
            %    end
            %end
            
            % if there is a query, select relevant objects
            if nargin>1
                %if strcmp(method, 'exact')
                %    jc = strmatch(query{1}, {annotation.object(:).name}, 'exact');
                %else
                    jc = findobject({annotation.object(j).name}, query, method); % find object index
                %end
                %j = intersect(jc, j);
                j = j(jc);
            end
            
            %annotation.object = rmfield(annotation.object, 'deleted');
            if ~isempty(j)
                annotation.object = annotation.object(j);
            else
                % if all the objects are deleted, remove the object field
                annotation = rmfield(annotation, 'object');
            end            
        else
            % if the object field was already empty, remove it.
            annotation = rmfield(annotation, 'object');
        end
        D(i).annotation = annotation;
    end
end


