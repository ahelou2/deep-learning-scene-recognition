function Ddetector = detectors2LMstruct(D, HOMERESULTS, validcategories, MaxNumDetections, mn)

% load the detector outputs. Build struct and table

nFiles = length(D);
Nobjects = length(validcategories);
clear Ddetector 
Nbb = 0; Nbbo=0;
for n = 1:nFiles
    filename = sprintf('%d.mat', n);
    
    % get image size
    [ncols nrows] = getaproximagesize(D(n).annotation);
    
    % loop on objects
    Ddetector(n).annotation.filename = D(n).annotation.filename;
    Ddetector(n).annotation.folder =  D(n).annotation.folder;
    Ddetector(n).annotation.imagesize = D(n).annotation.imagesize;
    
    for i = 1:Nobjects
        fileresults = fullfile(HOMERESULTS, validcategories{i}, filename);
        if exist(fileresults, 'file')
            boxClass = validcategories{i};
            clear det1
            load (fileresults); det = det1;
            if ~isempty(det)
                nrows = Ddetector(n).annotation.imagesize.nrows;
                ncols = Ddetector(n).annotation.imagesize.ncols;
                
                boundingBox = det(:,[1 3 2 4]); % Each row is boundingBox(i,:) = [xmin xmax ymin ymax]
                boxScores = det(:,5);
                
                areaobj = abs((boundingBox(:,2)-boundingBox(:,1)).*(boundingBox(:,4)-boundingBox(:,3)))/(nrows*ncols);                
                % prune by size
                k = find(areaobj>mn(i));
                boundingBox = boundingBox(k,:); 
                boxScores = boxScores(k);
                
                % prune by score
                [foo, k] = sort(boxScores, 'descend');
                k = k(1:min(MaxNumDetections, length(k)));
                
                Nbb = Nbb+length(k);
                Nbbo = Nbbo+length(boxScores);
                disp(round([n Nbb Nbbo Nbb/n]))
                
                Ddetector(n).annotation = addDetector2LM(uint16(boundingBox(k,:)), boxScores(k), boxClass, Ddetector(n).annotation);
            end
        end
    end
end

