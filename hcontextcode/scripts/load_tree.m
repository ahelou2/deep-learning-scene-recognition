if loc
    MODEL = [dataset '_' model '_with_location'];
else
    MODEL = [dataset '_' model '_binary_only'];
end

if gist
    MODEL = [MODEL '_with_gist'];
else
    MODEL = [MODEL '_without_gist'];
end

disp(MODEL)

switch model
    case 'cltree'
        load(priorModel)
    case 'indep'
        load(priorModelIndep)
end
load(measurementModel)
load(gistPredictions)

disp('loading detector outputs...')
load(detectorOutputs,'DdetectorTest')     
        
disp('loading groundtruth on test set...')        
load(groundTruth, 'Dtest')         

load(objectCategories) % Load object names and heights        



