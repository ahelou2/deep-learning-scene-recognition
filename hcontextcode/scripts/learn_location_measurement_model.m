
D = DdetectorTraining;
clear DdetectorTraining;

diffSqCorrect = zeros(N,K);
diffSqFalse = zeros(N,K);
diffCorrect = zeros(N,K);
diffFalse = zeros(N,K);
Ncorrect = zeros(N,1);
Nfalse = zeros(N,1);
for m=1:M
    if(mod(m,10)==0)
        disp(m)
    end
    [foo,true_obj] = ismember({Dtraining(m).annotation.object.name},names); 
    unique_true_obj = setdiff(unique(true_obj'),0);
    image_size(1) = Dtraining(m).annotation.imagesize.ncols;
    image_size(2) = Dtraining(m).annotation.imagesize.nrows;                
    [true_loc_index,true_loc] = getWindowLoc(Dtraining(m).annotation.object,names,image_size,heights);
    
    [foo, obj] = ismember({D(m).annotation.object.name},names(unique_true_obj));
    valid = (obj>0);
    [loc_index,loc_measurements] = getWindowLoc(D(m).annotation.object(valid),names,image_size,heights);        
    isCorrect = ismember({D(m).annotation.object(valid).detection}, {'correct'});

    for o=1:length(unique_true_obj)
        n = unique_true_obj(o);
        obj_n = (loc_index==n);
        med_true_loc = median(true_loc(true_loc_index==n,:),1);
            
        isCorrect_n = isCorrect(obj_n);
        Ncorrect(n) = Ncorrect(n) + sum(isCorrect_n);
        Nfalse(n) = Nfalse(n) + sum(~isCorrect_n);
        
        relativeWindowLoc = loc_measurements(obj_n,:) - repmat(med_true_loc,sum(obj_n),1);
        diffSqCorrect(n,:) = diffSqCorrect(n,:) + sum(relativeWindowLoc(isCorrect_n,:).^2,1);
        diffSqFalse(n,:) = diffSqFalse(n,:) + sum(relativeWindowLoc(~isCorrect_n,:).^2,1);
        diffCorrect(n,:) = diffCorrect(n,:) + sum(relativeWindowLoc(isCorrect_n,:),1);
        diffFalse(n,:) = diffFalse(n,:) + sum(relativeWindowLoc(~isCorrect_n,:),1);
    end
end

detectionWindowLoc.Ncorrect = Ncorrect;
detectionWindowLoc.Nfalse = Nfalse;
detectionWindowLoc.meanCorrect = diffCorrect./repmat(Ncorrect,1,K);
detectionWindowLoc.meanFalse = diffFalse./repmat(Nfalse,1,K);
detectionWindowLoc.varianceCorrect = diffSqCorrect./repmat(Ncorrect,1,K)-detectionWindowLoc.meanCorrect.^2;
detectionWindowLoc.varianceFalse = diffSqFalse./repmat(Nfalse,1,K)-detectionWindowLoc.meanFalse.^2;



