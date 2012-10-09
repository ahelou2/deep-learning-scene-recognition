function [J,h] = computeJhMeas(obj_index, loc_measurements, detectionWindowLoc, correct_detection)

% Compute J and h for the measurement model given locations of candidate
% windows.
%
% PARAMETERS:
%       obj_index = object index of detected windows.
%       loc_measurements = locations (world coordinates) of detected
%       windows.
%       detectionWindowLoc = mean and variance of p(location of a detection | location of
%       the true median).
%       correct_detection = current estimate of correct detections
%
% OUTPUTS:
%       J = C'R^{-1}C where measurements = C*x + v
%       h = C'R*{-1}y
%
% Myung Jin Choi, MIT, 2009 November
% Modified in 2010 March

if(nargin < 4)
    correct_detection = ones(length(obj_index),1);
end

K = size(loc_measurements,2);
Nvar = size(detectionWindowLoc.meanCorrect,1);
Nmeas = length(obj_index);
C = sparse(Nmeas*K,Nvar*K);
meas_var = zeros(Nmeas*K,1);
for i=1:Nmeas
    obj = obj_index(i);
    i_K = K*(i-1)+1:K*i;
    obj_K = K*(obj-1)+1:K*obj;
    C(i_K,obj_K) = speye(K);
    if(correct_detection(i))
        loc_measurements(i,:) = loc_measurements(i,:) - detectionWindowLoc.meanCorrect(obj,:);
        meas_var(i_K) = detectionWindowLoc.varianceCorrect(obj,:);
    else
        loc_measurements(i,:) = loc_measurements(i,:) - detectionWindowLoc.meanFalse(obj,:);
        meas_var(i_K) = detectionWindowLoc.varianceFalse(obj,:);        
    end
end

Rinv = spdiags(1./meas_var,0,Nmeas*K,Nmeas*K);
J = C'*Rinv*C;
loc_measurements = loc_measurements';
h = sparse(C'*Rinv*loc_measurements(:));