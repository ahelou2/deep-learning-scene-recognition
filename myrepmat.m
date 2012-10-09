function Z = myrepmat(A, Y, dims)

%http://www.psi.toronto.edu/~vincent/matlabindexrepmat.html

rowIdx = [1 : size(A,1)]';


if dims == 1
    Z = A(rowIdx(:, ones(Y(1),1))) ;
elseif dims == 2
    colIdx = [1 : size(A,2)]';
     Z = A(rowIdx(:, ones(Y(1),1)), colIdx(:, ones(Y(2),1))) ;
elseif dims == 3
       colIdx = [1 : size(A,2)]';
       D3Idx = [1 : size(A,3)]';
     Z = A(rowIdx(:, ones(Y(1),1)), colIdx(:, ones(Y(2),1)), D3Idx(:, ones(1,1))) ; 
end
