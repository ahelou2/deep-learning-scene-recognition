function adjmat = ChowLiu(mi,names)

% Learn a tree using Chow-liu algorithm

N = size(mi,1);
if(nargin < 2)
    names = mat2cell(num2str([1:N]'),ones(N,1),1);
end

entropy = diag(mi)/2;
mi = mi - diag(entropy*2);
[mi_sorted, index] = sort(mi(:), 'descend');

N = size(mi,1);
adjmat = zeros(N);
num_edge=0;
i=1;
while(num_edge < N-1)
    [sub1,sub2] = ind2sub([N,N],index(i));
    if(~connected(adjmat,sub1,sub2))
        name1 = mat2str(cell2mat(names(sub1)));
        name2 = mat2str(cell2mat(names(sub2)));
        fprintf('%d %s %d %s\n', sub1, name1, sub2, name2);
        adjmat(sub1,sub2) = 1;
        adjmat(sub2,sub1) = 1;
        num_edge = num_edge+1;
    else
        name1 = mat2str(cell2mat(names(sub1)));
        name2 = mat2str(cell2mat(names(sub2)));
        %fprintf('*** %d %s %d %s\n', sub1, name1, sub2, name2);        
    end
    i = i+1;
end
