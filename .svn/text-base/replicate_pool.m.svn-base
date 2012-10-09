function replicated_pool_mat = replicate_pool(pool_mat, C2D, N_H2D)

[numcases, ~] = size(pool_mat) ;
numhid = N_H2D(1)*N_H2D(2) ;
limits = N_H2D./C2D ;
pool_mat = reshape(pool_mat, [limits numcases]) ;

replicated_pool_mat = zeros([N_H2D numcases]) ;
for i=0:1:limits(1)-1
   for j=0:1:limits(2)-1 
        replicated_pool_mat(C2D(1)*i+1:C2D(1)*(i+1), C2D(2)*j+1:C2D(2)*(j+1),:) = ...
            repmat(pool_mat(i+1,j+1,:), C2D);
   end
end

replicated_pool_mat = reshape(permute(replicated_pool_mat, [3 1 2]), [1, numcases*numhid]) ;
replicated_pool_mat = reshape(replicated_pool_mat, [numcases numhid]) ;