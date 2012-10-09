function [hid_layer_out pool_layer_out hidden_states] = multinomial_exp(X, C2D, N_H2D)

%http://tipstrickshowtos.blogspot.com/2010/02/fast-replacement-for-sub2ind.
%html

[numcases, numhid] = size(X) ;
X_mat = reshape(X', [1 numcases*numhid]) ;
X_mat = reshape(X_mat, [N_H2D numcases]) ;
X_mat = exp(X_mat) ; 


limits = N_H2D./C2D ;
hidden_states = zeros(size(X_mat)) ;
denominator = zeros([limits numcases]) ;
denominators = zeros(size(X_mat)) ;
%denominators = ones(size(X_mat)) ;
for i=0:1:limits(1)-1
   for j=0:1:limits(2)-1 
       denominator(i+1,j+1,:) = ...
           sum(sum(X_mat(C2D(1)*i+1:C2D(1)*(i+1), C2D(2)*j+1:C2D(2)*(j+1),:))) ;
       denominators(C2D(1)*i+1:C2D(1)*(i+1), C2D(2)*j+1:C2D(2)*(j+1),:) = ...
           myrepmat(denominator(i+1,j+1,:), C2D, 3);
        portion = X_mat(C2D(1)*i+1:C2D(1)*(i+1), C2D(2)*j+1:C2D(2)*(j+1),:)./...
            (1 + denominators(C2D(1)*i+1:C2D(1)*(i+1), C2D(2)*j+1:C2D(2)*(j+1),:)) ;
         work_mat = portion ;
       work_mat = reshape(permute(work_mat, [3 1 2]), numcases, C2D(1)*C2D(2)) ;
       [work_mat IDX] = sort(work_mat,2) ;
       work_mat = cumsum(work_mat,2) ;
       thresh = rand(numcases,1) ;
       %thresh = repmat(thresh, 1, C2D(1)*C2D(2)) ;
       thresh = myrepmat(thresh, C2D(1)*C2D(2), 1) ;
       test = (-100*((work_mat - thresh) < 0)).*(work_mat - thresh) ;
       [val id] = min(test,[],2) ; 
       id = id(val == 0) ;
       non_zero_cases = (1:numcases)' ;
       non_zero_cases = non_zero_cases(val == 0) ;
       
       %id_1D = sub2ind([C2D(1)*C2D(2) numcases], IDX(1, id)', non_zero_cases) ;
       id_1D = mysub2ind2D([C2D(1)*C2D(2) numcases], IDX(1, id)', non_zero_cases) ;
       [I, J, K] = ind2sub([C2D numcases], id_1D) ; %test
       %sub = ind2sub2([C2D numcases], id_1D) ; %test
       %I = sub(:,1) ; J = sub(:,2) ; K = sub(:,3) ;
       %id_1D = sub2ind(size(hidden_states), C2D(1)*i + I, C2D(2)*j + J, K) ;
       id_1D = mysub2ind3D(size(hidden_states), C2D(1)*i + I, C2D(2)*j + J, K) ;
       hidden_states(id_1D) = 1 ; %test
   end
end

pool_layer_out = denominator./(1 + denominator) ;
num_pool_units = limits(1)*limits(2) ;
pool_layer_out = reshape(permute(pool_layer_out, [3 1 2]), [1, numcases*num_pool_units]) ;
pool_layer_out = reshape(pool_layer_out, [numcases num_pool_units]) ;

hid_layer_out = X_mat./(1 + denominators) ;

hid_layer_out = reshape(permute(hid_layer_out, [3 1 2]), [1, numcases*numhid]) ;
hid_layer_out = reshape(hid_layer_out, [numcases numhid]) ;

 hidden_states = reshape(permute(hidden_states, [3 1 2]), [1, numcases*numhid]) ; %test
 hidden_states = reshape(hidden_states, [numcases numhid]) ; %test
end

function isnan_isinf(X, name)
    if sum(sum(sum(isnan(X)))) || sum(sum(sum(isinf(X))))
        %X
       error('%s', name) ; 
       %error('fuck') ;
    end
end


