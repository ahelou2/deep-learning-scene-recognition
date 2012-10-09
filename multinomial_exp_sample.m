function hidden_states = multinomial_exp_sample(X, C2D, N_H2D)

% TEST 1
% C2D = [2 2] ;
% N_H2D = [4 4] ;
% X = 1/4*ones(3, 16) ;

% TEST 2
% C2D = [2 2] ;
% N_H2D = [4 4] ;
% X = 1/1000*ones(3, 16) ;

% TEST 3
% C2D = [2 2] ;
% N_H2D = [4 4] ;
% X = 1/1000*ones(3, 16) ;
% X(1,:) = 1/4*ones(1, 16) ;


[numcases, numhid] = size(X) ;
X_mat = reshape(X', [1 numcases*numhid]) ;
X_mat = reshape(X_mat, [N_H2D numcases]) ;

limits = N_H2D./C2D ;
hidden_states = zeros(size(X_mat)) ;
for i=0:1:limits(1)-1
   for j=0:1:limits(2)-1 
       portion = ...
           X_mat(C2D(1)*i+1:C2D(1)*(i+1), C2D(2)*j+1:C2D(2)*(j+1),:) ;
       work_mat = portion ;
       %work_mat = reshape(work_mat, numcases, C2D(1)*C2D(2)) ;
       work_mat = reshape(permute(work_mat, [3 1 2]), numcases, C2D(1)*C2D(2)) ;
       [work_mat IDX] = sort(work_mat,2) ;
       work_mat = cumsum(work_mat,2) ;
       thresh = rand(numcases,1) ;
       thresh = repmat(thresh, 1, C2D(1)*C2D(2)) ;
       test = (-100*((work_mat - thresh) < 0)).*(work_mat - thresh) ;
       [val id] = min(test,[],2) ; 
       id = id(val == 0) ;
       non_zero_cases = (1:numcases)' ;
       non_zero_cases = non_zero_cases(val == 0) ;
       %id_1D = sub2ind([numcases C2D(1)*C2D(2)], (1:numcases)', IDX(1, id)') ;
       %id_1D = sub2ind([C2D(1)*C2D(2) numcases], IDX(1, id)', (1:numcases)') ;
       id_1D = sub2ind([C2D(1)*C2D(2) numcases], IDX(1, id)', non_zero_cases) ;
       %[I, J, K] = ind2sub([C2D numcases], [(1:numcases)' IDX(1, id)']) ; %test
       [I, J, K] = ind2sub([C2D numcases], id_1D) ; %test
       id_1D = sub2ind(size(hidden_states), C2D(1)*i + I, C2D(2)*j + J, K) ;
       %hidden_states(C2D(1)*i + I, C2D(2)*j + J, K) = 1 ; %test
       hidden_states(id_1D) = 1 ; %test
   end
end

 hidden_states = reshape(permute(hidden_states, [3 1 2]), [1, numcases*numhid]) ; %test
 hidden_states = reshape(hidden_states, [numcases numhid]) ; %test



