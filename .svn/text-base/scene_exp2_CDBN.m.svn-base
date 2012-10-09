% This is running on normalized Torralba outdoor datasets with the number
% of training examples maximized. 800 test examples. 100*100 images.

train = traindata ;
op1.verbose = true ;
op1.gaussian_vis = true ;
op1.eta = 0.00001 ;
op1.batchsize = 8*14 ;
op1.maxepoch = 20 ;
op1.batch_perm = false ;
op1.simple_sparsification = true ;
K1 = 10 ;
N_V2D1 = [100 100] ;
N_W2D1 = [11 11] ;
% N_H2D1 = [90 90] ;
C2D1 = [2 2] ;

model = cell(3,1) ;
model{1} = crbm2D(train, N_V2D1, N_W2D1, K1, C2D1, op1) ;

op2 = op1 ;
op2.gaussian_vis = false ;
op2.eta = op1.eta*10 ;
op2.sparse_bias_val = -1.5 ;
K2 = 20 ;
N_V2D2 = model{1}.N_V2D_next ;
N_W2D2 = [10 10] ;
% N_H2D2 = [36 36] ;
C2D2 = [2 2] ;

model{2} = crbm2D(model{1}.pool_top, N_V2D2, N_W2D2, K2, C2D2, op2) ;

op3 = op2 ;
%op3.simple_sparsification = false ;
op2.sparse_bias_val = -1 ;
K3 = 30 ;
N_V2D3 = model{2}.N_V2D_next ;
N_W2D3 = [9 9] ;
% N_H2D1 = [10 10] ;
C2D3 = [2 2] ;

model{3} = crbm2D(model{2}.pool_top, N_V2D3, N_W2D3, K3, C2D3, op3) ;


