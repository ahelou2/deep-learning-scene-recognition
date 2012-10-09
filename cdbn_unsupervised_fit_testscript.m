train = air_motor_train_norm ;
op1.verbose = true ;
op1.gaussian_vis = true ;
op1.eta = 0.00005 ;
op1.batchsize = 80 ;
op1.maxepoch = 12 ;
K1 = 2 ;
N_V2D1 = [50 75] ;
N_W2D1 = [7 12] ;
C2D1 = [2 2] ;

model = cell(2,1) ;
model{1} = crbm2D(train, N_V2D1, N_W2D1, K1, C2D1, op1) ;

op2 = op1 ;
op2.gaussian_vis = false ;
op2.eta = op1.eta*10 ;
K2 = 3 ;
N_V2D2 = model{1}.N_V2D_next ;
N_W2D2 = [7 11] ;
C2D2 = [2 2] ;

model{2} = crbm2D(model{1}.pool_top, N_V2D2, N_W2D2, K2, C2D2, op2) ;

op3 = op2 ;
K3 = 3 ;
N_V2D3 = model{2}.N_V2D_next ;
N_W2D3 = [5 8] ;
C2D3 = [2 2] ;

model{3} = crbm2D(model{2}.pool_top, N_V2D3, N_W2D3, K3, C2D3, op3) ;

[err pstates states gendata] = c_unsuperv_offline_eval(train, model, 3, 'up') ;


