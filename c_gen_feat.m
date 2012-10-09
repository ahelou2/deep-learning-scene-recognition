function data = c_gen_feat(X, model, level, gibbs_iter)

numlayers = length(model) ;
data = X ;
[numcases numdims numfeat_maps]= size(data) ;
pstates = cell(numlayers,1) ;
states = cell(numlayers,1) ;

%Initialization up pass
for k=level+1:1:numlayers
    for i =1:1:model{k}.num_filters
        W_conv_data = zeros([(model{k}.N_V2D - model{k}.N_W2D + 1) numcases]) ;

        for j=1:1:numfeat_maps
            data_for_conv = reshape(data(:,:,j)', [model{k}.N_V2D numcases]) ;
            W_conv_data = W_conv_data + ...
                convn(data_for_conv, flipud(fliplr(model{k}.W(:,:,i,j))), 'valid') ;
        end
        W_conv_data = ...
            reshape(W_conv_data, [model{k}.N_H2D(1)*model{k}.N_H2D(2) numcases]) ;
        W_conv_data = W_conv_data' ;
        [pstates{k}.hid_top(:,:,i) pstates{k}.pool_top(:,:,i) states{k}.hid_top(:,:,i)] = multinomial_exp(W_conv_data + repmat(model{k}.b(1,i), ...
            [numcases model{k}.N_H2D(1)*model{k}.N_H2D(2)]), model{k}.C2D, model{k}.N_H2D) ;
%             states{k}.hid_top(:,:,i) = ...
%                 multinomial_exp_sample(pstates{k}.hid_top(:,:,i),
%                 model{k}.C2D, model{k}.N_H2D) ;
        states{k}.pool_top(:,:,i) = ...
            pstates{k}.pool_top(:,:,i) > rand(numcases, model{k}.N_V2D_next(1)*model{k}.N_V2D_next(2)) ;
    end
    data = pstates{k}.pool_top ;
    [numcases numdims numfeat_maps]= size(data) ;
end

% Initialization down pass
%states{numlayers}.hid_top =  states[];
%states{numlayers}.pool_top = X ;
for k=numlayers-1:-1:1
%for k=numlayers-2:-1:1
    for i =1:1:model{k}.num_filters
        % top-down signal
        T_conv_uphid_sum = ...
            zeros(numcases, model{k}.N_V2D_next(1)*model{k}.N_V2D_next(2)) ;
        for s=1:1:model{k+1}.num_filters
            uphid_for_conv = reshape(states{k+1}.hid_top(:,:,s)', [model{k+1}.N_H2D numcases]) ;
            T_conv_uphid = convn(uphid_for_conv, model{k+1}.W(:,:,s,i)) ;
            T_conv_uphid = ...
                reshape(T_conv_uphid, [model{k}.N_V2D_next(1)*model{k}.N_V2D_next(2), numcases]) ;
            T_conv_uphid = T_conv_uphid' ;
            T_conv_uphid_sum = T_conv_uphid_sum + T_conv_uphid ;
        end

        [pstates{k}.hid_top(:,:,i) pstates{k}.pool_top(:,:,i) states{k}.hid_top(:,:,i)] = multinomial_exp(replicate_pool(T_conv_uphid_sum,model{k}.C2D, model{k}.N_H2D) + repmat(model{k}.b(1,i), ...
            [numcases model{k}.N_H2D(1)*model{k}.N_H2D(2)]), model{k}.C2D, model{k}.N_H2D) ;
%             states{k}.hid_top(:,:,i) = ...
%                 multinomial_exp_sample(pstates{k}.hid_top(:,:,i), model{k}.C2D, model{k}.N_H2D) ;
        states{k}.pool_top(:,:,i) = ...
            pstates{k}.pool_top(:,:,i) > rand(numcases, model{k}.N_V2D_next(1)*model{k}.N_V2D_next(2)) ;
    end
end

% generate data
W_conv_hid_sum = zeros(numcases, model{1}.N_V2D(1)*model{1}.N_V2D(2)) ;
for i=1:1:model{1}.num_filters
    hid_for_conv = reshape(states{1}.hid_top(:,:,i)', [model{1}.N_H2D numcases]) ;
    W_conv_hid = convn(hid_for_conv, model{1}.W(:,:,i)) ;
    W_conv_hid = ...
        reshape(W_conv_hid, [model{1}.N_V2D(1)*model{1}.N_V2D(2) numcases]) ;
    W_conv_hid = W_conv_hid' ;
    W_conv_hid_sum = W_conv_hid_sum + W_conv_hid ;
end
if model{1}.type == 'BB'
    pdata = logistic(W_conv_hid_sum + repmat(model{1}.c,numcases, model{1}.N_V2D(1)*model{1}.N_V2D(2)));
    data = pdata > rand(numcases,model{1}.N_V2D(1)*model{1}.N_V2D(2));   
else %model{1}.type == 'CB'
    pdata = W_conv_hid_sum + repmat(model{1}.c,numcases, model{1}.N_V2D(1)*model{1}.N_V2D(2));
    data = pdata ;
end

dataX = data ;

for a=1:1:gibbs_iter
    % unidrected down pass
    % WARNING: I may still need to sample the last layers when I resample
    % multiple times.
    if numlayers > 2
        for k=numlayers-1:-1:1
            if k > 1
                data = pstates{k-1}.pool_top ;
            else
                data = dataX ;
            end
            [numcases numdims numfeat_maps]= size(data) ;
            for i =1:1:model{k}.num_filters
                % bottom-up signal
                W_conv_data = zeros([(model{k}.N_V2D - model{k}.N_W2D + 1) numcases]) ; % was k
                for j=1:1:numfeat_maps
                    %size(data(:,:,j)')
                    %[model{k}.N_V2D numcases]
                    data_for_conv = reshape(data(:,:,j)', [model{k}.N_V2D numcases]) ; % was k
                    W_conv_data = W_conv_data + ...
                        convn(data_for_conv, flipud(fliplr(model{k}.W(:,:,i,j))), 'valid') ; %was k
                end
                W_conv_data = ...
                    reshape(W_conv_data, [model{k}.N_H2D(1)*model{k}.N_H2D(2) numcases]) ;
                W_conv_data = W_conv_data' ;

                % top-down signal
                T_conv_uphid_sum = ...
                    zeros(numcases, model{k}.N_V2D_next(1)*model{k}.N_V2D_next(2)) ;
                for s=1:1:model{k+1}.num_filters
                    uphid_for_conv = reshape(states{k+1}.hid_top(:,:,s)', [model{k+1}.N_H2D numcases]) ;
                    T_conv_uphid = convn(uphid_for_conv, model{k+1}.W(:,:,s,i)) ;
                    T_conv_uphid = ...
                        reshape(T_conv_uphid, [model{k}.N_V2D_next(1)*model{k}.N_V2D_next(2), numcases]) ;
                    T_conv_uphid = T_conv_uphid' ;
                    T_conv_uphid_sum = T_conv_uphid_sum + T_conv_uphid ;
                end

                [pstates{k}.hid_top(:,:,i) pstates{k}.pool_top(:,:,i) states{k}.hid_top(:,:,i)] = multinomial_exp(W_conv_data + replicate_pool(T_conv_uphid_sum,model{k}.C2D, model{k}.N_H2D) + repmat(model{k}.b(1,i), ...
                    [numcases model{k}.N_H2D(1)*model{k}.N_H2D(2)]), model{k}.C2D, model{k}.N_H2D) ;
%                 states{k}.hid_top(:,:,i) = ...
%                     multinomial_exp_sample(pstates{k}.hid_top(:,:,i), model{k}.C2D, model{k}.N_H2D) ;
                states{k}.pool_top(:,:,i) = ...
                    pstates{k}.pool_top(:,:,i) > rand(numcases, model{k}.N_V2D_next(1)*model{k}.N_V2D_next(2)) ;
            end
        end
    end

    % resample top layer
    if numlayers > 1
        data = pstates{numlayers-1}.pool_top ;
    else
        data = dataX ;
    end
    [numcases numdims numfeat_maps]= size(data) ;
    for k=numlayers:1:numlayers
    %for k=1:1:numlayers
        for i =1:1:model{k}.num_filters
            W_conv_data = zeros([(model{k}.N_V2D - model{k}.N_W2D + 1) numcases]) ;
            for j=1:1:numfeat_maps
                data_for_conv = reshape(data(:,:,j)', [model{k}.N_V2D numcases]) ;
                W_conv_data = W_conv_data + ...
                    convn(data_for_conv, flipud(fliplr(model{k}.W(:,:,i,j))), 'valid') ;
            end
            W_conv_data = ...
                reshape(W_conv_data, [model{k}.N_H2D(1)*model{k}.N_H2D(2) numcases]) ;
            W_conv_data = W_conv_data' ;
            [pstates{k}.hid_top(:,:,i) pstates{k}.pool_top(:,:,i) states{k}.hid_top(:,:,i)] = multinomial_exp(W_conv_data + repmat(model{k}.b(1,i), ...
                [numcases model{k}.N_H2D(1)*model{k}.N_H2D(2)]), model{k}.C2D, model{k}.N_H2D) ;
%             states{k}.hid_top(:,:,i) = ...
%                 multinomial_exp_sample(pstates{k}.hid_top(:,:,i), model{k}.C2D, model{k}.N_H2D) ;
            states{k}.pool_top(:,:,i) = ...
                pstates{k}.pool_top(:,:,i) > rand(numcases, model{k}.N_V2D_next(1)*model{k}.N_V2D_next(2)) ;
        end
        data = pstates{k}.pool_top ;
        [numcases numdims numfeat_maps]= size(data) ;
    end

    % generate data
    W_conv_hid_sum = zeros(numcases, model{1}.N_V2D(1)*model{1}.N_V2D(2)) ;
    for i=1:1:model{1}.num_filters
        hid_for_conv = reshape(states{1}.hid_top(:,:,i)', [model{1}.N_H2D numcases]) ;
        W_conv_hid = convn(hid_for_conv, model{1}.W(:,:,i)) ;
        W_conv_hid = ...
            reshape(W_conv_hid, [model{1}.N_V2D(1)*model{1}.N_V2D(2) numcases]) ;
        W_conv_hid = W_conv_hid' ;
        W_conv_hid_sum = W_conv_hid_sum + W_conv_hid ;
    end
    if model{1}.type == 'BB'
        pgendata = logistic(W_conv_hid_sum + repmat(model{1}.c,numcases, model{1}.N_V2D(1)*model{1}.N_V2D(2)));
        gendata = pgendata > rand(numcases,model{1}.N_V2D(1)*model{1}.N_V2D(2));   
    else %model{1}.type == 'CB'
        pgendata = W_conv_hid_sum + repmat(model{1}.c,numcases, model{1}.N_V2D(1)*model{1}.N_V2D(2));
        gendata = pgendata ;
    end
end

data = gendata ;