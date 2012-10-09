% Use the hierarchical context model to adjust probabilities of objects in
% the current image.

if(useSamples)
    Nite = 12;
    disp('Using sampling methods for inference...')
else
    Nite = 3;
    disp('Using MAP estimates for inference...')
end

for n=1:Nimages
    disp(n)
    objects = {DdetectorTestContext(n).annotation.object.name};
    [foo,obj] = ismember(objects, names); obj = obj';
    scores = [DdetectorTest(n).annotation.object.confidence]';
    
    % find valid objects
    valid = find(obj>0);
    obj = obj(valid);
    scores = scores(valid);

    W = length(scores); % Total number of candidate windows
    prior_node_pot_g = prior_node_pot;
    if(gist) % Add gist
        p_b_gist1 = p_b_gist_test(n,:)';
        pnp_g = prior_node_pot(1:Ncategories,:).*[(1-p_b_gist1)./(1-prob_bi1) p_b_gist1./prob_bi1];
        prior_node_pot_g(1:Ncategories,:) = pnp_g./repmat(sum(pnp_g,2),1,2);
    end 

    % Combine the measurement and the prior model
    [adjmat, node_pot, edge_pot, pw1lscore] = addCandidateWindows(obj,scores,windowScore,prior_adjmat,prior_node_pot_g,prior_edge_pot,prob_bi1);

    % Inference
    tree_msg_order = treeMsgOrder(adjmat,root);    

    if(loc)
        image_size(1) = DdetectorTestContext(n).annotation.imagesize.ncols;
        image_size(2) = DdetectorTestContext(n).annotation.imagesize.nrows;   
        [loc_index,loc_measurements,loc_img_coords] = getWindowLoc(DdetectorTestContext(n).annotation.object(valid),names,image_size,heights);

        init_node_pot = node_pot;
        init_edge_pot = edge_pot;   
        max_ll = -1e10;
        for ite=1:1:Nite
            if(mod(ite,3)==1)
                node_pot = init_node_pot;
                edge_pot = init_edge_pot;
            end
            if(useSamples)
                [smp, ll_bin] = sampleFromBinTree(adjmat, node_pot, edge_pot, tree_msg_order);
            else
                smp = maxProductBin(adjmat, node_pot, edge_pot, tree_msg_order);   
                ll_bin = ite*1000; % For MAP estimates, always replace with new estimates.
            end

            % Compute location statistics conditioned on the samples of binary variables.
            [Jprior,hprior] = computeJhPrior(locationPot, smp(1:Ncategories),prior_edges);
            correct_detection = (smp(Ncategories+1:end)==2);
            [Jmeas,hmeas] = computeJhMeas(loc_index(correct_detection), loc_measurements(correct_detection,:), detectionWindowLoc);

            % Compute the estimate of gi's
            map_g = (Jprior + Jmeas)\(hprior + hmeas);
            map_g = reshape(map_g',K,Ncategories)'; 
            ll_loc = 0.5*log(det(Jprior+Jmeas));
            %names((smp(1:Ncategories)==2))
            %fprintf('%d %f %f %f\n',ite,ll_bin,ll_loc, ll_bin+ll_loc);
            ll = ll_bin + ll_loc;

            % Update binary potentials using location estimates
            node_pot = binPotCondLocMeas(init_node_pot,loc_index,map_g,loc_measurements,detectionWindowLoc); % Compute p(cik | bi, Li, Wik)
            edge_pot = binPotCondLoc(init_edge_pot,locationPot,prior_edges,map_g); % Compute p(bj | bi, Li, Lj)

            if(ll > max_ll)
                max_ll = ll;
                max_node_pot = node_pot;
                max_edge_pot = edge_pot;
            end                        
        end
        node_pot = max_node_pot;
        edge_pot = max_edge_pot;
    end
    node_marginals = sumProductBin(adjmat, node_pot, edge_pot, tree_msg_order);
    new_scores = node_marginals(:,2);   
    
    % Insert scores in final result struct
    small_number = 1e-5;  % Used to avoid numerical issues.
    for i = 1:W
        DdetectorTestContext(n).annotation.object(valid(i)).confidence = (1-small_number)*new_scores(i+Ncategories)+small_number*(scores(i)+10)/20;
        DdetectorTest(n).annotation.object(valid(i)).p_w_s =  (1-small_number)*pw1lscore(i)+small_number*(scores(i)+10)/20;
    end      

    % Insert presence prediction
    [foo, trueobj] = ismember({Dtest(n).annotation.object.name}, names);
    presence_truth(setdiff(trueobj,0),n) = 1;        
    for m = 1:Ncategories
         presence_score(m,n) = max([0; pw1lscore(obj==m)]);
    end         
    presence_score_c(:,n) = (1-small_number)*new_scores(1:Ncategories)+small_number*presence_score(:,n);

    if(debugmode)
        numDisplay = 6;
        Dbaseline = DdetectorTest(n);
        Dcontext = DdetectorTestContext(n);
        [confidence_sorted,conf_ind] = sort([Dbaseline.annotation.object.p_w_s],'descend');
        Dbaseline.annotation.object = Dbaseline.annotation.object(conf_ind(1:numDisplay));
        for i=1:numDisplay
            Dbaseline.annotation.object(i).confidence = Dbaseline.annotation.object(i).p_w_s;
        end
        [confidence_sorted,conf_ind] = sort([Dcontext.annotation.object.confidence],'descend');
        Dcontext.annotation.object = Dcontext.annotation.object(conf_ind(1:numDisplay));
        figure(1); clf;
        subplot(131); LMplot(Dtest(n),1,HOMEIMAGES); title('Ground-truth');
        subplot(132); LMplot(Dbaseline,1,HOMEIMAGES); title('Baseline');
        subplot(133); LMplot(Dcontext,1,HOMEIMAGES); title('Context');
        drawnow
    end       
end    