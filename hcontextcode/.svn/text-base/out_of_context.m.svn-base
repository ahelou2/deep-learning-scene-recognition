clear all

configure

gist = true;
loc = true;
use_true_labels = true;
debugmode = false;

% load detector outputs DdetectorTest and ground-truth labels Dtest
load sun_outofcontext_data

% Select images with at least one out-of-context object among the 107
% categories.
valid = false(length(outofcontext_groundtruth),1);
for i = 1:length(outofcontext_groundtruth)
    if outofcontext_groundtruth{i}(1)==-1
        valid(i)=false;
    else
        objects = {Doutofcontext(i).annotation.object.name};
        outofcontextNames = objects(outofcontext_groundtruth{i});
        tf = ismember(outofcontextNames,names);
        if(any(tf))
            valid(i) = true;
        end
    end
end
outofcontext_groundtruth = outofcontext_groundtruth(valid);
DdetectorTest = DdetectorOutOfContext(valid);
Dtest = Doutofcontext(valid);
p_b_gist_test = p_b_gist_test(valid, :);

if(use_true_labels) % Detecting objects out-of-context conditioned on ground-truth labels
    DdetectorTest = Dtest;
end

DdetectorTestContext = DdetectorTest;
Nimages = length(DdetectorTest);
match_outofcontext = zeros(Nimages,5);
match_outofcontext_rand = zeros(Nimages,5);

prob_bi = [1-prob_bi1, prob_bi1];
prior_loc = locationPot.hmar./locationPot.Jmar;
for n=1:Nimages
    objects = {DdetectorTestContext(n).annotation.object.name};
    [foo,obj] = ismember(objects, names); obj = obj';
    
    % find valid objects
    valid = find(obj>0);
    obj = obj(valid);
    unique_obj = unique(obj);
    
    image_size(1) = DdetectorTestContext(n).annotation.imagesize.ncols;
    image_size(2) = DdetectorTestContext(n).annotation.imagesize.nrows;   
    [loc_index,loc_measurements,loc_img_coords] = getWindowLoc(DdetectorTestContext(n).annotation.object(valid),names,image_size,heights);

    if(~use_true_labels) % Use detector outputs only
        scores = [DdetectorTest(n).annotation.object.confidence]';
        scores = scores(valid);
        
        % Inference using an independent model
        prior_node_pot_g = indep_node_pot;
        if(gist) % Add gist
            p_b_gist1 = p_b_gist_test(n,:)';
            pnp_g = prior_node_pot_g(1:Ncategories,:).*[(1-p_b_gist1)./(1-prob_bi1) p_b_gist1./prob_bi1];
            prior_node_pot_g(1:Ncategories,:) = pnp_g./repmat(sum(pnp_g,2),1,2);
        end 
        [adjmat, node_pot, edge_pot, pw1lscore] = addCandidateWindows(obj,scores,windowScore,prior_adjmat,prior_node_pot_g,indep_edge_pot,prob_bi1);   
        tree_msg_order = treeMsgOrder(adjmat,root);
        node_marginals = sumProductBin(adjmat, node_pot, edge_pot, tree_msg_order); % Computes p(bi | measurement_i)
        b = ones(Ncategories,1);
        b(node_marginals(1:Ncategories,2)>0.05) = 2;
    
        % Compute the most likely location for each object conditioned on
        % measurements
        temp_node_pot = node_pot; 
        temp_node_pot(1:Ncategories,:) = repmat([0,1],Ncategories,1);
        smp = maxProductBin(adjmat, node_pot, edge_pot, tree_msg_order);
        [Jprior,hprior] = computeJhPrior(locationPot, smp(1:Ncategories),prior_edges);
        correct_detection = (smp(Ncategories+1:end)==2);
        [Jmeas,hmeas] = computeJhMeas(loc_index(correct_detection), loc_measurements(correct_detection,:), detectionWindowLoc);
        map_loc = (Jprior + Jmeas)\(hprior + hmeas);
        map_loc = reshape(map_loc',K,Ncategories)';         
        
    else % Get ground truth labels and compute the median location for each object        
        b = ones(Ncategories,1);
        b(unique_obj) = 2; 
        node_marginals = repmat([1,0],Ncategories,1);
        node_marginals(unique_obj,1) = 0;
        node_marginals(unique_obj,2) = 1;
        
        map_loc = prior_loc; % Use prior locations for objects that are not present.
        for oi=1:length(unique_obj)
            o = unique_obj(oi);
            map_loc(o,:) = median(loc_measurements(obj==o,:),1); 
        end          
    end
    
    context_scores_bin = zeros(Ncategories,1);
    context_scores = zeros(Ncategories,1);
    for e=1:size(prior_edges,1);
        p = prior_edges(e,1);
        c = prior_edges(e,2);
        
        % The ratio between prob(b_c|b_p) with context and without context 
        ratio_bin = repmat(prob_bi(c,:),2,1)./full(prior_edge_pot(2*p-1:2*p,2*c-1:2*c)); 
        
        g = map_loc(c,:)';
        J = diag(locationPot.Jmar(c,:));
        mu = prior_loc(c,:)';
        prob_loc_off = exp(-0.5*(g-mu)'*J*(g-mu))*sqrt(det(J)); % Probability of g with context off         
        J = diag(locationPot.Jcond(e,1:K));
        h = locationPot.hcond(e,:)' - diag(locationPot.Jcond(e,K+1:end))*map_loc(p,:)';   
        mu = J\h;
        prob_loc_cond = exp(-0.5*(g-mu)'*J*(g-mu))*sqrt(det(J)); % Prob. of g when parent is present with context on    
        J = diag(locationPot.Jcondnp(c,:));
        h = locationPot.hcondnp(c,:)';        
        mu = J\h;
        prob_loc_condnp = exp(-0.5*(g-mu)'*J*(g-mu))*sqrt(det(J)); % Prob. of g when parent is not present with context on    
        
        ratio_loc = [1, prob_loc_off/prob_loc_condnp; 1, prob_loc_off/prob_loc_cond];
         
        nm_mat = node_marginals(p,:)'*node_marginals(c,:);
        context_scores_bin(c) = sum(sum(nm_mat./(1+ratio_bin)));
        context_scores(c) = sum(sum(nm_mat./(1+ratio_bin.*ratio_loc)));
    end
    
    if(~loc)
        context_scores = context_scores_bin;
    end
    
    % Evaluate performance
    context_scores(root) = 1000;
    context_scores(b(1:Ncategories)==1) = 1000;  % Ignore objects that are not likely present
    true_objects = {Dtest(n).annotation.object.name};
    outofcontextNames = true_objects(outofcontext_groundtruth{n});
    cand_obj = find(b(1:Ncategories)==2);
    rand_names = names(cand_obj(randperm(length(cand_obj))));
    [sorted_scores, sort_ind] = sort(context_scores, 'ascend');
    sorted_names = names(sort_ind);
    k = min(5,length(cand_obj));
    match_outofcontext(n,1:k) = ismember(sorted_names(1:k),outofcontextNames);
    match_outofcontext_rand(n,1:k) = ismember(rand_names(1:k),outofcontextNames);
    
    if(debugmode)
        disp(n)
        fprintf('Ground truth out of objects:'); disp(outofcontextNames)
        fprintf('Unexpected objects:'); disp(sorted_names(1:k))
        fprintf('Random orders:'); disp(rand_names(1:k))
        for oi=1:length(cand_obj)
            o = cand_obj(oi);
            fprintf('%d, %d, %s, \t %.2f, %.2f\n',o,oi,names{o}, context_scores_bin(o), context_scores(o));
        end
        
        Dcontext = DdetectorTestContext(n);
        Dcontext.annotation.object = Dcontext.annotation.object(valid(obj==sort_ind(1)));
        figure(1); clf;
        subplot(121); LMplot(Dtest(n),1,HOMEIMAGES); title('Image');
        subplot(122); LMplot(Dcontext,1,HOMEIMAGES); title('Most unexpected');
        %keyboard
    end
end    


% Compute the number of images in which at least one of the top N
% unexpected is actually out-of-context.
num_match_random_guess = sum(logical(cumsum(match_outofcontext_rand,2)),1);
num_match_images = sum(logical(cumsum(match_outofcontext,2)),1);

figure(3)
bar(([num_match_random_guess; num_match_images])');
axis([0 6 0 Nimages])
grid on
legend('Random guess','Context')
xlabel('Top N unexpected object')
ylabel('Number of images correct')

