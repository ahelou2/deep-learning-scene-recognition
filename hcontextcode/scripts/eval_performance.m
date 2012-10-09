% Evaluation localization
clear ap apc au auc ap_bound apc_bound
for c = 1:Ncategories
    objectname = names{c};
        
    [recall, precision, DdetectorTest, threholds, score, correct, ap(c)] = LMrecallPrecision(Dtest, DdetectorTest, objectname, 'nomisses');
    [recallc, precisionc, DdetectorTestContext, threholdsc, scorec, correctc, apc(c)] = LMrecallPrecision(Dtest, DdetectorTestContext, objectname, 'nomisses');
    ap_bound(c) = max(recall);
    apc_bound(c) = max(recallc);
    [prRecall,prPrecision,foo, au(c)]= precisionRecall(presence_score(c,:), presence_truth(c,:));
    [prRecallc,prPrecisionc, foo, auc(c)] = precisionRecall(presence_score_c(c,:), presence_truth(c,:));
   
    figure(2); clf
    annotation('textbox',[0.45 0.01 .1 .1],...
        'String',objectname,'FontSize',14,'FontName','Arial','EdgeColor','none');   
    subplot(221)
    plot(recall, precision, 'b', recallc, precisionc, 'r')
    legend('Baseline','Context')
    title('Localization: precision-recall')
    axis('square')
    subplot(222)
    plot(prRecall, prPrecision, 'b', prRecallc, prPrecisionc, 'r')
    title('Presence prediction: precision-recall')
    axis('square')
    subplot(223)
    figROC(score,correct,'b');
    hold on
    figROC(scorec,correctc,'r');
    hold off
    title('Localization: ROC')
    subplot(224)
    figROC(presence_score(c,:), presence_truth(c,:), 'b');
    hold on
    figROC(presence_score_c(c,:), presence_truth(c,:), 'r');
    hold off
    title('Presence Prediction: ROC')
    drawnow        
end

% Precision
disp(MODEL)
disp(sprintf('       Category \t base \t context'))
for c = 1:Ncategories
    disp(sprintf('%15s \t %2.2f \t %2.2f', names{c}, ap(c), apc(c)) )
end
disp(sprintf('%15s \t %2.2f \t %2.2f', 'AVERAGE', mean(ap), mean(apc)) )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NEW VISUALIZATIONS

% 1) bar plots
[foo, k] = sort(apc-ap, 'descend');
disp(sprintf('       Category \t base  context  context-base bound'))
for c = (k)
    disp(sprintf('%15s \t %2.2f \t %2.2f \t %2.2f \t %2.2f', names{c}, ap(c), apc(c), apc(c)-ap(c), apc_bound(c)))
end
disp(sprintf('%15s \t %2.2f \t %2.2f \t %2.2f \t %2.2f \t %2.2f', 'AVERAGE', mean(ap), mean(apc), mean(apc-ap), mean(apc_bound)))

% As current context models only enhance the scores of bounding boxes proposed by the detector, context can not increase the recall, 
% which poses a limit to the maximum performances that can be achieved. This bound can be very low for some object classes. 
figure
subplot(211)
bar(apc(k)-ap(k),'r')
title('LOCALIZATION PERFORMANCE')
xlim([0 Ncategories+1])
ylim([-6 15])
subplot(212)
bar(auc(k)-au(k),'r')
xlim([0 Ncategories+1])
ylim([-2 35])
title('PRESENCE PREDICTION PERFORMANCE')

% 2) performance at recogniton of top detections: scene parsing perfomance.
% One of the goals of multiclass object detection is to provide an
% interpretation of an image. For this, it might be sufficient to be able
% to confidently recognize a few objects within each image. Here we
% evaluate the average recognition performance for the top N most confindent
% detections on each image.
correct_baseline = zeros([10 1]);
counts_baseline = zeros([10 1]);
correct_model = zeros([10 1]);
counts_model = zeros([10 1]);
Ncorrect_baseline = zeros(Nimages,1);
Ncorrect_model = zeros(Nimages,1);
Ncorrect_bound = zeros(Nimages,1);
for n = 1:Nimages
    % baseline
    scores_baseline = [DdetectorTest(n).annotation.object.p_w_s];
    type_baseline = ismember({DdetectorTest(n).annotation.object.detection}, {'correct'});
    
    [s,k] = sort(scores_baseline, 'descend');
    Nd = length(k);
    Nd = min(Nd,10);
    correct_baseline(1:Nd) = correct_baseline(1:Nd)+type_baseline(k(1:Nd))';
    counts_baseline(1:Nd) = counts_baseline(1:Nd)+1;
    Ncorrect_baseline(n) = find(type_baseline(k)==0, 1 )-1;    
    
    % model
    scores_model = [DdetectorTestContext(n).annotation.object.confidence];
    type_model = ismember({DdetectorTestContext(n).annotation.object.detection}, {'correct'});
    
    [s,k] = sort(scores_model, 'descend');
    Nd = length(k); 
    Nd = min(Nd,10);
    %Nd = min(Nd, sum(type_baseline));
    correct_model(1:Nd) = correct_model(1:Nd)+type_model(k(1:Nd))';
    counts_model(1:Nd) = counts_model(1:Nd)+1;
    Ncorrect_model(n) = find(type_model(k)==0, 1 )-1;
    
    % ground truth
    objects = {Dtest(n).annotation.object.name};
    [foo,true_obj] = ismember(objects, names); 
    Ncorrect_bound(n) = length(find(true_obj>0));        
end
perf_basline = correct_baseline./counts_baseline;
perf_model = correct_model./counts_model;


figure
subplot(121)
h_t = hist(Ncorrect_bound, 0:5);
ht = cumsum(h_t(6:-1:2));
h_b = hist(Ncorrect_baseline, 0:5);
hb = cumsum(h_b(6:-1:2))./ht;
h_m = hist(Ncorrect_model, 0:5);
hm = cumsum(h_m(6:-1:2))./ht;
bar(([hb(end:-1:1); hm(end:-1:1)])')
title('Localization')
axis('square')
xlabel('N')
ylabel('Percentage of Images')
grid on
legend('Baseline','Context')

% 3) Repeat the above evaluation for presence prediction, not localization.
correct_baseline2 = zeros([10 1]);
counts_baseline2 = zeros([10 1]);
correct_model2 = zeros([10 1]);
counts_model2 = zeros([10 1]);
Ncorrect_baseline2 =zeros(Nimages,1);
Ncorrect_model2 = zeros(Nimages,1);
for n = 1:Nimages
    true_presence = presence_truth(:,n);

    % baseline
    scores_baseline = presence_score(:,n);
    
    [s,k] = sort(scores_baseline, 'descend');
    Nd = length(k);
    Nd = min(Nd,10);
    correct_baseline2(1:Nd) = correct_baseline2(1:Nd)+true_presence(k(1:Nd));
    counts_baseline2(1:Nd) = counts_baseline2(1:Nd)+1;
    Ncorrect_baseline2(n) = find(true_presence(k)==0, 1 )-1;
    
    % model
    scores_model = presence_score_c(:,n);
    
    [s,k] = sort(scores_model, 'descend');
    Nd = length(k); 
    Nd = min(Nd,10);
    correct_model2(1:Nd) = correct_model2(1:Nd)+true_presence(k(1:Nd));
    counts_model2(1:Nd) = counts_model2(1:Nd)+1;
    Ncorrect_model2(n) = find(true_presence(k)==0, 1 )-1;
end
perf_basline2 = correct_baseline2./counts_baseline2;
perf_model2 = correct_model2./counts_model2;

Ncorrect_bound2 = sum(presence_truth,1);

subplot(122);cla
h_t2 = hist(Ncorrect_bound2, 0:5);
ht2 = cumsum(h_t2(6:-1:2));
h_b2 = hist(Ncorrect_baseline2, 0:5);
hb2 = cumsum(h_b2(6:-1:2))./ht2;
h_m2 = hist(Ncorrect_model2, 0:5);
hm2 = cumsum(h_m2(6:-1:2))./ht2;
bar(([hb2(end:-1:1); hm2(end:-1:1)])')
title('Presence Prediction')
axis('square')
xlabel('N')
ylabel('Percentage of Images')
grid on
legend('Baseline','Context')