
itn_i=1000;
itn_i_next=2000;
trailingAvgGenerator=[];
trailingAvgSqGenerator = [];
% learnRateGenerator = 0.0002; % original value
learnRateGenerator = 0.00002;
gradientDecayFactor = 0.5;
squaredGradientDecayFactor = 0.999;

data_pdVpdG=load("mat_files/partial_V_partial_G_iterations_"+itn_i+"_i_64.mat");
data_record=load("mat_files/model_weights.mat","record");
data_state=load("mat_files/state_G_"+itn_i+".mat","stateGenerator");
data_Z=load("mat_files/Z_for_queries_"+itn_i+".mat");

partial_V_partial_G=data_pdVpdG.partial_V_partial_G;
dlZ=data_Z.dlZ;
stateGenerator = data_state.stateGenerator;

[dlnetGenerator_curr_itn, ~, ~]=load_model_weights(data_record,itn_i,1);
[dlnetGenerator_next_itn, ~, ~]=load_model_weights(data_record,itn_i_next,1);

% times -1 for gradient since Matlab don't allow negative learning rate, 
% while we are maximizing V_human instead of minimizing V_human
minus_partial_V_partial_G = minus_gradient(partial_V_partial_G);

dlnetGenerator_after_human=compare_3_models(dlnetGenerator_curr_itn,dlnetGenerator_next_itn, ...
    minus_partial_V_partial_G,stateGenerator,dlZ, trailingAvgGenerator, trailingAvgSqGenerator, ...
    itn_i, itn_i_next, learnRateGenerator, gradientDecayFactor, squaredGradientDecayFactor)


function dlnetGenerator_after_human=compare_3_models(dlnetGenerator_curr_itn,dlnetGenerator_next_itn, ...
    gradients,stateGenerator,dlZ, trailingAvgGenerator, trailingAvgSqGenerator, itn_i, itn_i_next, ...
    learnRateGenerator, gradientDecayFactor, squaredGradientDecayFactor)

%% output of the model at current iteration

dlXGenerated_curr_itn = predict(dlnetGenerator_curr_itn,dlZ);

%% output of the model updated by human response

dlnetGenerator_after_human = dlnetGenerator_curr_itn;

% Update the generator network parameters.
dlnetGenerator_after_human.State = stateGenerator;
[dlnetGenerator_after_human.Learnables,~,~] = ...
    adamupdate(dlnetGenerator_after_human.Learnables, gradients, ...
    trailingAvgGenerator, trailingAvgSqGenerator, itn_i, ...
    learnRateGenerator, gradientDecayFactor, squaredGradientDecayFactor);

dlXGenerated_after_human = predict(dlnetGenerator_after_human,dlZ);

%% output of the model at the next iteration


dlXGenerated_next_itn = predict(dlnetGenerator_next_itn ,dlZ);

%% result comparison

diff_curr_next = dlXGenerated_next_itn - dlXGenerated_curr_itn;
diff_curr_human = dlXGenerated_after_human - dlXGenerated_curr_itn;
diff_human_next = dlXGenerated_next_itn - dlXGenerated_after_human;

EuD_curr_next = gather(extractdata(sum(sum(diff_curr_next.^2))));
EuD_curr_human = gather(extractdata(sum(sum(diff_curr_human.^2))));
EuD_human_next = gather(extractdata(sum(sum(diff_human_next.^2))));

change_percent= (EuD_human_next(:)-EuD_curr_next(:))./EuD_curr_next(:);
histogram(change_percent);

title(["Histogram of percentage change of distance", ...
    "between outputs at iteration ("+itn_i+", "+itn_i_next+")"])
xlabel("Percentage Change of d(x_{Curr},x_{Next}) vs. d(x_{Human},x_{Next})")
ylabel("Frequency")

%% plot selected data

visualize_dlXGenerated(dlXGenerated_curr_itn, ...
    dlXGenerated_after_human,dlXGenerated_next_itn);

end

function visualize_dlXGenerated(dlXGenerated_curr_itn, ...
    dlXGenerated_after_human,dlXGenerated_next_itn)

Ns_plot_selected=[2, 5, 7,9, 10, 12]; % show the data to be plotted

fig = figure;

% num_data=size(dlXGenerated_curr_itn,4);
num_selected=length(Ns_plot_selected);

for j=1:num_selected
    i=Ns_plot_selected(1,j);
    
    img_curr_itn = gather(extractdata(dlXGenerated_curr_itn(:,:,:,i)));
    img_after_human = gather(extractdata(dlXGenerated_after_human(:,:,:,i)));
    img_next_itn = gather(extractdata(dlXGenerated_next_itn(:,:,:,i)));
    
    subplot(num_selected,3,(j-1)*3+1);
    imshow(img_curr_itn,'InitialMagnification','fit');
    
    subplot(num_selected,3,(j-1)*3+2);
    imshow(img_after_human,'InitialMagnification','fit');
    
    subplot(num_selected,3,(j-1)*3+3);
    imshow(img_next_itn,'InitialMagnification','fit');
    
end

end

function minus_grad = minus_gradient(grad)

% the program is to get grad_minus = - grad
% need to use this program if V is to be maximized (i.e. V=sum(D(G(z)))
% no need to use if V is to be minimized (i.e. V=-sum(D(G(z)))

% input: grad belongs to a tensor w.r.t. the model structure

num_layers=size(grad,1);

minus_grad=grad;
for i=1:num_layers
    minus_grad.Value{i,1}=(-1)*grad.Value{i,1};
end

end
