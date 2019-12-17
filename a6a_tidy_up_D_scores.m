num_data=25;
R=10;
num_person=10;

iterations=[1000, 500, 100];
num_itn=size(iterations,2);

% comparing inv_Sigma of Matlab v2019b, Mathematica 12 and Numpy v1.6.2,
% we conclude that the inv_Sigma of Numpy and Mathematica are numerically more stable than Matlab's 
filename_Sigma="mat_files/inv_Sigma_from_Numpy.csv";
inv_Sigma_Matrix_theta=csvread(filename_Sigma);

% lists is a 10x3 array
folder_prefix="human_response/Collected/";
lists=create_lists(folder_prefix);

fig=figure;
for itn_idx=1:num_itn
    
    itn_i=iterations(1,itn_idx);
    
    list=lists(:,itn_idx);
    
    % folder="human_response/"
    
    [D_scores_combine, num_empty]=combine_D_1_iteration(list,num_data,R, ...
        itn_i,fig,num_itn,num_person,itn_idx);
    
    ptl_V_ptl_x=zeros(num_data,64^2);
    for n=1:num_data
        ptl_V_ptl_xn=partial_V_partial_x_hat_n(D_scores_combine,num_empty,n,itn_i,inv_Sigma_Matrix_theta);
        ptl_V_ptl_x(n,:)=ptl_V_ptl_xn';
    end
    
    save("mat_files/ptl_V_ptl_x_iteration_"+itn_i+".mat","ptl_V_ptl_x");
    
end

function ptl_V_ptl_xn=partial_V_partial_x_hat_n(D_scores_combine, num_empty,n,itn_i,inv_Sigma_Matrix_theta)

[num_data, R] = size(D_scores_combine);

if num_empty~=0
    error("num_empty shoule be 0 in partial_V_partial_x_hat_n().")
end

filename_X_plus_minus="mat_files/X_plus_minus_"+itn_i+".mat";
data=load(filename_X_plus_minus,"X_plus","X_minus");
delta_X_all=data.X_plus-data.X_minus;

[h_img, w_img, num_channel, num_data, num_pertn]=size(delta_X_all);

if num_pertn~=R
    error("The number of perturbation doesn't match.")
end

sum_product=zeros(h_img, w_img, num_channel);
for r=1:R
    delta_D=D_scores_combine(n,r);
    delta_X=delta_X_all(:,:,:,n,r);
    sum_product=sum_product+delta_D*delta_X;
end

sum_product_reshaped=reshape(sum_product,[h_img*w_img, 1]); % 1 changed to 3 for num_channel=3?
ptl_V_ptl_xn=inv_Sigma_Matrix_theta*sum_product_reshaped/2/R;

end

function [D_scores_combine, num_empty]=combine_D_1_iteration(filepaths, ...
    num_data,R,itn_i,fig,num_itn,num_person,itn_idx)

% input: filepaths is a column vector of strings

num_files=size(filepaths,1);

D_scores_combine=zeros(num_data,R);
for i=1:num_files
    filepath=filepaths(i,1);
    
    data=load(filepath);
    data_j=data.data_j;
    
    if data_j.iteration~=itn_i;
        error("The iteration number doesn't match.")
    end
    
    D_scores=reasonableness_check_1_file(data_j,fig,num_itn,num_person,itn_idx);
    
    % overlapping check
    space_empty=(D_scores_combine==0);
    space_to_be_filled=(D_scores~=0);
    space_diff=space_empty-space_to_be_filled;
    num_fill_non_empty=sum(sum(space_diff<0));
    if num_fill_non_empty>=1
        error("Some D_scores of different files overlap with each other")
    end
    
    D_scores_combine=D_scores_combine+D_scores;
    
end

mask_combine_non_empty=(D_scores_combine~=0);
num_empty=sum(sum(not(mask_combine_non_empty)));

% see histogram of D_scores_combine
figure(fig)
subplot(num_itn,num_person+1,itn_idx*(num_person+1));
edges = -1:(2/8):1;
histogram(D_scores_combine(mask_combine_non_empty),edges)
title(["COMBINED D-Scores" , "#(Tasks Done): "+sum(sum(mask_combine_non_empty)),"Iteration: "+itn_i]);
xlabel('\Delta D')
ylabel('Frequency')

end

function D_scores=reasonableness_check_1_file(data_j,fig,num_itn,num_person,itn_idx)

% please note that indeices of D_scores are consecutive in the same row

num_data=data_j.num_data;
R=data_j.R;
tasks=data_j.tasks;
person_j=data_j.person_j;
task_done=data_j.task_done;
D_scores=data_j.D_scores;
user_name=data_j.user_name;
iteration=data_j.iteration;

task_done_valid = setdiff(task_done,[-1]); % removed "-1", removed duplicated, ordered

normal_order=1:(num_data*R);
mask_non_zero_transpose=(D_scores~=0)';
selected_posi=normal_order(mask_non_zero_transpose(:));

% task_done check
if length(tasks(person_j,:))~=length(task_done)
    error("number of task_done doesn't match with tasks")
    
    if prod(tasks(person_j,:)-task_done)~=0
        error("task_done doesn't match with tasks and person_j")
    end
end

% rank of D_scores check, which should be 2
if length(size(D_scores))~=2
    error("rank of D_scores should be 2.")
end

% size of D_scores check, which should be NxR
if size(D_scores,1)~=num_data || size(D_scores,2)~=R
    error("size of D_scores doesn't match with N and R.")
end

% check whether task_done_valie match with the non-zero D_scores
if prod((task_done_valid - selected_posi)==0)~=1
    error("task_done_valie doesn't match with non-zero D_scores")
end

if max(max(D_scores))>1 || min(min(D_scores))>1
    error("The min. or max. value of D should be within [-1, 1].")
end

if length(unique(D_scores))>9
    error("D_scores have too many different values.")
end

% see histogram of D_scores
figure(fig)
subplot(num_itn,num_person+1,(itn_idx-1)*(num_person+1)+person_j);
edges = -1:(2/8):1;
histogram(D_scores(D_scores~=0),edges)
title(["Name: "+user_name, "#(Tasks Done): "+length(task_done_valid),"Iteration: "+iteration]);
xlabel('\Delta D')
ylabel('Frequency')

end

function lists=create_lists(folder_prefix)

files = dir(fullfile(folder_prefix, "result.*.mat"));

num_file=size(files,1);

lists=[""];

for i=1:num_file
    filename=files(i).name;
    itn_i=str2num(extractBefore(extractAfter(filename,"iteration_"),".mat"));
    person_j=str2num(extractBefore(extractAfter(filename,"person_"),"_iteration"));
    
    itn_idx=vlookup(itn_i,[1000, 500, 100]',[1, 2, 3]');
    
    if length(itn_idx)~=1
        error("itn_idx should have length 1")
    end
    lists(person_j,itn_idx)=folder_prefix+filename;
    
end

end

function returns=vlookup(value,col_values,col_returns)

returns=col_returns(col_values==value);

end
