clear all

iteration = 1000; % please choose 100, 500, 1000 respectively

num_data=25; % fix as 25
R=10; % fix as 10
person_j=1; % can change from 1 to 10


%% human interaction

% use fullfile() so that the code would be OS independent
data_task=load(fullfile("mat_files","tasks.mat"),"tasks");
tasks=data_task.tasks;

filename_X_plus_minus=fullfile("mat_files","X_plus_minus_"+iteration+".mat");

D_scores=zeros(num_data,R);
[D_scores, user_name, task_done]=a4b_human_choose_pictures_itn_i_person_j(filename_X_plus_minus,tasks,person_j,D_scores);


%% save the result

clear("data_j")
data_j.num_data=num_data;
data_j.R=R;
data_j.person_j=person_j;
data_j.iteration=iteration;
data_j.tasks=tasks;
data_j.task_done=task_done;
data_j.D_scores=D_scores;
data_j.user_name=user_name;

clk = clock; % get current time

filename=fullfile("human_response","result." ...
+clk(1)+"."+clk(2)+"."+clk(3)+"."+clk(4)+"."+clk(5)+"."+clk(6) ...
+"_"+user_name+"_iteration_"+iteration+".mat");
save(filename,"data_j");

spaces4="\n \n \n \n ";
spaces1="\n ";
msg="Can you assist to send the file '"+filename+"' to me? Thanks!";

fprintf(spaces4);
disp(msg);
fprintf(spaces1);
