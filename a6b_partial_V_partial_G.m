% this m-file is used to calculate partial_V_partial_G and save the values

% iterations=[100, 500, 1000];
iterations=[1000, 500, 100];
num_data=25;

num_iterations=size(iterations,2);

for itn_idx=1:num_iterations
    
    itn_i=iterations(1,itn_idx); % don't change this number in the sample code
    
    %% load data_ptl_V_ptl_x and model weight (i.e. won't reset by GPU)
    
    data_ptl_V_ptl_x=load("mat_files/ptl_V_ptl_x_iteration_"+itn_i+".mat","ptl_V_ptl_x");
    ptl_V_ptl_x=data_ptl_V_ptl_x.ptl_V_ptl_x;
    
    % can hide the below 2 lines in the 2nd run, to save running time
    data_record=load("mat_files/model_weights.mat","record");
    [dlnetGenerator, ~, iteration]=load_model_weights(data_record,itn_i,1);
    
    %% to load the same noises for images generation; will disappear after GPU reset
    filename_Z="mat_files/Z_for_queries_"+itn_i+".mat";
    
    % matrix_from_to=[(8:8:64)'-7, (8:8:64)']; % for GPU Titan V
    matrix_from_to=[(2:2:64)'-1, (2:2:64)']; % for GPU GeForce 940M
    
    num_fragment=size(matrix_from_to,1);
    
    start = tic;
    for i=1:64
        for fragment_ii=1:num_fragment
            
            j_from=matrix_from_to(fragment_ii,1);
            j_to=matrix_from_to(fragment_ii,2);
            
            data_Z=load(filename_Z);
            dlZ=data_Z.dlZ;
            
            if i==1 && fragment_ii==1
                partial_V_partial_G="Nil"; % initialize partial_V_partial_G
            else
            end
            
            partial_V_partial_G = dlfeval(@get_partial_sum_within_GPU_reset, partial_V_partial_G ...
                ,ptl_V_ptl_x,dlnetGenerator,itn_i, num_data,i, j_from, j_to, dlZ,start);
            
            filename_pdVpdG="mat_files/partial_V_partial_G_iterations_"+itn_i ...
                +"_i_"+i+"_frag_"+fragment_ii+".mat";
            
        end
        
        if mod(i,8)==0
            filename_pdVpdG="mat_files/partial_V_partial_G_iterations_"+itn_i ...
                +"_i_"+i+".mat";
            save(filename_pdVpdG,"partial_V_partial_G")
        end
        
    end
    
end

%% functions

function partial_V_partial_G=get_partial_sum_within_GPU_reset(partial_V_partial_G ...
    ,ptl_V_ptl_x,dlnetGenerator,itn_i,num_data,i, j_from, j_to, dlZ,start)

% input: partial_V_partial_G should be "Nil" in the first sum, and then in
% the format of gradients

% input: grads_1_i is the grads of specific i, is thus a 64*num_data
% tensor, while each entry is a gradient structure object

% GPU needs to be reset constantly in view of the resources limitation
% inside this function, GPU won't be reset

% Calculate the predictions for generated data with the discriminator network.
[dlXGenerated,stateGenerator] = forward(dlnetGenerator,dlZ);

flag_print=0;
for j=1:64
    index=64*(j-1)+i; % vertical and then horizontal
    if j>=j_from && j<=j_to
        flag_print=1;
        
        for data_k=1:num_data
            
            % to get the gradient pixel by pixel
            x_hat_1_pixel_1_data=dlXGenerated(i,j,1,data_k);
            
            % "RetainData" is used in dG to speed up the next step of dD which reuse dlgradient()
            grad_1_pixel_1_data = dlgradient(x_hat_1_pixel_1_data, dlnetGenerator.Learnables,'RetainData',true);
            % gradientsGenerator = dlgradient(x_hat_1_pixel_1_data, dlnetGenerator.Learnables,'RetainData',false);
            
            % code to avoid error for the 1st sum
            if i==1 && j==1 && data_k==1
                if partial_V_partial_G~="Nil"
                    error("partial_V_partial_G shouldn't have value in the first sum")
                end
                partial_V_partial_G=zerolize_gradients(grad_1_pixel_1_data);
            end
            
            % update partial_V_partial_G
            weight2 = ptl_V_ptl_x(data_k,index); % scalar
            partial_V_partial_G = grad_sum(1, partial_V_partial_G, weight2, grad_1_pixel_1_data);
            
        end
        
    end
    
    if flag_print==1 % display progress
        toc(start) % running time
        disp("(itn, i, j) = ("+itn_i+", "+i+", "+j+")"); % show current i, j
        flag_print=0;
    end
end


end

function zerolized_gradients=zerolize_gradients(gradients)

num_layers=size(gradients,1);

zerolized_gradients=gradients;
for i=1:num_layers
    zerolized_gradients.Value{i,1}=0*gradients.Value{i,1};
end

end

function gradients_sum=grad_sum(weight1, gradients1, weight2, gradients2)

% the program is to compute weight1*gradients1+weight2*gradients2

% input: weight1, weight2 are scalar
% input: gradients1, gradients2 belongs to a tensor w.r.t. the model structure

num_layers1=size(gradients1,1);
num_layers2=size(gradients2,1);

if num_layers1~=num_layers2
    error("the num_layers of the 2 gradient doesn't match.")
end

gradients_sum=gradients1;
for i=1:num_layers1
    gradients_sum.Value{i,1}=weight1*gradients1.Value{i,1} ...
        + weight2*gradients2.Value{i,1};
end

end