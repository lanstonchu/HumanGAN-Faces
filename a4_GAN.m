% D is now updated with G together in every loops in the GAN training

rng(123); % seed

%% load data

% middle: imds is the ImageDatastore; augimds is the resized version of imds

datasetFolder = 'faces/';

data_faces=load("mat_files/faces_matrix2.mat","filenames_face_keep");
filenames_face_keep=data_faces.filenames_face_keep;

file_paths=add_prefix_to_filenames(datasetFolder,filenames_face_keep);

% create ImageDatastore imds to easy loading of images;
imds = imageDatastore(file_paths);

% augimds is the image resized version of imds
augmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandScale',[1 2]);
augimds = augmentedImageDatastore([64 64],imds,'DataAugmentation',augmenter);

%% model construction
num_color_channels=1; % 3 for RGB and 1 for Grey
numLatentInputs = 100; % dimension of input noise of generator
[dlnetGenerator,dlnetDiscriminator,lgraphGenerator,lgraphDiscriminator]=model_construction(num_color_channels,numLatentInputs);
plot_model_structure(lgraphGenerator,lgraphDiscriminator);

%% Training
numEpochs = 1000;
miniBatchSize = 128;
iterations_select=[1, 100, 500, 1000, 2000, 4000]; % save weight for selected iterations

augimds.MiniBatchSize = miniBatchSize;
%[dlnetGenerator, dlnetDiscriminator]=model_training(numEpochs,miniBatchSize,augimds,numLatentInputs, ...
%      dlnetGenerator, dlnetDiscriminator,iterations_select,"mat_files/model_weights.mat");

%% Load weight and save X_plus_minus

num_data=25;
R=10;
sigma=1/100;

data_record=load("mat_files/model_weights.mat","record");
data_eigen=load("mat_files/eigen.mat","V","D");

V=data_eigen.V; % V: each column refers to 1 eigen face
D=data_eigen.D; % eigenvalues

q=100;
flag_eigen=1;

num_select=length(iterations_select);

for i=1:num_select
    
    itn_i=iterations_select(1,i);
    [dlnetGenerator, dlnetDiscriminator, iteration]=load_model_weights(data_record,itn_i,0);
    
    filename_X_plus_minus="mat_files/X_plus_minus_"+itn_i+".mat";
    filename_Z="mat_files/Z_for_queries_"+itn_i+".mat";
    filename_stateG="mat_files/state_G_"+itn_i+".mat";
    [~, ~,Sigma_Matrix_theta]=save_X_plus_minus_1_itn( ...
        dlnetGenerator,num_data,numLatentInputs, num_color_channels, R, ...
        sigma,filename_X_plus_minus,filename_Z,V,D,q,flag_eigen,filename_stateG);
    
end

filename_Sigma="mat_files/inv_Sigma_theta.mat";
% save_inverse_Sigma(Sigma_Matrix_theta,filename_Sigma);

%% tasks division

num_person=10;
tasks=tasks_division_1_itn(num_data, R,num_person,"mat_files/tasks.mat");

%% functions

function save_inverse_Sigma(Sigma_Matrix_theta,filename_Sigma)

inv_Sigma_Matrix_theta=inv(Sigma_Matrix_theta);

save(filename_Sigma,"inv_Sigma_Matrix_theta");

% also save as csv format for Numpy/Mathematica to check, because Sigma is closed
% to Singular with large size and thus its inverse is numerically unstable

csvwrite('Sigma_from_Matlab.csv',Sigma_Matrix_theta);
csvwrite('inv_Sigma_from_Matlab.csv',inv_Sigma_Matrix_theta);

end

function tasks=tasks_division_1_itn(num_data, R,num_person,filename)

% output: each row refers to the tasks for person_i.
% value -1 in the entry means "can be ignored"; just used for padding

num_total=num_data*R;

quotient=floor(num_total/num_person);
remainder=mod(num_total,num_person);

num_pad=mod(num_person-remainder,num_person); % take mod(), as it won't be num_person

list_length=(num_total+num_pad)/num_person;


shuffled_order=randperm(num_total);
left_vector=shuffled_order(1,1:(num_total-remainder));
right_vector=shuffled_order(1,(num_total-remainder+1):end);
right_vector_padded=[right_vector, repmat(-1, 1, num_pad)];

if num_pad==0
    left_matrix=reshape(left_vector,num_person,list_length);
else
    left_matrix=reshape(left_vector,num_person,list_length-1);
end
right_matrix=right_vector_padded';

tasks=[left_matrix, right_matrix];

save(filename,"tasks");

end

function [X_plus, X_minus,Sigma_Matrix_theta]=save_X_plus_minus_1_itn(dlnetGenerator,num_data, ...
    numLatentInputs,num_color_channels,R,sigma,filename_X_plus_minus,filename_Z,...
    V,D,q,flag_eigen,filename_stateG)

% input: V: each column refers to 1 eigen face
% input: q: number of eigenfaces being used
% input: R is the number of perturbation
% output: X_plus and X_minus are rank-5 tensors

executionEnvironment = "auto";

Z = randn(1,1,numLatentInputs,num_data,'single');
dlZ = dlarray(Z, 'SSCB');

% If training on a GPU, then convert data to gpuArray.
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlZ = gpuArray(dlZ);
end

[dlXGenerated,stateGenerator] = forward(dlnetGenerator,dlZ);
% dlXGenerated = predict(dlnetGenerator,dlZ); stateGenerator="Missed";

dlXGenerated_rank5=repmat(dlXGenerated,1,1,1,1,R); % extend dlXGenerated from rank 4 to rank 5

if flag_eigen==1
    
    V_select=V(:,1:q);
    diag_D=diag(D);
    
    mu=zeros(1,q);
    Sigma_Matrix_Zq=sigma^2*diag(diag_D(1:q));
    Sigma_Matrix_theta=V_select*Sigma_Matrix_Zq*V_select';
    
    Z_eigen_temp=mvnrnd(mu,Sigma_Matrix_Zq,num_data*R);
    Z_eigen=reshape(Z_eigen_temp',q,1,1,num_data,R); % here num_color_channels is 1
    
    VZq=multiprod(V_select,Z_eigen); % high dimensional mutiplication of (4096,q) and (q, 1, 1, N,R) to give (4096,1,1,N,R)
    delta_X = reshape(VZq,64,64,1,num_data,R);
    
elseif flag_eigen==0
    delta_X = sigma*randn(64,64,num_color_channels,num_data,R,'single');
else
    error("flag_eigen should be either 0 or 1")
end

X_plus = gather(extractdata(dlXGenerated_rank5 + delta_X));
X_minus = gather(extractdata(dlXGenerated_rank5 - delta_X));

save(filename_X_plus_minus,"X_plus","X_minus");
save(filename_Z,"dlZ");
save(filename_stateG,"stateGenerator");

end

function [dlnetGenerator,dlnetDiscriminator,lgraphGenerator,lgraphDiscriminator]...
    =model_construction(num_color_channels,numLatentInputs)

filterSize = [4 4];
numFilters = 64;

% construct generator
layersGenerator = [
    imageInputLayer([1 1 numLatentInputs],'Normalization','none','Name','in')
    transposedConv2dLayer(filterSize,8*numFilters,'Name','tconv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    transposedConv2dLayer(filterSize,4*numFilters,'Stride',2,'Cropping',1,'Name','tconv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    transposedConv2dLayer(filterSize,2*numFilters,'Stride',2,'Cropping',1,'Name','tconv3')
    batchNormalizationLayer('Name','bn3')
    reluLayer('Name','relu3')
    transposedConv2dLayer(filterSize,numFilters,'Stride',2,'Cropping',1,'Name','tconv4')
    batchNormalizationLayer('Name','bn4')
    reluLayer('Name','relu4')
    transposedConv2dLayer(filterSize,num_color_channels,'Stride',2,'Cropping',1,'Name','tconv5')
    tanhLayer('Name','tanh')];

lgraphGenerator = layerGraph(layersGenerator);

dlnetGenerator = dlnetwork(lgraphGenerator)

% construct discriminator

scale = 0.2;

layersDiscriminator = [
    imageInputLayer([64 64 num_color_channels],'Normalization','none','Name','in')
    convolution2dLayer(filterSize,numFilters,'Stride',2,'Padding',1,'Name','conv1')
    leakyReluLayer(scale,'Name','lrelu1')
    convolution2dLayer(filterSize,2*numFilters,'Stride',2,'Padding',1,'Name','conv2')
    batchNormalizationLayer('Name','bn2')
    leakyReluLayer(scale,'Name','lrelu2')
    convolution2dLayer(filterSize,4*numFilters,'Stride',2,'Padding',1,'Name','conv3')
    batchNormalizationLayer('Name','bn3')
    leakyReluLayer(scale,'Name','lrelu3')
    convolution2dLayer(filterSize,8*numFilters,'Stride',2,'Padding',1,'Name','conv4')
    batchNormalizationLayer('Name','bn4')
    leakyReluLayer(scale,'Name','lrelu4')
    convolution2dLayer(filterSize,1,'Name','conv5')];

lgraphDiscriminator = layerGraph(layersDiscriminator);

dlnetDiscriminator = dlnetwork(lgraphDiscriminator)

end

function plot_model_structure(lgraphGenerator,lgraphDiscriminator)

figure
subplot(1,2,1)
plot(lgraphGenerator)
title("Generator")

subplot(1,2,2)
plot(lgraphDiscriminator)
title("Discriminator")

end

function [dlnetGenerator, dlnetDiscriminator]=model_training(numEpochs,miniBatchSize,augimds,numLatentInputs, ...
    dlnetGenerator, dlnetDiscriminator,iterations_select,filename_weight)

% section: to train the model

learnRateGenerator = 0.0002;
learnRateDiscriminator = 0.0001;

trailingAvgGenerator = [];
trailingAvgSqGenerator = [];
trailingAvgDiscriminator = [];
trailingAvgSqDiscriminator = [];

gradientDecayFactor = 0.5;
squaredGradientDecayFactor = 0.999;

freq_visualize = 10;
% freq_save_weight = 500;

executionEnvironment = "auto";

ZValidation = randn(1,1,numLatentInputs,64,'single'); % show the same 64 images whenever need to visualize result
dlZValidation = dlarray(ZValidation,'SSCB'); % used the same dlZValidation to generate images in every 100 iterations

if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlZValidation = gpuArray(dlZValidation);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
iteration = 0;
start = tic;

% Loop over epochs.
record_i=1;
for i = 1:numEpochs
    
    % Reset and shuffle datastore.
    reset(augimds);
    augimds = shuffle(augimds);
    
    % Loop over mini-batches.
    while hasdata(augimds)
        
        % Read mini-batch of data.
        data = read(augimds);
        
        % Ignore last partial mini-batch of epoch.
        if size(data,1) < miniBatchSize
            continue
        end
        
        iteration = iteration + 1;
        
        % Concatenate mini-batch of data and generate latent inputs for the
        % generator network.
        X = cat(4,data{:,1}{:}); % column 1 refers to the data, 2 refers to file info
        Z = randn(1,1,numLatentInputs,size(X,4),'single');
        
        % Normalize the images
        X = (single(X)/255)*2 - 1;
        
        % Convert mini-batch of data to dlarray specify the dimension labels
        % 'SSCB' (spatial, spatial, channel, batch).
        dlX = dlarray(X, 'SSCB');
        dlZ = dlarray(Z, 'SSCB');
        
        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
            dlZ = gpuArray(dlZ);
        end
        
        % Evaluate the model gradients and the generator state, by plugging
        % the modelGradients() into dlfeval()
        
        % dlfeval() will link up the variables (i.e. inside the say GPU) 
        % in the sense of "pass by reference" instead of "pass by values"
        
        [gradientsGenerator, gradientsDiscriminator, stateGenerator] = ...
            dlfeval(@modelGradients, dlnetGenerator, dlnetDiscriminator, dlX, dlZ);
        dlnetGenerator.State = stateGenerator;
        
        % Update the discriminator network parameters.
        [dlnetDiscriminator.Learnables,trailingAvgDiscriminator,trailingAvgSqDiscriminator] = ...
            adamupdate(dlnetDiscriminator.Learnables, gradientsDiscriminator, ...
            trailingAvgDiscriminator, trailingAvgSqDiscriminator, iteration, ...
            learnRateDiscriminator, gradientDecayFactor, squaredGradientDecayFactor);
        
        % Update the generator network parameters.
        [dlnetGenerator.Learnables,trailingAvgGenerator,trailingAvgSqGenerator] = ...
            adamupdate(dlnetGenerator.Learnables, gradientsGenerator, ...
            trailingAvgGenerator, trailingAvgSqGenerator, iteration, ...
            learnRateGenerator, gradientDecayFactor, squaredGradientDecayFactor);
        
        % Every 100 iterations, display batch of generated images using the
        % held-out generator input.
        if mod(iteration,freq_visualize) == 0 || iteration == 1
            
            % Generate images using the held-out generator input.
            dlXGeneratedValidation = predict(dlnetGenerator,dlZValidation);
            
            % Rescale the images in the range [0 1] and display the images.
            I = imtile(extractdata(dlXGeneratedValidation));
            I = rescale(I);
            image(I,'CDataMapping','scaled');
            colormap(gray);
            
            % Update the title with training progress information.
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            title(...
                "Epoch: " + i + ", " + ...
                "Iteration: " + iteration + ", " + ...
                "Elapsed: " + string(D))
            
            drawnow
        end
        
        if any(iterations_select==iteration) % save weight for selected iterations
            % recreate a structure variable called "record_i"
            record(record_i).dlnetGenerator=dlnetGenerator;
            record(record_i).dlnetDiscriminator=dlnetDiscriminator;
            record(record_i).iteration=iteration;
            record_i=record_i+1;
            save(filename_weight,"record");
        end
    end
end

end

function [gradientsGenerator, gradientsDiscriminator, stateGenerator] = ...
    modelGradients(dlnetGenerator, dlnetDiscriminator, dlX, dlZ)

% input: dlnetGenerator and dlnetDiscriminator are "dlnetwork" object, which includes layers' structure and layers' weights.
% input: X is a 64x64x1x128 (i.e. "SSCB") array; dlX = dlarray(X);
% input: Z is a 1x1x100x128 (i.e. "SSCB") array; dlZ = dlarray(Z);
% middle: dlXGenerated: 64x64x1x128 dlarray
% middle: dlYPred, dlYPredGenerated: 1x1x1x128 dlarray in (-inf, inf);
% middle: lossGenerator, lossDiscriminator: scalar
% output: gradientsGenerator and gradientsDiscriminator are 18x3 and 16x3 tables, which include the gradient of the layers' weights


% Calculate the predictions for real data with the discriminator network.
dlYPred = forward(dlnetDiscriminator, dlX);

% Calculate the predictions for generated data with the discriminator network.
[dlXGenerated,stateGenerator] = forward(dlnetGenerator,dlZ);
dlYPredGenerated = forward(dlnetDiscriminator, dlXGenerated);

% Calculate the GAN loss
[lossGenerator, lossDiscriminator] = ganLoss(dlYPred,dlYPredGenerated);

% For each network, calculate the gradients with respect to the loss.
% "RetainData" is used in dG to speed up the next step of dD which reuse dlgradient()
gradientsGenerator = dlgradient(lossGenerator, dlnetGenerator.Learnables,'RetainData',true);
gradientsDiscriminator = dlgradient(lossDiscriminator, dlnetDiscriminator.Learnables);

end

function [lossGenerator, lossDiscriminator] = ganLoss(dlYPred,dlYPredGenerated)

% input: dlYPred, dlYPredGenerated: 1x1x1x128 dlarray in (-inf, inf)
% therefore "D(x) or D(G(z))" = signoid(Y) can give output of [0, 1]
% in below code, L_D = -E[log D(x)] -E[log (1 - D(G(z)))]
%                L_G = -E[log D(G(z))]
% D(x) = 1 means "D thinks that x is real"
% therefore outside this function, we need min(L_D) and min(L_G)


% Calculate losses for the discriminator network.
lossGenerated = -mean(log(1-sigmoid(dlYPredGenerated)));
lossReal = -mean(log(sigmoid(dlYPred)));

% Combine the losses for the discriminator network.
lossDiscriminator = lossReal + lossGenerated;

% Calculate the loss for the generator network.
lossGenerator = -mean(log(sigmoid(dlYPredGenerated)));

end

function paths=add_prefix_to_filenames(folder_path,filenames_face_keep)

% input: filenames_face_keep is a Mx1 cell
% output: paths is a 1xN cell

num_faces=size(filenames_face_keep,1);

paths={};
for i=1:num_faces
    filename_i= filenames_face_keep(i,1);
    path_i=join([folder_path, filename_i],'');
    paths{end+1}=char(path_i);
end


end