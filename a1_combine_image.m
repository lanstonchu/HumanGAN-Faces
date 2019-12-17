% this m-file is used to sample images randomly, to combine images and to show images

time_start=tic;

close all

path_folder="faces/";
files=dir(path_folder+'*.*');
h_img = 64;
w_img = 64;

% get the names of all file-0001
files_select=[""];
for k=1:length(files)
    name=files(k).name;
    if length(name)<7
        2+2;
    else
        if name((end-7):end)=="0001.pgm"
            files_select=[files_select;name];
        end
    end
    
end
files_select = files_select(2:end);


% show combined images with unwanted images
s = RandStream('mlfg6331_64'); % seed
n_file = length(files_select);
filenames_face_700 = datasample(s,files_select,700,'Replace',false);

big_i=0;
show_combine_img(path_folder,big_i,filenames_face_700,h_img,w_img);

% to kick out the unwanted images
% load("mat_files/index_kick.mat");
load("mat_files/index_kick.mat");
index_keep = setdiff(1:700,index_kick);
filenames_face_keep=filenames_face_700(index_keep);

big_i=2;
show_combine_img(path_folder,big_i,filenames_face_keep,h_img,w_img);
% after first round kick: 587 images left in the 700 faces

faces_matrix=images_2_matrix(path_folder,filenames_face_keep,h_img,w_img);
save("mat_files/faces_matrix2.mat","faces_matrix","filenames_face_keep");

time_end=toc(time_start)

function show_combine_img(path_folder,big_i,faces,h_img,w_img)

image_combine=uint8(zeros(h_img*10,w_img*10));
num_face=length(faces);

for i=0:9
    for j=0:9
        index=big_i*100+10*i+j+1;
        
        if index<=num_face
        face = imread(path_folder+faces(index));
        image_combine((h_img*i+1):(h_img*(i+1)),(w_img*j+1):(w_img*(j+1)))=face;
        end
    end  
end

imshow(image_combine)

end

function faces_matrix=images_2_matrix(path_folder,faces,h_img,w_img)

% input: faces is a column vector of string (i.e. file names)
% output: faces_matrix stores the images value. each row refers to 1 face

num_faces=length(faces);

faces_matrix=zeros(num_faces,h_img*w_img);
for i=1:num_faces
    face = imread(path_folder+faces(i));
    faces_matrix(i,:)=reshape(face,[1,h_img*w_img]);
end

end