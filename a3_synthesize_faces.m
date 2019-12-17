% this program is used to synthesize some faces by low dimensional noises
% to get a sigificance-feeling of the impact of changing weight for eigenfaces

close all

h_img=64;
w_img=64;

data = load("mat_files\eigen.mat");

% V: each column is 1 eigenface
V=data.V;
D=data.D;
diag_D=diag(D);

rng(123); % seed

q=580;
mu=zeros(1,q);
sigma=diag(diag_D(1:q));

faces_q = V(:,1:q);

num_Z=100;
Z=mvnrnd(mu,sigma,num_Z);

show_many_newfaces(faces_q,h_img,w_img,Z);
show_1_newface(faces_q,h_img,w_img,Z,1);

function show_many_newfaces(eigen_faces,h_img,w_img,Z)

% Z is a matrix; #(rows) = #(noise); #(columns) = dim(noise);

num_Z=size(Z,1);
new_faces = eigen_faces*Z';

image_combine=zeros(h_img*10,w_img*10);
for i=0:9
    for j=0:9
        index=10*i+j+1;
        
        if index<=num_Z
            face=reshape(new_faces(:,index),h_img,w_img);
            image_combine((h_img*i+1):(h_img*(i+1)),(w_img*j+1):(w_img*(j+1)))=face;
        end
        
    end
end

figure
colormap(gray);
img=imagesc(image_combine)
title("64 random synthesized faces");
filename=['plots/a3_synthesized_faces.png'];
saveas(img,filename);

figure
imshow(image_combine);

end

function show_1_newface(eigen_faces,h_img,w_img,Z,index)

% Z is a matrix; #(rows) = #(noise); #(columns) = dim(noise);

num_Z=size(Z,1);
new_faces = eigen_faces*Z';

image_combine=uint8(zeros(h_img*1,w_img*1));

face=reshape(new_faces(:,index),h_img,w_img);
image_combine=face;

figure
colormap(gray);
imagesc(image_combine);

figure
imshow(image_combine);

end