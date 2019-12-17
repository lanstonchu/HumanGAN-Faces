function main()

close all

% X: each row is one image

data=load("mat_files\faces_matrix2.mat");
X_train = data.faces_matrix;

h_img_ori=64;
w_img_ori=64;
ratio_resize=1;
Q=1000; % number of eigenfaces stored in the .mat file
q=400; % number of eigenfaces whose coordinates will be recorded; i.e. q <= Q

[X_train_resized,h_img_new,w_img_new]=imgResize(X_train,h_img_ori,w_img_ori,ratio_resize);

tStart = tic;

% Part a: PCA
[V,D]=PCA(X_train_resized); % V: each column is 1 eigenface
X_tilde_U=PCA_coord(X_train_resized,V,q);
% [V, D, X_tilde_U]=save_load_eigen_faces_values(V,D,X_tilde_U,"mat_files/eigen.mat",Q,1); % save eigenfaces
[V, D, X_tilde_U]=save_load_eigen_faces_values("Nil","Nil","Nil","mat_files/eigen.mat","Nil",0); % load eigenfaces
show_eigen_faces(V,D,h_img_new,w_img_new,10);

% Part b: plot eigenvalues/variance explained
plot_eigenvalue(D)

% Part c: test some faces reconstruction
ns=[102, 295]; % files to be picked for reconstruction testing
face_reconstruction(X_train_resized,V,X_tilde_U,ns,h_img_new,w_img_new);

tElapsed = toc(tStart)

end

function face_reconstruction(X,V,X_tilde_U,ns,h_img,w_img)
% this function is for face reconstruction, to confirm that the PCA is fine

% input: X: resized original faces; each row refers to 1 face
% input: V: eigenfaces; each column refers to 1 face
% input: X_tilde_U: U coordinates
% input: ns: to reconstruct face of index ns = [n1, n2, ..., nN]

N=size(ns,2);
q=size(X_tilde_U,2);
dim_V = size(V,2);

U_coord=X_tilde_U(ns,:);
U_coord_padded=[transpose(U_coord);zeros(dim_V-q,N)];
new_faces=V*U_coord_padded;

for i=1:N
    face_num_i=ns(1,i);
    
    fig=figure
    subplot(1,2,1);
    colormap(gray);
    face1=reshape(X(face_num_i,:),h_img,w_img);
    imagesc(face1);
    title(["Original Face:", num2str(face_num_i)]);
    
    subplot(1,2,2);
    colormap(gray);
    face2=reshape(new_faces(:,i),h_img,w_img);
    imagesc(face2);
    title(["Reconstructed Face:", num2str(face_num_i)]);
    
    position = get(gcf, 'Position');
    position(1,3) = 1.8*position(1,3); % wider windows
    set(gcf, 'Position',  position);
    
    filename=["plots/a2_PartC_face_reconstruction_"+face_num_i+".png"];
    saveas(fig,filename);
    
end

end

function X_tilde_U=PCA_coord(X,V,q)

% to give coordinates of X based on selected eigenvectors

% input: X: resized original faces; each row refers to 1 face
% input: V: eigenfaces; each column refers to 1 face
% input: q: reduced dimension

X_tilde=X_recentered(X);

% get the top eigenvectors; from high to low
top_eig_vec=V(:,1:q);

X_tilde_U=X_tilde*top_eig_vec; %n*q matrix

end

function Dist=dist_matrix(X1,X2)
% input: both X1 and X2 have p columns
% input: X1 and X2 are not necessarily having same number of rows

% output: distance between X1 and X2 in outer product manner
% output: X1 would be vertical; X2 would be horizontal

n1=size(X1,1);
n2=size(X2,1);

Dist=zeros(n1,n2);
for i=1:n1
    for j=1:n2
        Dist(i,j)=norm(X1(i,:)-X2(j,:));
    end
end

end

function plot_eigenvalue(D)

lambdas=diag(D)'; % eigenvalue in decreasing order
n=size(lambdas,2);
lambdas_sum = cumsum(lambdas);
L_sum=lambdas_sum(1,end);
var_explained=lambdas_sum/L_sum;

figure
img=plot(1:n,lambdas)
title('elbow plot: The i-th Eigenvalues ')
xlabel('i')
ylabel('i-th Eigenvalue')
filename=['plots/a2_PartB_Eigenvalues_Plot1.png'];
saveas(img,filename);

figure
img=plot(1:n,lambdas)
title('elbow plot: The i-th Eigenvalues (Zoomed In)')
xlabel('i')
ylabel('i-th Eigenvalue')
axis([0 150 0 180000])
filename=['plots/a2_PartB_Eigenvalues_Plot2.png'];
saveas(img,filename);

figure
img=plot(1:n,var_explained)
title('Part B: Variance explained by i eigenvectors')
xlabel('i')
ylabel('Variance explained')
filename=['plots/a2_PartB_VarExplained_Plot3.png'];
saveas(img,filename);

end

function [X_resized,h_img_new,w_img_new]=imgResize(X,h_img_ori,w_img_ori,ratio)

if ratio==1
    % do nothing
    X_resized=X;
    h_img_new=h_img_ori;
    w_img_new=w_img_ori;
    
elseif ratio<1
    % input: each row is one image
    
    [n, m]=size(X);
    
    X_resized=[];
    for i=1:n
        img=reshape(X(i,:),h_img_ori,w_img_ori);
        img_resized=imresize(img,ratio);
        [h_img_new,w_img_new]=size(img_resized);
        X_i_resized_row=reshape(img_resized,1,h_img_new*w_img_new);
        X_resized=[X_resized;X_i_resized_row];
    end
    
else
    print('error in imgResize()')
    
end
end

function show_eigen_faces(V,D,h_img,w_img,num_face)

% input V: each column is 1 eigenface
% input D: eigenvalues

top_eig_vec=V(:,1:num_face);

for i=1:num_face
    figure
    colormap(gray);
    img=imagesc(reshape(top_eig_vec(:,i),h_img,w_img));
    title([num2str(i),'-th eigenface']);
    filename=['plots/a2_PartA_eigenface',num2str(i,'%03.f'),'.png'];
    saveas(img,filename);
end

end

function [V, D,X_tilde_U]=save_load_eigen_faces_values(V,D,X_tilde_U,file_name,Q,save_load_flag)

% input V: each column is 1 eigenface
% input D: eigenvalues
% save_load_flag = 1 for save
% save_load_flag = 0 for load

if save_load_flag==1 % save
    V=V(:,1:Q);
    D=D(1:Q,1:Q);
    save(file_name, 'V','D','X_tilde_U');
    
elseif save_load_flag==0 % load
    data = load(file_name);
    V=data.V; % eigenvectors
    D=data.D; % eigenvalues
    X_tilde_U = data.X_tilde_U; % U coordinates
else
    error("save_load_flag should be either 1 or 0")
end

end

function [V_ordered,D_ordered] = PCA(X)
% V_ordered,D_ordered will be in descending order
% output V: each column is 1 eigenface

order_flag = 1; % 1: descending order

[n, m]=size(X);

X_tilde=X_recentered(X);

S=X_tilde'*X_tilde/n;

[V,D] = eig(S);

[V_ordered,D_ordered,D_diag_ordered]=ordering_matrix(V,D,order_flag);

end

function [V_ordered,D_ordered,D_diag_ordered]=ordering_matrix(V,D,order_flag)

% input: D should be diagonal matrix
% input: order_flag can be 0 or 1. 0 -> ascending; 1 -> descending

if isdiag(D)
    2; % do nothing
else
    error('D should be diagonal matrix')
end

D_diag=diag(D); % colume vector

if order_flag==0
    [D_diag_ordered, idx]=sort(D_diag);
elseif order_flag==1
    [D_diag_ordered, idx]=sort(D_diag,'descend');
else
    error('order_flag should be either 0 or 1')
end

D_ordered=diag(D_diag_ordered);
V_ordered=V(:,idx);

end

function X_tilde=X_recentered(X)

[n, m]=size(X);
X_mean_row=mean(X);
X_mean=repmat(X_mean_row,n,1);
X_tilde=X-X_mean;

end