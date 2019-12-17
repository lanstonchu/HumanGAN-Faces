% this m-file is used to manually kick out images with unwanted angles

% Part A: Manually change big_i and selected unwanted images
% to store unwatned images in ij_select

% [x_select,y_select] = ginput(20)
% 
% big_i=6
% i_select = ceil(y_select/64);
% j_select = ceil(x_select/64);
% % ij_select=zeros(20,2,7);
% ij_select(:,:,big_i+1)=[i_select, j_select];
% ij_select(:,:,big_i+1)

% Part B: Manually unhide. 
% To combind ij_select as index_kick

index_kick=[];
for big_i=0:6
    index_100=big_i*100+(ij_select(:,1,big_i+1)-1)*10+ij_select(:,2,big_i+1);
    index_kick=[index_kick; index_100];
end

index_kick
save("mat_files/index_kick.mat","index_kick")