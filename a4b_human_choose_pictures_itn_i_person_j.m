function [D_scores, user_name, task_person_j]=a4b_human_choose_pictures_itn_i_person_j(...
    filename_X_plus_minus,tasks,person_j,D_scores)

% input: tasks may contains -1 at the end, which means "to be ignored"
% output: entries of D_scores range from -1 to +1;
% -1 means LHS is more realistic; +1 means RHS is more realistic
% D_scores is a NxR matrix

% pictures: Left image is X_minus; Right image is X_plus

data=load(filename_X_plus_minus,"X_plus","X_minus");
X_plus=data.X_plus;
X_minus=data.X_minus;
% X_plus and X_minus are rank 5 tensors, i.e. (H, W, C, Batch, #(ptb))

num_batch=size(X_plus,4);
num_ptb=size(X_plus,5);
num_total=num_batch*num_ptb;

task_person_j=tasks(person_j,:);
% task_person_j=tasks(person_j,1:5);

h_grey=7;
h_blk=16;
w_blk=16;
num_blk=8;
bottom_graph=draw_bar(h_grey,h_blk,w_blk,num_blk);

margin_scale=(1-(-1))/num_blk/2; % D ranges from -1 to 1
D_scale=linspace(-1 + margin_scale,1 - margin_scale,num_blk);


user_name = input('Please input your name (initial is OK): ', 's');

waitfor(warndlg('I hope you won''t be scared by the coming images.','Warning'));

fig=figure
switch_mistake=0; % either 0 or 1
index_show=0;
for i=1:num_batch
    for j=1:num_ptb
        index=(i-1)*num_ptb+j;
        if any(task_person_j==index)
            num_total_show=length(setdiff(task_person_j,-1));
            index_show=index_show+1;
            face_plus = X_plus(:,:,:,i,j);
            face_minus = X_minus(:,:,:,i,j);
            face_minus_plus=[face_minus, face_plus];
            plot_pairs=[face_minus_plus;bottom_graph];
            
            % show images and adjust windows size
            imshow(plot_pairs,'InitialMagnification','fit');
            if index_show==1
                position_ori = get(gcf, 'Position');
                position_wider = position_ori + [0, 0, 100, 0];
            end
            set(gcf, 'Position',  position_wider);
            
            % imshow(plot_pairs,'InitialMagnification','fit');
            title([index_show+" of "+num_total_show+...
                " pairs; (N, R, idx): ("+i+", "+j+", "+index+")",...
                "Which one is more realistic? (Click the bar)", ...
                ""])
            annotation('textbox',[0.08, 0.05, .1, .2],'String','Left is more realistic','EdgeColor','none')
            annotation('textbox',[0.87, 0.05, .1, .2],'String','Right is more realistic','EdgeColor','none')
            
            flag_loop=1;
            while flag_loop==1
                [x_select,y_select] = ginput(1);
                
                if x_select>=1 && x_select<=size(plot_pairs,2) && ...
                        y_select>=(size(face_minus_plus,1)+h_grey+1) && y_select<=size(plot_pairs,1)
                    % cursor position accepted
                    
                   score=ceil(x_select/h_blk);
                   D_score=D_scale(score);
                   D_scores(i,j)=D_score;
                    
                    message="(Index, Score, x, y): (" ...
                        +index_show+", "+score+", "+x_select+", "+y_select+")";
                    disp(message);
                    flag_loop=0;
                else
                    % cursor position rejected
                    
                    switch_mistake=mod(switch_mistake+1,2);
                    space=strjoin(repmat(" ", 1, switch_mistake+1));
                    title([index_show+" of "+num_total_show+...
                        " pairs; (N, R): ("+i+", "+j+")",...
                        "Which one is more realistic? (Click the bar)", ...
                        space+"PLEASE CLICK THE SCALE BAR"])
                    flag_loop=1;
                end
            end
        end
    end
end
close(fig)

end

function bottom_graph=draw_bar(h_grey,h_blk,w_blk,num_blk)

blk=ones(h_blk,w_blk);
grey_bar=0.5*ones(h_grey,w_blk*num_blk);

scale_bar_values=linspace(0,1,num_blk);

scale_bar=[];
for i=1:num_blk
    scale_bar=[scale_bar, scale_bar_values(i)*blk];
end

bottom_graph=[grey_bar; scale_bar];

end


