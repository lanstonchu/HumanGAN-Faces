function [dlnetGenerator, dlnetDiscriminator, iteration]=load_model_weights(data_record,iteration,flag_G_only)

% input: flag_G_only = 1 means "to load G only";
% flag_G_only = 0 means " to load both G and D"

% to extract the model weights of specific iteration

iterations=[data_record.record.iteration]; % get all iterations data
index=find(iterations == iteration); % search for the specific iteration

if size(index,2)==1
    dlnetGenerator=data_record.record(index).dlnetGenerator;
    
    if flag_G_only ==0
        dlnetDiscriminator=data_record.record(index).dlnetDiscriminator;
    elseif flag_G_only ==1
        dlnetDiscriminator="Nil";
    else
        error("flag_G_only should be either 0 or 1")
    end
    
    iteration=data_record.record(index).iteration;
    
elseif size(index,2)==0
    error("The iteration mentioned doesn't exist.")
    
else
    error("Unexpected error")
end

end