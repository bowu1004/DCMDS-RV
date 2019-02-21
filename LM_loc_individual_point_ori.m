function [ x_final,y_final ,dis_tmp] = LM_loc_individual_point_ori( p_dist, obs_1,xd0,yd0,xd01,yd01,nneigh,mode )
%LM_loc_center Using LM method to locate the next center in 2D space
%   Details:
% e.g. p_dist=[4 4 6];%point(x,y) and distance to,e.g.xD=4;yD=4;D=6;
% obs_1=[0]; %values of equations
% %original guess. Also treat as the stard point:
% (xd0,yd0);
% y_init=(sqrt((xd0-xD)^2+(yd0-yD)^2)-D)^2;

%% Parameters
Ndata=length(obs_1); %number of equations
Nparams=2; %target dimensions
%% 1,simple and fast
% the smaller,more local properties preserved;
% the larger,results go towards to the global optimization.
n_iters=3; % 6 maximum iterations.
%%
lamda=0.05; %LM tuning parameter
% lamda=100*rand(1);
%% Iterations
% step1: search start point
updateJ=1;
a_est=xd0;
b_est=yd0;

% step2: LM iterations
dis_tmp=[];
val_tmp=sum(((a_est-p_dist(:,1)).^2+(b_est-p_dist(:,2)).^2)-p_dist(:,3));
%val_tmp=(xd01-a_est)^2+(yd01-b_est)^2;
dis_tmp=[dis_tmp;val_tmp a_est b_est];
% cu=zeros(Ndata,Ndata);
% for i=1:Ndata
%     pu(i,i)=0.1*rand(1);
% end;

dis_vector=1;


J=zeros(Ndata,Nparams);
for it=1:n_iters
    % add lamda so that it keeps finding the minimal during iterations
    if updateJ==1
        % Find Jacobian Matrix  
        rho_=sum(exp(-(a_est-p_dist(:,1)).^2./dis_vector^2-(b_est-p_dist(:,2)).^2./dis_vector^2-1/dis_vector^2))-1;
        J=[((2*p_dist(:,1)-2*a_est).*(p_dist(:,3)-((p_dist(:,1)-a_est).^2+(p_dist(:,2)-b_est).^2).^(1/2)))./((p_dist(:,1)-a_est).^2 +(p_dist(:,2)-b_est).^2).^(1/2)+2*rand(1)-1,...
            ((2*p_dist(:,2)-2*b_est).*(p_dist(:,3)-((p_dist(:,1)-a_est).^2+(p_dist(:,2)-b_est).^2).^(1/2)))./((p_dist(:,1)-a_est).^2+(p_dist(:,2)-b_est).^2).^(1/2)+2*rand(1)-1];
            
%             B=[(-1)*sum((2*p_dist(:,1)-2*a_est).*exp(-(p_dist(:,1)-a_est).^2./dis_vector^2-(p_dist(:,2)-b_est).^2./dis_vector^2-1/dis_vector^2))/rho_^2/dis_vector^2,...
%             (-1)*sum((2*p_dist(:,2)-2*b_est).*exp(-(p_dist(:,1)-a_est).^2./dis_vector^2-(p_dist(:,2)-b_est).^2./dis_vector^2-1/dis_vector^2))/rho_^2/dis_vector^2];
      
%             J=[J;B];
        % target function
        y_est = (sqrt((a_est-p_dist(:,1)).^2+(b_est-p_dist(:,2)).^2)-p_dist(:,3)).^2;
%         y_est=[y_est;100];
        % error
        d=obs_1-y_est;
                
        % Hessian
        H=J'*J;
        if updateJ==1
            e=dot(d,d);
        end
    end
    H_lm=H+lamda*eye(Nparams,Nparams);
    %
    g=J'*d;
    dp=inv(H_lm)*g;
    a_lm=a_est+dp(1);
    b_lm=b_est+dp(2);
    % results of target function and errors
    y_est_lm =(sqrt((a_lm-p_dist(:,1)).^2+(b_lm-p_dist(:,2)).^2)-p_dist(:,3)).^2;
%     y_est_lm=[y_est_lm;100];
    d_lm=obs_1-y_est_lm;
    e_lm=dot(d_lm,d_lm);
    % rules of changing lamda
    if e_lm<e
        lamda=lamda/10;
        a_est=a_lm;
        b_est=b_lm;
        e=e_lm;
        %disp(e);
        updateJ=1;
    else
        updateJ=0;
        lamda=lamda*10;
    end

    %% For sort
    val_tmp=sum(((a_est-p_dist(:,1)).^2+(b_est-p_dist(:,2)).^2)-p_dist(:,3));
    if mode==1 %when locating individual patterns
        dis_tmp=[dis_tmp;(xd01-a_est)^2+(yd01-b_est)^2 a_est b_est]; %for locate individual patterns
    elseif mode==2 %when locating centers
        dis_tmp=[dis_tmp;val_tmp a_est b_est]; %for locate centers
    elseif mode==3 %when locating centers
        dis_tmp=[dis_tmp;sum((p_dist(end-nneigh:end,1)-a_est).^2+(p_dist(end-nneigh:end,2)-b_est).^2)+0.1*((xd0-a_est)^2+(yd0-b_est)^2) a_est b_est]; %for locate centers
    else
        dis_tmp=[dis_tmp;0.9*(abs(xd01-a_est)+abs(yd01-b_est))+0.1*val_tmp a_est b_est]; %for locate individual patter
    end;
    
%     if val_tmp>dis_tmp(it)
%         break;
%     end;
end
%final results for x and y
x_final=a_est;
y_final=b_est;

end