clear all; close all; warning off;
% % This main function covers the main process of the published paper:
%       [1]DCMDS-RV: density-concentrated multi-dimensional scaling for relation visualization
%       by Wu, B., Smith, J.S., Wilamowski, B.M. et al. Journal of Visualization (2018). 
%       https://doi.org/10.1007/s12650-018-0532-0
%
% % Welcome to cite the original article.
%
%

%% Iteration #
Max_ite = 5;
%% Choose Dataset
% DataFile = 'MNIST';        %--MNIST dataset
% DataFile = 'OlivettiFace'; %--Olivetti Face dataset
% DataFile = 'HAR';        %--Human activity recognition (HAR) using smartphone dataset
DataFile = 'text_50d';     %--Wikipedia 2014 words dataset
    
switch DataFile
    %% MNIST
    case 'MNIST',
        no_category=10;
        load './data/mnist_data.mat'
        no_point=1000;
        np=no_point;
        
        inp_data_model = 1;
        if inp_data_model > 0
            inp_Data=[img_0(:,1:no_point/no_category) img_1(:,1:no_point/no_category) ...
                img_2(:,1:no_point/no_category) img_3(:,1:no_point/no_category) ...
                img_4(:,1:no_point/no_category) img_5(:,1:no_point/no_category) ...
                img_6(:,1:no_point/no_category) img_7(:,1:no_point/no_category) ...
                img_8(:,1:no_point/no_category) img_9(:,1:no_point/no_category)];
            inp_Labels=[zeros(1,no_point/no_category) ones(1,no_point/no_category) ...
                2*ones(1,no_point/no_category) 3*ones(1,no_point/no_category) ...
                4*ones(1,no_point/no_category) 5*ones(1,no_point/no_category) ...
                6*ones(1,no_point/no_category) 7*ones(1,no_point/no_category) ...
                8*ones(1,no_point/no_category) 9*ones(1,no_point/no_category)];
            labels=inp_Labels;
        else
            inp_Data=images(:,1:no_point);
            train_labels = labels(1:no_point)'; %1*no_patterns
            labels=train_labels;
        end;
        
        inp_Data=inp_Data'; %np*ndim
        
       % Normalize input data
        X=inp_Data;
        X = X - min(X(:));
        X = X / max(X(:));
        X = bsxfun(@minus, X, mean(X, 1));
        
        inp_Data=X;
    %% Olivetti Face dataset
    case 'OlivettiFace',
        inp_all=[];
        labels=[];
        total_folder={'s1','s2','s3','s4','s5','s6','s7','s8','s9','s10',...
            's11','s12','s13','s14','s15','s16','s17','s18','s19','s20',...
            's21','s22','s23','s24','s25','s26','s27','s28','s29','s30',...
            's31','s32','s33','s34','s35','s36','s37','s38','s39','s40'};
        
        vis_folder_des=[1:1:10]; %specify desired folder for visualization
        for k=1:length(vis_folder_des)
            file_path = strcat('./data/olivetti_face/',total_folder(vis_folder_des(k)),'/');
            file_path = char(file_path);
            img_path_list = dir(strcat(file_path,'*.pgm'));
            img_num = length(img_path_list);
            if img_num > 0
                for j = 1:img_num
                    image_name = img_path_list(j).name;
                    image =  imread(strcat(file_path,image_name));
                    image=mat2gray(image);
                    [n_row,n_col]=size(image);
                    inp_all=[inp_all reshape(image,n_row*n_col,1)];
                    labels=[labels k];
                end
            end
        end
        images=inp_all;
        [ndim, np]=size(images);
        no_point=np;
        no_category = length(vis_folder_des);
        % Normalize input data
        inp_Data=images';
        X=inp_Data;
        X = X - min(X(:));
        X = X / max(X(:));
        X = bsxfun(@minus, X, mean(X, 1));
        
        inp_Data=X;  
    %% Human activity recognition (HAR) using smartphone dataset
    case 'HAR',
        x = load('./data/har.txt'); 
        label_y = load('./data/har_labels.txt');
        train_labels=label_y;
        
        no_point = 1000;
        train_X = x(1:no_point,:);
        labels = label_y(1:no_point);
        inp_Data = train_X;
        no_category = 6;
        % Normalize input data
        X = inp_Data;
        X = X - min(X(:));
        X = X / max(X(:));
        X = bsxfun(@minus, X, mean(X, 1));
        inp_Data=X;
    %% Wikipedia 2014 words dataset
    case 'text_50d',
        text_coordinates=open('./data/text_data_50d.mat');
        inp_Data=text_coordinates.text_data_5d;
        no_point=956;
        labels=ones(1,no_point);
        
        str = fileread('./data/text_words.txt');   %read entire file into string
        parts = strtrim(regexp( str, '(\r|\n)+', 'split'));  %split by each line
        parts{end}={};
end;
%% Define the output colormap
display_M1 = true;
if display_M1
    color_map = {[0.368 0.149 0.071],'b',[0 0.9 0],'c',...
        'm',[0.9 0.9 0.1],[0.3 0.4 0.5],[0.5 0.7 0.3],'r',[0.6 0.324 0.29],[0.6 0.2 0.7],...
        [0.2 0.9 0.4],[0.56 0.29 0.81],[0.13 0.59 0.33],[0.29 0.69 0.28]};
else
    color_map = hsv(no_category+1);
end;
%% Compute mutual distances
Dis_mat = pdist2(inp_Data,inp_Data);
[val_ori,ord_ori] = sort(Dis_mat,2);
%% Density-based clustering  
%
% calculate density, rho
%
ND = no_point;
N = 0.5*ND*(ND-1);
percent = 100;
%fprintf('average percentage of neighbours: %5.6f\n', percent);
position = round(N*percent/100);
xx_tmp = reshape(Dis_mat,2*N+ND,1);
sda = sort(xx_tmp);
dc = sda(position); % dc=30
%--- Compute Rho:local Density of data points in the paper ---%
rho=zeros(1,ND);
rho_mat=zeros(ND,ND);
%
%  Gaussian kernel
%
%fprintf('Computing Rho with gaussian kernel of radius: %12.6f\n', dc);
for i=1:ND-1
    for j=i+1:ND
        rho_tmp=exp(-(Dis_mat(i,j)/dc)*(Dis_mat(i,j)/dc));
        rho(i)=rho(i)+rho_tmp;
        rho(j)=rho(j)+rho_tmp;
        rho_mat(i,j)=rho_tmp;
    end
end
[val_rho,ord_rho]=sort(rho);
rho_mat=rho_mat+rho_mat';
[val_rho_mat,ord_rho_mat]=sort(rho_mat,2);
%% Statistical resutls analysis
% 
% generate decision graph
%
maxd=max(max(Dis_mat)); %maxium distance between points
% find the delta
[rho_sorted,ordrho]=sort(rho,'descend'); %sort density rho
delta(ordrho(1))=-1.;
nneigh(ordrho(1))=0;

for ii=2:ND
   delta(ordrho(ii))=maxd;
   for jj=1:ii-1
     if(Dis_mat(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))
        delta(ordrho(ii))=Dis_mat(ordrho(ii),ordrho(jj));
        nneigh(ordrho(ii))=ordrho(jj);
     end
   end
end
delta(ordrho(1))=max(delta(:));
%
% find the closest data to MaxRho-data
%
nneigh_of_maxRho=find(nneigh==ordrho(1));
[v_tmp,ord_tmp]=min(Dis_mat(ordrho(1),nneigh_of_maxRho));
nn_of_maxRho=nneigh_of_maxRho(ord_tmp);
%
% first nearest neighbor even closer
%
val = val_ori;
ord = ord_ori;
%% adjust the distance according to rho
mean_rho=median(rho);
for i=1:no_point
    if rho(i)<=0.9*rho(ord(i,2))
        val(i,2) = val(i,2)*(rho(i))^2/(rho(ord(i,2)))^2;
    end;
end;

val_area = val;
for i=1:no_point
    for j=2:no_point
        dis_mat_for_Eigen(i,ord_ori(i,j))=val_area(i,j);
    end;
end;
%% Get Position via Matrix Eigen Decomposition
%
% equations (4)-(11) in the original paper
%
m = no_point;
M = zeros(m,m);

A2=dis_mat_for_Eigen.^2;

ta2=sum(sum(A2));
for j=1:m
    tj(j)=sum(A2(:,j));
end;
for i=1:m
    ti=sum(A2(i,:));
    for j=1:m
        M(i,j)=(ti+tj(j)-A2(i,j)-ta2)/2;
    end;
end;
[V,D]=eigs(M);
%--- only select eigen-value >0 ---%
x=sqrt(D(2:3,2:3))*V(:,2:3)';
%--- show initial positions ---%
show_Initial_Positions = false;
if show_Initial_Positions
    figure(8),clf;plot(x(1,:),x(2,:),'ro');title('initial positions');
end;
%% 
P_mat = x';

p_considered = [no_point:-1:2];
no_equations = length(p_considered);
obs_1 = zeros(no_equations,1);

%% Density concentration
%
% calculate the moving rate, equation (17) in the original paper
%
move_rate=zeros(1,no_point);

for ord_i = 2:no_point
    
    inp_i=ord_rho(-ord_i+no_point+1); %from rho-max->min
    move_rate(inp_i)=(delta(inp_i))/((delta(ord_rho(end))));
    move_rate(inp_i)=move_rate(inp_i)*(rho(inp_i)/max(rho));%^border_tag;
    
    move_rate(inp_i)=1-move_rate(inp_i)*((no_point-ord_i)/no_point)^2; %
    if  move_rate(inp_i)>0.85
        move_rate(inp_i)= move_rate(inp_i)*2;
    end;
    
end;
%-- Normalize input data ---%
X=move_rate;
X=X/(max(X(:))^2-min(X(:))^2)-0.1-min(X(:))^2/(max(X(:))^2-min(X(:))^2);  %parabola normalization

move_rate=0.3*X+0.7*move_rate;
%% Main Loop
show_Results_in_Each_Iteration = false;
disp('Running start');
tic;
for ite = 2:Max_ite   
    if show_Results_in_Each_Iteration
        figure(1);clf;
    end;
    for ord_i = 2:no_point  %first CNTER is P_mat(1,:)        
        inp_i = ord_rho(-ord_i+no_point+1); %from rho-max->min
        
        xd0 = P_mat(inp_i,1);
        yd0 = P_mat(inp_i,2);
        xd01 = P_mat(ord(inp_i,2),1);
        yd01 = P_mat(ord(inp_i,2),2);
        
        p_dist = [P_mat(ord(inp_i,p_considered),1) P_mat(ord(inp_i,p_considered),2) val(inp_i,p_considered)'];
        %% move towards density centers 
        cnt_p_x = P_mat(nneigh(inp_i),1);
        cnt_p_y = P_mat(nneigh(inp_i),2);
        
        start_p_x = P_mat(inp_i,1);
        start_p_y = P_mat(inp_i,2);
        %--- larger move_rate will move point more closer to target cnt --%
        % equation (19) in the original paper
        %
        x_final2 = move_rate(inp_i)*cnt_p_x+(1-move_rate(inp_i))*start_p_x;
        y_final2 = move_rate(inp_i)*cnt_p_y+(1-move_rate(inp_i))*start_p_y;    
        %% LM, equation (12)-(16) in the original paper
        [ x_final,y_final,dis_tmp ] = LM_loc_individual_point_ori( p_dist, obs_1,xd0,yd0,xd01,yd01,20,1);
        
        [val_tmp,ind_tmp] = sort(dis_tmp(:,1),1);
        x_final = dis_tmp(ind_tmp(1),2);
        y_final = dis_tmp(ind_tmp(1),3);
        
        x_final = 0.35*x_final+0.5*x_final2+0.15*xd01;
        y_final = 0.35*y_final+0.5*y_final2+0.15*yd01;
 
        %% Update P_mat
        P_mat(inp_i,1) = x_final;
        P_mat(inp_i,2) = y_final;
        
        
        %--- Wikipedia 2014 words dataset ---%
%         if strcmp(DataFile,'text_50d') && ite==Max_ite-1
        if show_Results_in_Each_Iteration
            line([ x_final cnt_p_x],[ y_final cnt_p_y]);hold on;
        end;
        if strcmp(DataFile,'text_50d') && ite==Max_ite-1
            line([ x_final cnt_p_x],[ y_final cnt_p_y]);hold on;
        end;
    end;
    if show_Results_in_Each_Iteration
        figure(1);
        if ~strcmp(DataFile,'text_50d')
            for jj=1:no_category
                plot(P_mat(jj,1),P_mat(jj,2),'.','markersize',20,'color',cell2mat(color_map(jj)));hold on;
            end;
            %legend('0','1','2','3','4','5','6','7','8','9',-1); %'location','NorthEast');
            for jj=1:no_point
                plot(P_mat(jj,1),P_mat(jj,2),'.','markersize',20,'color',cell2mat(color_map(labels(jj)+1)));hold on;
            end;
            title('result');
        end;
    end;
    %--- plot the resulted relations for Wikipedia 2014 words dataset ----%
    if strcmp(DataFile,'text_50d') && ite==Max_ite-1
        figure(1);
        gscatter_BW(P_mat(:,1), P_mat(:,2), labels, [0 0 1;0 0.9 0;0 1 1;...
            1 0 1;0.9 0.9 0.1;0.3 0.4 0.5;0.5 0.7 0.3;1 0 0;0.6 0.324 0.29;0.6 0.2 0.7;...
            0.2 0.9 0.4;0.56 0.29 0.81;0.13 0.59 0.33;0.29 0.69 0.28],'.',1);hold on;
        title('result for Wikipedia 2014 words dataset');
        legend('off');
        for i_tmp_sep26=1:no_point
            text(P_mat(i_tmp_sep26,1),P_mat(i_tmp_sep26,2),parts{i_tmp_sep26},'FontSize',16);hold on;
        end;
    end;
    
    %
    % move Max-Rho data to its closest data
    %    
    inp_i = ord_rho(end);    
    mr_tmp = 0.9;
    
    cnt_p_x = P_mat(nn_of_maxRho,1);
    cnt_p_y = P_mat(nn_of_maxRho,2);
    
    start_p_x = P_mat(inp_i,1);
    start_p_y = P_mat(inp_i,2);
    %--- larger move_rate will move point more closer to target cnt ---%
    P_mat(inp_i,1) = mr_tmp*cnt_p_x + (1-mr_tmp)*start_p_x;
    P_mat(inp_i,2) = mr_tmp*cnt_p_y + (1-mr_tmp)*start_p_y;
    
    disp(['  --> iteration: ',num2str(ite), '/',num2str(Max_ite)]);
    
    %--- show Kruskal Stress factor in each iteration ---%
    % equation (20) in the original paper
    %
    show_Kruskal_Stress_Factor = false;
    if show_Kruskal_Stress_Factor
        stress_factor = Stress_VisEvaluate_BW(inp_Data,P_mat,0); % usually the last stress factor is the smallest
        disp(['Kruskal Stress factor is: ' num2str(stress_factor)]);
    end;
end;
t=toc;
disp('Done');
disp(['processing time: ' num2str(t)]);
%% Calculate the Kruskal stress factor
stress_factor = Stress_VisEvaluate_BW(inp_Data,P_mat,0); % usually the last stress factor is the smallest
disp(['Kruskal Stress factor is: ' num2str(stress_factor)]); 
%% Plot results
if ~strcmp(DataFile,'text_50d')
    figure(13),clf;
    for jj=1:no_category
        plot(P_mat(jj,1),P_mat(jj,2),'.','markersize',20,'color',cell2mat(color_map(jj)));hold on;
    end;
    legend('0','1','2','3','4','5','6','7','8','9',-1); %'location','NorthEast');
    for jj=1:no_point
        plot(P_mat(jj,1),P_mat(jj,2),'.','markersize',20,'color',cell2mat(color_map(labels(jj)+1)));hold on;
    end;
    title('Final result.');
end;
%% Photo On -- Olivetti Face dataset
if strcmp(DataFile,'OlivettiFace')
    figure(15);clf;
    range_pic_window=2.1;
    for jj=1:no_point
        plot(P_mat(jj,1),P_mat(jj,2),'.','markersize',0.1,'color',cell2mat(color_map(labels(jj)+1)));hold on;
        x_final = P_mat(jj,1);
        y_final = P_mat(jj,2);
        pic_tmp = reshape(images(:,jj),112,92);
        xx = [x_final-range_pic_window x_final+range_pic_window];
        yy = [y_final+range_pic_window y_final-range_pic_window];
        J = double(pic_tmp);
        K = repmat(pic_tmp,[1 1 3]);
        for i = 1:112
            for j = 1:92
                K(i,j,1) = 200*J(i,j)/255;
                K(i,j,2) = 200*J(i,j)/255;
                K(i,j,3) = 200*J(i,j)/255;
            end
        end
        imagesc(xx,yy,K);hold on;
    end;
end;