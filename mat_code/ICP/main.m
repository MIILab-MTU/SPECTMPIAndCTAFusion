% 程序说明：输入data_source和data_target两个点云，找寻将data_source映射到data_targe的旋转和平移参数
clear;
close all;
clc;
%% 参数配置
kd = 1;
inlier_ratio = 0.999;
Tolerance = 0.05;
step_Tolerance = 0.01;
max_iteration =100;
show = 1;
 
%% 生成数据
% data_source=pcread('C:\Users\Administrator\Desktop\vescl\ply\adis800_pc_wave.ply');
% data_target = pcread('C:\Users\Administrator\Desktop\vescl\result1\scene3_dis500_ref400\sparse_point_cloud.ply');
% data_source=pcread('C:\Users\Administrator\Desktop\vescl\ply\adis500_pc_egg.ply');
% data_target = pcread('C:\Users\Administrator\Desktop\vescl\result1\scene4_dis500_ref400\sparse_point_cloud.ply');
% data_source=pcread('C:\Users\Administrator\Desktop\vescl\ply\adis500_pc_step.ply');
% data_target = pcread('C:\Users\Administrator\Desktop\vescl\result1\scene2_dis500_ref400\sparse_point_cloud.ply');
% data_source=pcread('C:\Users\Administrator\Desktop\vescl\ply\adis500_pc_plane.ply');
% data_target = pcread('C:\Users\Administrator\Desktop\vescl\result1\scene1_dis500_ref400\sparse_point_cloud.ply');
 
% data_source=data_source.Location';
% data_target=data_target.Location';

patientname = 'zengyuming';
load(sprintf('F:/cta-spect/CTA-SPECT/data_process/cta/final/%s/ijkcta.mat',patientname));
load(sprintf('E:/proj/peizhun/file_process/test425/%s/cu.txt',patientname));
%load(sprintf('F:/cta-spect/CTA-SPECT/result/%s/cu.txt',patientname));
data_source = cu';
data_target = ijkcta';

% 绘制原始点与旋转后的点图像
figure;
scatter3(data_source(1,:),data_source(2,:),data_source(3,:),'b.');
hold on;
scatter3(data_target(1,:),data_target(2,:),data_target(3,:),'r.');
hold off;
daspect([1 1 1]);
 
%% 开始ICP
T_final=eye(4,4);   %旋转矩阵初始值
iteration=0;
Rf=T_final(1:3,1:3);
Tf=T_final(1:3,4);
data_source=Rf*data_source+Tf*ones(1,size(data_source,2));    %初次更新点集（代表粗配准结果）
err=1;
data_source_old = data_source;
%% 迭代优化
while(1)
    iteration=iteration+1;
    if kd == 1
        %利用Kd-tree找出对应点集
        kd_tree = KDTreeSearcher(data_target','BucketSize',10);
        [index, dist] = knnsearch(kd_tree, data_source');
    else
        %利用欧式距离找出对应点集
        k=size(data_source,2);
        for i = 1:k
            data_q1(1,:) = data_target(1,:) - data_source(1,i);    % 两个点集中的点x坐标之差
            data_q1(2,:) = data_target(2,:) - data_source(2,i);    % 两个点集中的点y坐标之差
            data_q1(3,:) = data_target(3,:) - data_source(3,i);    % 两个点集中的点z坐标之差
            distance = sqrt(data_q1(1,:).^2 + data_q1(2,:).^2 + data_q1(3,:).^2);  % 欧氏距离
            [dist(i), index(i)] = min(distance);   % 找到距离最小的那个点
        end
    end
    
    disp(['误差err=',num2str(mean(dist))]);
    disp(['迭代次数ieration=',num2str(iteration)]);
    err_rec(iteration) = mean(dist);
    
    % 按距离排序，只取前面占比为inlierratio内的点以应对外点
    [~, idx] = sort(dist);
    inlier_num = round(size(data_source,2)*inlier_ratio);
    idx = idx(1:inlier_num);
    data_source_temp = data_source(:,idx);
    dist = dist(idx);
    index = index(idx);
    data_mid = data_target(:,index);
    
    % 去中心化后SVD分解求解旋转矩阵与平移向量
    [R_new, t_new] = rigidTransform3D(data_source_temp', data_mid');
    
    % 计算累计的旋转矩阵与平移向量
    Rf = R_new * Rf;
    Tf = R_new * Tf + t_new;
    
%     更新点集
%     data_source=R_new*data_source+t_new*ones(1,size(data_source,2));
    data_source=Rf*data_source_old+Tf*ones(1,size(data_source_old,2));
    
    % 显示中间结果
    if show == 1
        h = figure(2);
        scatter3(data_source(1,:),data_source(2,:),data_source(3,:),'b.');
        hold on;
        scatter3(data_target(1,:),data_target(2,:),data_target(3,:),'r.');
        hold off;
        daspect([1 1 1]);
        pause(0.1);
        drawnow
    end
    
    if err < Tolerance
        disp('————————————————————————————');
        disp('情况1：优化结果已经达到目标，结束优化');
        break
    end
    if iteration > 1 && err_rec(iteration-1) - err_rec(iteration) < step_Tolerance
        disp('————————————————————————————');
        disp('情况2：迭代每一步带来的优化到极限值，结束优化');
        break
    end
    if iteration>=max_iteration
        disp('————————————————————————————');
        disp('情况3：迭代已经达到最大次数，结束优化');
        break
    end
end
 
%% 计算最后结果的误差
if kd == 1
    %利用Kd-tree找出对应点集
    kd_tree = KDTreeSearcher(data_target','BucketSize',10);
    [index, dist] = knnsearch(kd_tree, data_source');
else
    %利用欧式距离找出对应点集
    k=size(data_source,2);
    for i = 1:k
        data_q1(1,:) = data_target(1,:) - data_source(1,i);    % 两个点集中的点x坐标之差
        data_q1(2,:) = data_target(2,:) - data_source(2,i);    % 两个点集中的点y坐标之差
        data_q1(3,:) = data_target(3,:) - data_source(3,i);    % 两个点集中的点z坐标之差
        distance = sqrt(data_q1(1,:).^2 + data_q1(2,:).^2 + data_q1(3,:).^2);  % 欧氏距离
        [dist(i), index(i)] = min(distance);   % 找到距离最小的那个点
    end
end
disp(['最终误差err=',num2str(mean(dist))]);
err_rec(iteration+1) = mean(dist);
 
%% 迭代优化过程中误差变化曲线
figure;
plot(0:iteration,err_rec);
grid on
 
% 最后点云匹配的结果
figure;
scatter3(data_source(1,:),data_source(2,:),data_source(3,:),'b.');
hold on;
scatter3(data_target(1,:),data_target(2,:),data_target(3,:),'r.');
hold off;
daspect([1 1 1]);
 
% disp('旋转矩阵的真值：');
% disp(T0);  %旋转矩阵真值
disp('计算出的旋转矩阵：');
T_final = [Rf,Tf];
T_final=[T_final;0,0,0,1];
disp(T_final);


%%保存变换矩阵
icp_tfm = T_final;
% save(sprintf('E:/ME/Study/dianyunpeizhun/Open3Dmanual/test10_29/%s/icp_tfm.mat',patientname),'icp_tfm');