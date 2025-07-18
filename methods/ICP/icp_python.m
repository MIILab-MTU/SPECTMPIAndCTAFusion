function [final_dianyun, err_final] = icp_python(input_dir,cloud_dir, name,isori)

kd = 1;
inlier_ratio = 0.999;
Tolerance = 0.05;
step_Tolerance = 0.01;
max_iteration =100;
show = 1;
 


patientname = name;
load(sprintf('%s/%s/ijkcta.txt',input_dir,patientname));
if isori
    load(sprintf('F:/cta-spect/CTA-SPECT/result/%s/cu.txt',patientname));
else
    load(sprintf('%s/%s/cu.txt',cloud_dir,patientname));
end
%
data_source = cu';
data_target = ijkcta';


figure;
scatter3(data_source(1,:),data_source(2,:),data_source(3,:),'b.');
hold on;
scatter3(data_target(1,:),data_target(2,:),data_target(3,:),'r.');
hold off;
daspect([1 1 1]);
 

T_final=eye(4,4);
iteration=0;
Rf=T_final(1:3,1:3);
Tf=T_final(1:3,4);
data_source=Rf*data_source+Tf*ones(1,size(data_source,2));
err=1;
data_source_old = data_source;

while(1)
    iteration=iteration+1;
    if kd == 1

        kd_tree = KDTreeSearcher(data_target','BucketSize',10);
        [index, dist] = knnsearch(kd_tree, data_source');
    else

        k=size(data_source,2);
        for i = 1:k
            data_q1(1,:) = data_target(1,:) - data_source(1,i);
            data_q1(2,:) = data_target(2,:) - data_source(2,i);
            data_q1(3,:) = data_target(3,:) - data_source(3,i);
            distance = sqrt(data_q1(1,:).^2 + data_q1(2,:).^2 + data_q1(3,:).^2);
            [dist(i), index(i)] = min(distance);
        end
    end
    
    disp(['err=',num2str(mean(dist))]);
    disp(['ieration=',num2str(iteration)]);
    err_rec(iteration) = mean(dist);
    

    [~, idx] = sort(dist);
    inlier_num = round(size(data_source,2)*inlier_ratio);
    idx = idx(1:inlier_num);
    data_source_temp = data_source(:,idx);
    dist = dist(idx);
    index = index(idx);
    data_mid = data_target(:,index);
    

    [R_new, t_new] = rigidTransform3D(data_source_temp', data_mid');
    

    Rf = R_new * Rf;
    Tf = R_new * Tf + t_new;
    

    data_source=Rf*data_source_old+Tf*ones(1,size(data_source_old,2));
    

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

        disp('1');
        break
    end
    if iteration > 1 && err_rec(iteration-1) - err_rec(iteration) < step_Tolerance

        disp('2');
        break
    end
    if iteration>=max_iteration

        disp('3');
        break
    end
end

if kd == 1
    kd_tree = KDTreeSearcher(data_target','BucketSize',10);
    [index, dist] = knnsearch(kd_tree, data_source');
else

    k=size(data_source,2);
    for i = 1:k
        data_q1(1,:) = data_target(1,:) - data_source(1,i);
        data_q1(2,:) = data_target(2,:) - data_source(2,i);
        data_q1(3,:) = data_target(3,:) - data_source(3,i);
        distance = sqrt(data_q1(1,:).^2 + data_q1(2,:).^2 + data_q1(3,:).^2);
        [dist(i), index(i)] = min(distance);
    end
end
disp(['Final err=',num2str(mean(dist))]);
err_final = num2str(mean(dist))
err_rec(iteration+1) = mean(dist);
 

figure;
plot(0:iteration,err_rec);
grid on
 

figure;
scatter3(data_source(1,:),data_source(2,:),data_source(3,:),'b.');
hold on;
scatter3(data_target(1,:),data_target(2,:),data_target(3,:),'r.');
hold off;
daspect([1 1 1]);
 

disp('ת');
T_final = [Rf,Tf];
T_final=[T_final;0,0,0,1];
disp(T_final);



icp_tfm = T_final;
final_dianyun=data_source;

end

