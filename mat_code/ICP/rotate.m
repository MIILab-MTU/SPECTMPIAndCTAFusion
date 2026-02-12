%% 对点云进行旋转与平移
function [data_q,T] = rotate2(data,theta_x, theta_y, theta_z, t)
theta_x = theta_x/180*pi;
rot_x = [1 0 0;0 cos(theta_x) sin(theta_x);0 -sin(theta_x) cos(theta_x)];
theta_y = theta_y/180*pi;
rot_y = [cos(theta_y) 0 -sin(theta_y);0 1 0;sin(theta_y) 0 cos(theta_y)];
theta_z = theta_z/180*pi;
rot_z = [cos(theta_z) sin(theta_z) 0;-sin(theta_z) cos(theta_z) 0;0 0 1];
% 变换矩阵
T = rot_x*rot_y*rot_z;
T = [T,t'];
T=[T;0,0,0,1];
%化为齐次坐标
rows=size(data,2);
rows_one=ones(1,rows);
data=[data;rows_one];
%返回三维坐标
data_q=T*data;
data_q=data_q(1:3,:);
end

