function [data_target,s,R,T, e]=fSICP2D(data_source,data_target)
% [s,R,T,e,it] = reg3D(data_source',data_target');
[s,R,T,e,it] = reg2D(data_source',data_target')

% plot_3d_3(data_source, data_target); % 显示出当前两个点集
for i=1:3
    data_target(:,i) = data_target(:,i)-T(i);
    data_target(:,i) = data_target(:,i)/s(i);
end
data_target = data_target*R;
plot_3d_3(data_source, data_target); % 显示出当前两个点集



