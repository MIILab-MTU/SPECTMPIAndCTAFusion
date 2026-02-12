clc;clear all;close all;

id = 72;
load(sprintf('E:/ME/Study/配准数据处理/左心/SPECT/id_final/Patient%02d/ijkspect.mat',id));
temp = array2table(ijkspect);
temp.Properties.VariableNames(1:3) = {'x', 'y', 'z'};
writetable(temp, sprintf('./Patient%02d/zhangxiliang.csv',id));
writetable(temp, '../final_csv/zhangxiliang.csv');


