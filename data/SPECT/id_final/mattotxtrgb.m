clc;close all;clear all;

% load('ijkcta.mat');
% save('ijkcta.txt','ijkcta','-ASCII');
% 
% load('ijkspect1add.mat');
% save('ijkspect1add.txt','ijkspect1add','-ASCII');

% load('ijkspect1add.mat');
% ijkspectbig=ijkspect1add*16.5874;
% save('ijkspectbig.txt','ijkspectbig','-ASCII');

% load('E:/ME/Study/配准数据处理/左心/CTA/final/changsuping/ijkcta.mat');
% rgb1=[];
% for i=1:5086
%     rgb1(i,:)=[0,0,255];
% end
% for j=5087:length(ijkcta)
%     rgb1(j,:)=[255,0,0];
% end
% ijkctargb=[ijkcta,rgb1];
% save('E:/ME/Study/配准数据处理/左心/CTA/final/changsuping/ijkctargb.txt','ijkctargb','-ASCII');

ipatient = 8;
load(sprintf('E:/ME/Study/配准数据处理/左心/SPECT/id_final/Patient%02d/ijkspect.mat',ipatient));
rgb2=[];
for i=1:5200
    rgb2(i,:)=[0,0,255];
end
for j=5201:length(ijkspect)
    rgb2(j,:)=[255,0,0];
end
ijksprgb=[ijkspect,rgb2];
save(sprintf('E:/ME/Study/配准数据处理/左心/SPECT/id_final/Patient%02d/ijksprgb.txt',ipatient),'ijksprgb','-ASCII');


% load('spect_result.txt');
% save('spect_result.mat','spect_result');

% load('spect_result.mat');
% load('ijkcta.mat');
% plot3(spect_result(:,1),spect_result(:,2),spect_result(:,3),'b.');hold on;
% plot3(ijkcta(:,1),ijkcta(:,2),ijkcta(:,3),'r.');hold on;

% load('ijkspect1add.mat');
% rgb3=[];
% for i=1:5200
%     rgb3(i,:)=[0,0,255];
% end
% for j=5201:length(ijkspect1add)
%     rgb3(j,:)=[255,0,0];
% end
% ijkspect1addrgb=[ijkspect1add,rgb3];
% save('ijkspect1addrgb.txt','ijkspect1addrgb','-ASCII');

%% 
% load('spect_result4_14.txt');
% % save('spect_result4_14.mat','spect_result4_14');
% min(spect_result4_14)
















