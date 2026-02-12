clear all;
close all;
clc;

%% 本文件用于处理CTA和SPECT特殊点，实现特殊点由前沟到后沟顺序存储

% load ijkcta;
% yuan = ijkcta(1:5111,:);
% te = ijkcta(5112:5:end,:);
% ctc = mean(yuan);
% % save('ctate.txt','te','-ASCII');
% [k,dist]=dsearchn(yuan,te);
% xinte = yuan(k,:);
% figure
% plot3(yuan(:,1),yuan(:,2),yuan(:,3),'b.');hold on;
% plot3(te(:,1),te(:,2),te(:,3),'r*');hold on;
% plot3(ctc(:,1),ctc(:,2),ctc(:,3),'bo');hold on;
% 
% load('testy1.txt');
% spyuan = testy1(1:5200,1:3);
% spc = mean(spyuan);
% spteqian = testy1(5213:2:end,1:3);
% sptehou = testy1(5214:2:end,1:3);
% spte = [];
% for i=1:6
%     spte(i,1:3) = spteqian(i,1:3); 
% end
% for i=1:5
%     spte(6+i,1:3) = sptehou(i,1:3);
% end
% % save('spte.txt','spte','-ASCII');
% [k1,dist1]=dsearchn(spyuan,spte);
% nspte = [];
% for i = 1:11
%     nspte(i,:) = spyuan(k1(i),:);
% end
% % save('nspte.txt','nspte','-ASCII');
% figure
% plot3(spyuan(:,1),spyuan(:,2),spyuan(:,3),'b.');hold on;
% plot3(nspte(1:6,1),nspte(1:6,2),nspte(1:6,3),'k*');hold on;
% plot3(nspte(7:end,1),nspte(7:end,2),nspte(7:end,3),'r*');hold on;
% plot3(spc(:,1),spc(:,2),spc(:,3),'bo');hold on;
% plot3(spte(1:6,1),spte(1:6,2),spte(1:6,3),'k*');hold on;
% plot3(spte(7:end,1),spte(7:end,2),spte(7:end,3),'r*');hold on;


% load('E:/ME/Study/配准数据处理/左心/CTA/name_selectedpoints/changsuping/SelectedPoints.mat');
% load('E:/ME/Study/配准数据处理/左心/CTA/final/changsuping/ijkcta.mat');
% load('E:/ME/Study/配准数据处理/左心/SPECT/id_final/Patient03/ijksprgb.txt');
% load('E:/ME/Study/配准数据处理/左心/SPECT/id_selectedpoints/Patient03/SelectedPoints.mat');
% yuan = ijkcta(1:5086,:);
% te = ijkcta(5087:5:end,:);
% save('ctate1.txt','te','-ASCII');
% figure
% plot3(yuan(:,1),yuan(:,2),yuan(:,3),'b.');hold on;
% plot3(te(:,1),te(:,2),te(:,3),'r*');hold on;
% spyuan = ijksprgb(1:5200,1:3);
% spte = Positions_SelectedPoints';
% spqian = spte(1:2:end,:);
% sphou = spte(2:2:end,:);
% for i = 1:length(spqian)
%     spte(i,:) = spqian(i,:);
% end
% for i = 1:length(sphou)
%     spte(length(spqian)+i,:) = sphou(i,:);
% end
% spte = spte(1:13,:);
% figure
% plot3(spyuan(:,1),spyuan(:,2),spyuan(:,3),'b.');hold on;
% % plot3(spqian(:,1),spqian(:,2),spqian(:,3),'k*');hold on;
% % plot3(sphou(:,1),sphou(:,2),sphou(:,3),'r*');hold on;
% % plot3(spte(:,1),spte(:,2),spte(:,3),'r*');hold on;
% [k1,dist1]=dsearchn(spyuan,spte);
% nspte = [];
% for i = 1:13
%     nspte(i,:) = spyuan(k1(i),:);
% end
% plot3(nspte(:,1),nspte(:,2),nspte(:,3),'k*');hold on;
% save('nspte1.txt','nspte','-ASCII');



%% 从这里开始10_29
patientname = 'zhangxiliang';
patientid = 72;

load(sprintf(strcat('E:/ME/Study/配准数据处理/左心/CTA/final/',patientname,'/ijkcta.mat')));
dirlist = dir(sprintf('E:/ME/Study/配准数据处理/左心/CTA/name_selectedpoints/%s',patientname));
filenum = length(dirlist)-2;
if filenum == 1   
    load(sprintf(strcat('E:/ME/Study/配准数据处理/左心/CTA/name_selectedpoints/',patientname,'/SelectedPoints.mat')));
    ctte = Positions_SelectedPoints';
    numctte = length(ctte);
else
    load(sprintf(strcat('E:/ME/Study/配准数据处理/左心/CTA/name_selectedpoints/',patientname,'/SelectedPoints.mat')));
    load(sprintf(strcat('E:/ME/Study/配准数据处理/左心/CTA/name_selectedpoints/',patientname,'/SelectedPoints1.mat')));
    ctte = Positions_SelectedPoints';
    ctte1 = Positions_SelectedPoints1';
%     ctte1 = flipud(ctte1);
    ctte = [ctte;ctte1];
    numctte = length(ctte);
end

load(sprintf('E:/ME/Study/配准数据处理/左心/SPECT/id_final/Patient%02d/ijkspect.mat',patientid));
load(sprintf('E:/ME/Study/配准数据处理/左心/SPECT/id_selectedpoints/Patient%02d/SelectedPoints.mat',patientid));
spete = Positions_SelectedPoints';
numspte = length(spete);

dirpath = sprintf(strcat('./',patientname));
if ~exist(dirpath,'dir')
    mkdir(dirpath);
end

multi = floor(numctte/numspte);
disp(multi);

% ijkcta中添加了特殊点，减去特殊点点数，是原始cta点数。
ctyuanl = length(ijkcta)-length(ctte);
figure;
plot3(ijkcta(1:ctyuanl,1),ijkcta(1:ctyuanl,2),ijkcta(1:ctyuanl,3),'b.');hold on;
plot3(ctte(:,1),ctte(:,2),ctte(:,3),'r*');hold on;
 
ctate = ctte(1:multi:end,:);
% 第一种 从大到小
% ctate = ctate(length(ctate)-numspte+1:end,:);
% 第二种 从小到大
% % ctate = ctate(1:end-(length(ctate)-numspte),:);
% % ctate = flipud(ctate); %矩阵上下翻转，实现特殊点由前沟到后沟顺序排列
% % 单独处理
% ctate = ctate(1:2:end,:);
% ctate = ctate(1:end-(length(ctate)-13),:);
% ctate = ctate(length(ctate)-13+1:end,:);
% ctate = flipud(ctate);
% ctate = ctate(1:(end-1),:);

numctate = length(ctate);

figure;
plot3(ijkcta(1:ctyuanl,1),ijkcta(1:ctyuanl,2),ijkcta(1:ctyuanl,3),'b.');hold on;
plot3(ctate(:,1),ctate(:,2),ctate(:,3),'k*');hold on;
plot3(ctate(1,1),ctate(1,2),ctate(1,3),'r*');hold on;
plot3(ctate(end,1),ctate(end,2),ctate(end,3),'g*');hold on;


spyuan = ijkspect(1:5200,:);
spqian = spete(1:2:end,:);
sphou = spete(2:2:end,:);
figure;
plot3(spyuan(:,1),spyuan(:,2),spyuan(:,3),'b.');hold on;
plot3(spqian(:,1),spqian(:,2),spqian(:,3),'r*');hold on;
plot3(sphou(:,1),sphou(:,2),sphou(:,3),'g*');hold on;

for i = 1:length(spqian)
    spte(i,:) = spqian(i,:);
end
sphou = flipud(sphou);%调整后沟特殊点顺序为从心尖到基底
for i = 1:length(sphou)
    spte(length(spqian)+i,:) = sphou(i,:);
end
% plot3(spte(1:6,1),spte(1:6,2),spte(1:6,3),'g*');hold on;
% [k1,dist1]=dsearchn(spyuan,spte);%找出距离最近的外表面上的点
% nspte = [];
% for i = 1:length(spte)
%     nspte(i,:) = spyuan(k1(i),:);
% end
% 一般情况
nspte = spte(:,:);
% 单独处理
% nspte = spte(2:end-1,:);
figure;
plot3(spyuan(:,1),spyuan(:,2),spyuan(:,3),'b.');hold on;
plot3(nspte(:,1),nspte(:,2),nspte(:,3),'k*');hold on;
plot3(nspte(1,1),nspte(1,2),nspte(1,3),'r*');hold on;
plot3(nspte(end,1),nspte(end,2),nspte(end,3),'g*');hold on;

% save(sprintf('%s/ctate.txt',dirpath),'ctate','-ASCII');
% save(sprintf('%s/nsptexin.txt',dirpath),'nspte','-ASCII');




%% 查看处理后效果
% patientname = 'guanyang';
% patientid = 20;
% load(sprintf('E:/ME/Study/配准数据处理/左心/tedata/%s/ctate.txt',patientname));
% load(sprintf('E:/ME/Study/配准数据处理/左心/tedata/%s/nsptexin.txt',patientname));
% load(sprintf(strcat('E:/ME/Study/配准数据处理/左心/CTA/final/',patientname,'/ijkcta.mat')));
% load(sprintf('E:/ME/Study/配准数据处理/左心/SPECT/id_final/Patient%02d/ijkspect.mat',patientid));
% 
% figure;
% plot3(ijkcta(:,1),ijkcta(:,2),ijkcta(:,3),'b.');hold on;
% plot3(ijkspect(:,1),ijkspect(:,2),ijkspect(:,3),'g.');hold on;
% 
% plot3(ctate(end,1),ctate(end,2),ctate(end,3),'ro');hold on;
% plot3(nsptexin(end,1),nsptexin(end,2),nsptexin(end,3),'ko');hold on;





% load('./chenzongyu/ctate.txt');
% nspte = load('./chenzongyu/nspte放大过的.txt');
% % a = nspte(7:11,:);
% % a = flipud(a);
% % nspte(7:11,:) = a;
% % figure;
% % plot3(ctate(:,1),ctate(:,2),ctate(:,3),'r*');hold on;
% % plot3(ctate(1,1),ctate(1,2),ctate(1,3),'b*');hold on;
% % plot3(ctate(end,1),ctate(end,2),ctate(end,3),'b*');hold on;
% figure;
% plot3(nspte(:,1),nspte(:,2),nspte(:,3),'r*');hold on;
% plot3(nspte(1,1),nspte(1,2),nspte(1,3),'b*');hold on;
% plot3(nspte(end,1),nspte(end,2),nspte(end,3),'b*');hold on;
% % save('./chenzongyu/nspte放大过的.txt','nspte','-ASCII');


% load('./changsuping/ctate.txt');
% nspte = load('./changsuping/nspte.txt');
% temp = nspte(9:end,:);
% temp = flipud(temp);
% nspte(9:end,:) = temp;
% figure;
% plot3(ctate(:,1),ctate(:,2),ctate(:,3),'r*');hold on;
% plot3(ctate(1,1),ctate(1,2),ctate(1,3),'b*');hold on;
% plot3(ctate(end,1),ctate(end,2),ctate(end,3),'b*');hold on;
% figure;
% plot3(nspte(:,1),nspte(:,2),nspte(:,3),'r*');hold on;
% plot3(nspte(1,1),nspte(1,2),nspte(1,3),'b*');hold on;
% plot3(nspte(end,1),nspte(end,2),nspte(end,3),'b*');hold on;
% % save('./changsuping/nsptexin.txt','nspte','-ASCII');



