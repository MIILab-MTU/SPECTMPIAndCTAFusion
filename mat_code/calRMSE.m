close all;clear all;clc;

%% 计算误差
% patientname = 'chenzongyu';
% load(sprintf('E:/ME/Study/配准数据处理/左心/CTA/final/%s/ijkcta.mat',patientname));
% load(sprintf('E:/ME/Study/dianyunpeizhun/Open3Dmanual/manualresult/%s/ICPresult.txt',patientname));
% load(sprintf('E:/ME/Study/dianyunpeizhun/result/%s/rigidresult.mat',patientname));
% % load(sprintf('E:/ME/Study/dianyunpeizhun/result/%s/sicpresult.mat',patientname));
% load(sprintf('E:/ME/Study/dianyunpeizhun/result/%s/wsicpresult.mat',patientname));
% load(sprintf('E:/ME/Study/dianyunpeizhun/result/%s/affineresult.mat',patientname));
% 
% X = ijkcta;
% Y1 = ICPresult;
% Y01 = delaunayn(Y1);
% [k1,dist1] = dsearchn(Y1,Y01,X);%在CTA中找SPECT
% Z1 = Y1(k1,:);
% dist21 = dot(dist1,dist1);
% RMSE1 = sqrt(dist21/(length(ijkcta)));
% juli1= sum(dist1)/length(ijkcta);
% 
% Y2 = rigidresult;
% Y02 = delaunayn(Y2);
% [k2,dist2] = dsearchn(Y2,Y02,X);%在CTA中找SPECT
% Z2 = Y2(k2,:);
% dist22 = dot(dist2,dist2);
% RMSE2 = sqrt(dist22/(length(ijkcta)));
% juli2 = sum(dist2)/length(ijkcta);
% 
% Y3 = wsicpresult;
% Y03 = delaunayn(Y3);
% [k3,dist3] = dsearchn(Y3,Y03,X);%在CTA中找SPECT
% Z3 = Y3(k3,:);
% dist23 = dot(dist3,dist3);
% RMSE3 = sqrt(dist23/(length(ijkcta)));
% juli3 = sum(dist3)/length(ijkcta);
% 
% Y4 = affineresult;
% Y04 = delaunayn(Y4);
% [k4,dist4] = dsearchn(Y4,Y04,X);%在CTA中找SPECT
% Z4 = Y4(k4,:);
% dist24 = dot(dist4,dist4);
% RMSE4 = sqrt(dist24/(length(ijkcta)));
% juli4 = sum(dist4)/length(ijkcta);


% load ijkcta.mat;
% % load('ICPresult.txt');
% % load CPDrigid.mat;
% % load SICPgai.mat;
% load CPDaffine.mat
% 
% 
% X = ijkcta;
% Y = result1;
% Y0 = delaunayn(Y);
% [k,dist] = dsearchn(Y,Y0,X);%在CTA中找SPECT
% Z = Y(k,:);
% dist2 = dot(dist,dist);
% RMSE = sqrt(dist2/(length(ijkcta)));
% juli= sum(dist)/length(ijkcta);


% X = result1;
% Y = ijkcta;
% Y0 = delaunayn(Y);
% [k,dist] = dsearchn(Y,Y0,X);%在SPECT中找CTA
% % Z = Y(k,:);
% % c =X-Z;
% dotcc = dot(dist,dist);
% % e = sum(dotcc);
% rmse = sqrt(dotcc/(length(X)));


%% 计算误差（新）10_29
data = readcell('在用71例病人数据.xlsx');
namelist=data(:,3);
nlnum = length(namelist);
% 输入病人名字
patientname1 = {''};

for i=1:nlnum
    if isequal(patientname1,namelist(i))
        index=i;
    end
end

temp = cell2mat(patientname1);
patientname = strrep(temp,' ','');

ps = cell2mat(data(index,12));

load(sprintf('E:/ME/Study/配准数据处理/左心/CTA/final/%s/ijkcta.mat',patientname));
load(sprintf('F:/result/%s/nicpresult.mat',patientname));
load(sprintf('F:/result/%s/nsicpresult.mat',patientname));
load(sprintf('F:/result/%s/nrigidresult.mat',patientname));
load(sprintf('F:/result/%s/naffineresult.mat',patientname));

X = ijkcta;
Y1 = nicpresult;
Y01 = delaunayn(Y1);
[k1,dist1] = dsearchn(Y1,Y01,X);%在CTA中找SPECT
Z1 = Y1(k1,:);
dist21 = dot(dist1,dist1);
RMSE1 = sqrt(dist21/(length(ijkcta)));
juli1= sum(dist1)/length(ijkcta);
jl1 = juli1*ps;

Y2 = nsicpresult;
Y02 = delaunayn(Y2);
[k2,dist2] = dsearchn(Y2,Y02,X);%在CTA中找SPECT
Z2 = Y2(k2,:);
dist22 = dot(dist2,dist2);
RMSE2 = sqrt(dist22/(length(ijkcta)));
juli2 = sum(dist2)/length(ijkcta);
jl2 = juli2*ps;

Y3 = nrigidresult;
Y03 = delaunayn(Y3);
[k3,dist3] = dsearchn(Y3,Y03,X);%在CTA中找SPECT
Z3 = Y3(k3,:);
dist23 = dot(dist3,dist3);
RMSE3 = sqrt(dist23/(length(ijkcta)));
juli3 = sum(dist3)/length(ijkcta);
jl3 = juli3*ps;

Y4 = naffineresult;
Y04 = delaunayn(Y4);
[k4,dist4] = dsearchn(Y4,Y04,X);%在CTA中找SPECT
Z4 = Y4(k4,:);
dist24 = dot(dist4,dist4);
RMSE4 = sqrt(dist24/(length(ijkcta)));
juli4 = sum(dist4)/length(ijkcta);
jl4 = juli4*ps;


%% 计算平均值和标准差
% a = [2.9594
% 3.7274
% 3.6334
% 3.7298
% 4.4882
% 4.1492
% 4.3352
% 4.2159
% 3.9218
% 3.3623
% 4.6544
% 3.2378
% 3.8223
% 3.8177
% 4.1465];
% m_a = mean(a);
% std_a = std(a);
% b = [3.3228
% 3.4875
% 3.7394
% 4.1326
% 3.0687
% 3.5751
% 4.5436
% 3.4304
% 2.9688
% 3.8372
% 4.3350
% 4.1685
% 3.6649
% 2.5433
% 3.3729];
% m_b = mean(b);
% std_b = std(b);
% c = [2.8526
% 3.1015
% 3.2400
% 3.5601
% 2.7859
% 3.2983
% 3.1731
% 3.4097
% 3.1871
% 2.9449
% 3.4312
% 2.7403
% 3.3837
% 2.6789
% 3.2114];
% m_c = mean(c);
% std_c = std(c);
% d = [2.1601
% 2.3486
% 2.0903
% 3.0658
% 2.3374
% 2.6539
% 2.3025
% 2.0084
% 2.2184
% 2.3317
% 3.1509
% 2.5253
% 2.3559
% 1.8405
% 2.0775];
% m_d = mean(d);
% std_d = std(d);















