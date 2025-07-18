function [] = IJK_python(i_patient, i_gate)
%% lps点云转换到ijk，并加入标定的特殊点

%% ijkcta有特殊点，ijkcta1无特殊点

data = readcell('在用71例病人数据.xlsx');
namelist=data(:,3);
nlnum = length(namelist);
% 输入病人名字
patientname = {'zhang xi liang'};

for i=1:nlnum
    if isequal(patientname,namelist(i))
        index=i;
    end
end

temp = cell2mat(patientname);
patientname1 = strrep(temp,' ','');
ctapoints = load(sprintf('./downsampling/%s/newLV.txt',patientname1));
numcta = length(ctapoints);
lps=[ctapoints,ones(length(ctapoints),1)];

l0 = cell2mat(data(index,9));
p0 = cell2mat(data(index,10));
s0 = cell2mat(data(index,11));
ps = cell2mat(data(index,12));
as = cell2mat(data(index,13));

transf=[ps,0,0,l0;
        0,ps,0,p0;
        0,0,as,s0;
        0,0,0,1];

ijk=inv(transf)*lps';
ijk=ijk(1:3,:)';
figure
plot3(ijk(:,1),ijk(:,2),ijk(:,3),'b.');hold on;

%% 保存ijkcta1，其中没有特殊点
% ijkcta1 = ijk;
% dirpath = sprintf('./final/%s',patientname1);
% if exist(dirpath)==0
%     mkdir(dirpath);
% else
%     disp('文件夹已存在');
% end
% save(sprintf('./final/%s/ijkcta1.mat',patientname1),'ijkcta1');
% save(sprintf('./final/%s/ijkcta1.txt',patientname1),'ijkcta1','-ASCII');


%% 保存ijkcta，有特殊点
% load(sprintf('./name_selectedpoints/%s/SelectedPoints.mat',patientname1));
% plot3(Positions_SelectedPoints(1,:),Positions_SelectedPoints(2,:),Positions_SelectedPoints(3,:),'r*');hold on;
ijkcta = ijk;

% load(sprintf('./name_selectedpoints/%s/SelectedPoints1.mat',patientname1));
% plot3(Positions_SelectedPoints1(1,:),Positions_SelectedPoints1(2,:),Positions_SelectedPoints1(3,:),'g*');hold on;
% ijkcta = [ijkcta;Positions_SelectedPoints1'];

figure
plot3(ijkcta(1:numcta,1),ijkcta(1:numcta,2),ijkcta(1:numcta,3),'b.');hold on;
% plot3(ijkcta(numcta+1:end,1),ijkcta(numcta+1:end,2),ijkcta(numcta+1:end,3),'r*');hold on;

save(sprintf('./final/%s/ijkcta.mat',patientname1),'ijkcta');
save(sprintf('./final/%s/ijkcta.txt',patientname1),'ijkcta','-ASCII');











