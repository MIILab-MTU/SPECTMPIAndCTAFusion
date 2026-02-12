clc; close all; clear;
addpath('..');

% DIRDATA =sprintf('%s/../../data',pwd);
% BASENAME=sprintf('chef_view');

fnm =sprintf('%s/../../bcpd',         pwd);
fnw =sprintf('%s/../../win/bcpd.exe', pwd);
if(ispc) bcpd=fnw; else bcpd=fnm; end;

%% parameters
omg ='0.2';
gma ='1';
J   ='300';
f   ='0.3';
c   ='1e-7';
n   ='500';
bet ='2.0';
lmd ='20.0';

%% paring
ntarget=22;
src=[0:1:15 3 8 1 8 4 2]; % target: j, source: src(j)

%% registration

%% input files
% y   =sprintf('%s/../../data/chef_view.txt',pwd);
% x   =sprintf('%s/../../data/chef_view.txt',pwd);
patientname = 'guozuyu';
x = sprintf('F:/cta-spect/CTA-SPECT/data_process/cta/final/%s/ijkcta.txt',patientname);
y = sprintf('E:/proj/peizhun/file_process/cloud_result0507/%s/cu.txt',patientname);


%% execution
prm1=sprintf('-w%s -b%s -l%s -g%s',omg,bet,lmd,gma);
prm2=sprintf('-J%s -p -f%s',J,f);
prm3=sprintf('-c%s -n%s -h -r1 ',c,n);
cmd =sprintf('%s -x%s -y%s %s %s %s -Tr -sT,v',bcpd,x,y,prm1,prm2,prm3);
system(cmd);


s = load('output_s.txt');
R = load('output_R.txt');
t = load('output_t.txt');



X0 = load(x)
T0 = load(y)

Y_transformed = s * (T0) * R' + t';

f2 = figure('Name', 'Before/After Registration', 'NumberTitle', 'off');
subplot(1, 2, 1);
plot3(T0(:,1), T0(:,2), T0(:,3), '.r', 'MarkerSize', 3); hold on;
plot3(X0(:,1), X0(:,2), X0(:,3), '.b', 'MarkerSize', 3); daspect([1 1 1]); grid on;
title('Before Registration', 'FontSize', 18);
subplot(1, 2, 2);
plot3(Y_transformed(:,1), Y_transformed(:,2), Y_transformed(:,3), '.r', 'MarkerSize', 3); hold on;
plot3(X0(:,1), X0(:,2), X0(:,3), '.b', 'MarkerSize', 3); daspect([1 1 1]); grid on;
title('After Registration (Transformed)', 'FontSize', 18);
%ntarget=15;

% %% storage
% D=3;
% s=zeros(ntarget,1);
% R=zeros(D,D,ntarget); 
% t=zeros(D,ntarget); 
% 
% 
% %% load rigid transformations
% for j=2:ntarget
%   fns =sprintf('Result/output_s%.3d.txt',j);
%   fnR =sprintf('Result/output_R%.3d.txt',j);
%   fnt =sprintf('Result/output_t%.3d.txt',j);
% 
%   s(j)     =load(fns);
%   R(:,:,j) =load(fnR);
%   t(:,j)   =load(fnt);
% end
% 
% %% apply rigid transformations
% X=load(sprintf('%s/%s001.txt',DIRDATA,BASENAME));
% N=size(X,1);
% 
% basecolor=21; red=1;green=2;blue=3;
% C=basecolor*ones(N,1);
% 
% for j=2:ntarget
%   fnt=sprintf('%s/%s%.3d.txt',DIRDATA,BASENAME,j);
%   x=load(fnt);
%   N=size(x,1);
%   
%   if ntarget-j == 1         % source
%     C=[C;ones(N,1)*blue]; 
%   elseif ntarget-j == 0     % target
%     C=[C;ones(N,1)*red];
%   else
%     C=[C;ones(N,1)*basecolor];
%   end
%   
%   while true
%     if src(j)==0; break; end
%     x=(x-ones(N,1)*t(:,j)')*R(:,:,j)/s(j);
%     j=src(j);
%   end
% 
%   X=[X;x];
% end
% 
% writematrix(X,'Result/recovChef.txt',     'Delimiter','tab');
% writematrix(C,'Result/recovChefColor.txt','Delimiter','tab');
