close all; clear; clc;
addpath('..');

%% input files
% x   =sprintf('%s/../../data/face-g-x.txt', pwd);
% y   =sprintf('%s/../../data/face-g-y.txt', pwd);
patientname = 'caochuande';
x = sprintf('E:/proj/peizhun/file_process/name/%s/ijkcta.txt',patientname);
y = sprintf('E:/proj/peizhun/file_process/cloud_result0507/%s/cu.txt',patientname);
fnm =sprintf('%s/../../bcpd',              pwd);
fnw =sprintf('%s/../../win/bcpd.exe',      pwd);
% fnf =sprintf('%s/../../data/face-g-triangles.txt',pwd);
fnf ='E:/proj/peizhun/file_process/name/caochuande/ijkcta-g-triangles.txt';
if(ispc) bcpd=fnw; else bcpd=fnm; end;
%% parameters
omg ='0.0';
bet ='1.3';
lmd ='100';
gma ='3';
K   ='300';
J   ='300';
c   ='1e-6';
n   ='500';
nrm ='x';
dwn ='B,3000,0.02';
tau ='.5';
%% execution
kern='geodesic,0.2,8,1';
prm1=sprintf('-w%s -b%s -l%s -g%s',omg,bet,lmd,gma);
prm2=sprintf('-J%s -K%s -p -u%s -D%s',J,K,nrm,dwn);
prm3=sprintf('-c%s -n%s -h -r1',c,n);
cmd =sprintf('%s -x%s -y%s %s %s %s  -G%s',bcpd,x,y,prm1,prm2,prm3,kern);
system(cmd);

X0=load(x);
T0=load(y);
T1=load('output_y.txt');
f =load(fnf); if min(min(f))==0; f=f+1; end;

tag=0;
optpathMeshPP;

