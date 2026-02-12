close all; clear;
addpath('..');
%% input files
% x   =sprintf('%s/../../data/armadillo-x.txt',pwd);
% y   =sprintf('%s/../../data/armadillo-y.txt',pwd);
patientname = 'guozuyu';
x = sprintf('F:/cta-spect/CTA-SPECT/data_process/cta/final/%s/ijkcta.txt',patientname);
y = sprintf('E:/proj/peizhun/file_process/cloud_result0507/%s/cu.txt',patientname);
fnm =sprintf('%s/../../bcpd',                pwd);
fnw =sprintf('%s/../../win/bcpd.exe',        pwd);
if(ispc) bcpd=fnw; else bcpd=fnm; end;
%% parameters
omg ='0.0';
bet ='2.0';
lmd ='20.0';
gma ='10';
K   ='70';
J   ='300';
c   ='1e-6';
n   ='500';
f=  '0.3';
%% execution
prm1=sprintf('-w%s -b%s -l%s -g%s',omg,bet,lmd,gma);
prm2=sprintf('-J%s -p -f%s',J,f);
prm3=sprintf('-c%s -n%s -h -r1 ',c,n);
cmd =sprintf('%s -x%s -y%s -s T %s %s %s -sY',bcpd,x,y,prm1,prm2,prm3);
system(cmd); optpath3;

