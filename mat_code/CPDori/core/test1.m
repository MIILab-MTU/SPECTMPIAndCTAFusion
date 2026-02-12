clear;
close all;
clc;
time=cputime;
%穷尽块匹配算法
%各位换成自己找的图，图的大小要是256*256的！！！
A=imread('hb.png');
B=imread('hb1.png');
f2=rgb2gray(A);
f1=rgb2gray(B);
fp=0;

figure,imshow(f2),title('目标帧');
figure,imshow(f1),title('锚定帧');

N=16;R=16;
height=256;
width=256;

for i=1:N:height-N+1
   for j=1:N:width-N+1
       MAD_min=256*N*N;
       dy=0;dx=0;
       for k=-R:1:R
           for l=-R:1:R
               if i+k<1
                   MAD=256*N*N;
               elseif i+k>height-N
                   MAD=256*N*N;
               elseif j+l<1
                   MAD=256*N*N;
               elseif j+l>width-N
                   MAD=256*N*N;
               else
                   MAD=sum(sum(abs(double(f1(i:i+N-1,j:j+N-1))-double(f2(i+k:i+k+N-1,j+l:j+l+N-1)))));
               end
               if MAD<MAD_min
                   MAD_min=MAD;
                   dy=k;dx=l;
               end;end;end;
       fp(i:i+N-1,j:j+N-1)=f2(i+dy:i+dy+N-1,j+dx:j+dx+N-1);
       
       iblk=floor((i-1)/N+1);
       jblk=floor((j-1)/N+1);
       mvx(iblk,jblk)=dx;
       mvy(iblk,jblk)=dy;
   end;end;
figure,imshow(uint8(fp)),title('预测帧');

[X,Y]=meshgrid(N/2:N:256-N/2);
Y=256-Y;
figure,quiver(X,Y,mvx,mvy),title('整像素精度的运动场');

diff=abs(double(f1)-fp);
figure,imshow(uint8(diff)),title('diff');%残差帧图像

PSNR=20*log10(255)+20*log10(256)-10*log10(sum(sum(diff.*diff)));%峰值信噪比
   time=cputime-time       