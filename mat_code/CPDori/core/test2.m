clear;
close all;
clc;
time=cputime;
mbSize = 16;
p = 7;
%各位换成自己找的图！！！
%记得路径也要换！！！
A=imread('hb.png');
B=imread('hb1.png');
f2=rgb2gray(A);
f1=rgb2gray(B);
figure,imshow(A),title('当前帧');
figure,imshow(B),title('参考帧');
[r,c]=size(A);
%绘制运动矢量图
[motionVect, TSScomputations] = motionEstTSS(B, A, mbSize, p);
a = zeros(1,r*c/mbSize^2);
a(:) = motionVect(1,1:r*c/mbSize^2);
b = zeros(1,r*c/mbSize^2);
b(:) = motionVect(2,1:r*c/mbSize^2);
for i = 1:r/mbSize
    for j = 1:c/mbSize
        mvx(i,j)=b(1,j+(i-1)*(c/mbSize));%运动矢量的X坐标
        mvy(i,j)=-a(1,j+(i-1)*(c/mbSize));%运动矢量的Y坐标
    end
end
figure;quiver(flipud(mvx),flipud(mvy));%flipud函数上下翻转矩阵
title('运动矢量图');
set(gca,'XLim',[-1,c/mbSize+2],'YLim',[-1,r/mbSize+2]);
%绘制预测帧
imgComp = motionComp(A,motionVect,mbSize);
I1 = uint8(imgComp);
figure;imshow(I1);
title('预测帧');
%绘制残差图
canc = zeros(r,c);
for i = 1:r
    for j = 1:c
        canc(i,j) = 255 - abs(B(i,j)-imgComp(i,j));
    end 
end
PSNR=20*log10(255)+20*log10(256)-10*log10(sum(sum(canc.*canc)));%峰值信噪比
time=cputime-time
