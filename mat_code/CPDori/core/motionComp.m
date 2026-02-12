% Computes motion compensated image using the givenmotion vectors（用给定的运动矢量计算运动补偿图像）

%

% Input

%?? imgI : Thereference image （参考图像）

%?? motionVect :The motion vectors（运动矢量）

%?? mbSize :Size of the macroblock（宏块大小）

%

% Ouput

%?? imgComp :The motion compensated image（运动补偿图像）

%

% Written by Aroh Barjatya

function imgComp = motionComp(imgI, motionVect, mbSize)
[row col] = size(imgI);

% we start off from the top left of the image（从图像左上角开始）

% we will walk in steps of mbSize（以宏块大小为步长）

% for every marcoblock that we look at we will readthe motion vector（对于看到的每一个宏块，读出它的运动矢量）

% and put that macroblock from refernce image in thecompensated image（并将参考图像中的该宏块放到补偿图像中）


mbCount = 1;

for i = 1:mbSize:row-mbSize+1
    for j =1:mbSize:col-mbSize+1
% dy isrow(vertical) index（dy为垂直方向上的指数）
% dx iscol(horizontal) index（dx为水平方向上的指数）
% thismeans we are scanning in order
        dy =motionVect(1,mbCount);
        dx =motionVect(2,mbCount);
        refBlkVer = i + dy;
        refBlkHor = j + dx;
        if (refBlkVer < 1 || refBlkVer+mbSize-1 > row || refBlkHor < 1 || refBlkHor+mbSize-1 > col)
            imageComp(i:i+mbSize-1,j:j+mbSize-1)=imgI(i:i+mbSize-1,j:j+mbSize-1);
            continue;
        end
        imageComp(i:i+mbSize-1,j:j+mbSize-1)= imgI(refBlkVer:refBlkVer+mbSize-1, refBlkHor:refBlkHor+mbSize-1);
        mbCount= mbCount + 1;
    end
end


imgComp = imageComp;

