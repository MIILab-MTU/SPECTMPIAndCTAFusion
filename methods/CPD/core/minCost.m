% Finds the indices of the cell that holds the minimumcost（找到拥有最小绝对差的点的指数）

%

% Input

%?? costs : Thematrix that contains the estimation costs for a

%?? macroblock（包含宏块的估计代价的矩阵）

%

% Output

%?? dx : the motionvector component in columns（列方向上运动矢量组成）

%?? dy : themotion vector component in rows（行方向上运动矢量组成）

%

% Written by Aroh Barjatya

function [dx, dy, min] = minCost(costs)

[row, col] = size(costs);

% we check whether the current value of costs is lessthen the already

% present value in min.

% If its inded smaller then we swap the min value withthe current one and

% note the indices.

% （检测costs的当前值是否比已经出现的最小值小。如果小的话，我们将当前值与最小值对调，并注明指数）


% 补充下面程序

minnum=256;

x=8;

y=8;

for i=1:row
    for j=1:col
        if (costs(i,j)<minnum)
            minnum=costs(i,j);
            x=i;
            y=j;
        end
    end
end

dx=x;

dy=y;

min=minnum;


