% clear all; 
% L=imread('222.jpg');  %read the first frame 
% I2=imread('222.jpg');  %read the second frame
% dm=7;
% L=double(L);
% I2=double(I2);
% [rownum,colnum] = size(L);    
% G=zeros(rownum+2*dm,colnum+2*dm); 
% G(dm+1:dm+rownum,dm+1:dm+co1num)=L; 
% for i=1:dm    
%     G(i,dm+1:dm+colnum)=G(dm+1,dm+1:dm+colnum);
%     G(rownum+dm+i,dm+1:dm+colnum)=G(dm+rownum,dm+1:dm+colnum); 
% end
% for j=1:dm
%     G(1:rownum+2*dm,j)=G(1:rownum+2*dm,dm+1);  
%     G(1:rownum+2*dm,colnum+dm+j)=G(1:rownum+2*dm,dm+colnum); 
% end
%  blocksize=16;
%  rowblocks =rownum/blocksize; 
%   colblocks =colnum/blocksize;
%   A=99999999999999999999;
%   Eij=0;  xrecord=ones(16,16);            %xrecord，yrecord用于存放匹配快的块号，即运动矢量 
%   yrecord=ones(16,16);  
%   diff=zeros(256,256);           %这幅图的大小为256*256  tic             
%   for x=0:(rowblocks-1)         %x表示行中第几个子块                  
%       row=x*blocksize;                  
%       for y=0:(colblocks-1)         %y表示列中第几个子块                   
%           col=y*blocksize; %                      
%           tempx=x*blocksize+1:(x+1)*blocksize; %                     
%           tempy=y*blocksize+1:(y+1)*blocksize;                                    
%           for p=-dm:dm 
%             for q=-dm:dm   
%                 Eij=0;            
%                 Eij=sum(sum((I2(row+1:row+blocksize,col+1:col+blocksize)-G(row+dm+p+1:row+dm+p+blocksize,col+dm+q+1:col+dm+q+blocksize)).^2))/(blocksize^2);           
%                 if Eij<A                                     
%                     A=Eij;                                  
%                     xrecord(x+1,y+1)=p;                                        
%                     yrecord(x+1,y+1)=q;                              
%                 end
%             end
%           end
%         A=999999999999999999;                        
%         for mx=1:blocksize                                
%             for ny=1:blocksize
%  diff(row+mx,col+ny)=I2(row+mx,col+ny)-G(row+mx+dm+xrecord(x+1,y+1),col+ny+dm+yrecord(x+1,y+1)); 
%             end
%         end
%       end
%   end
%   toc
%    figure,imshow(L,[]);   
%    title('the first frame');       
%    figure,imshow(I2,[]);            
%    title('the second frame');         
%    IIII=I2-L;           
%    figure,imshow(IIII,[]);  
%    title('帧间差值');          
%    figure,imshow(diff,[]);       
%    title('DFD');        
%    %title('利用全搜索算法匹配后的帧间差');        
%    for x=0:(rowblocks-1)         %      
%        row=x*blocksize;           
%        for y=0:(colblocks-1)         %         
%            col=y*blocksize; 
%  III(row+1:row+blocksize,col+1:col+blocksize)=G(row+dm+xrecord(x+1,y+1)+1:row+dm+xrecord(x+1,y+1)+blocksize,col+dm+yrecord(x+1,y+1)+1:col+dm+yrecord(x+1,y+1)+blocksize)+diff(row+1:row+blocksize,col+1:col+blocksize); 
%        end
%    end
%    %III=I1+abs(diff);       
%    figure,imshow(III,[]);         
%    title('?????ó????????????');                    
%    ERR=diff; %                     figure,imshow(ERR,[]); %         title('DFD');
%    numberarray=0:1:255; 
%    for m=1:255  
%        numberarray(m+1)=0; 
%    end;  
%    zeronumber=0; 
%    for n=1:rownum
%        for m=1:colnum   
%            dif=abs(ERR(m,n));   
%            if(dif==0)    
%                temp=zeronumber;
%                zeronumber=temp+1;   
%            else
%                numberarray(dif)=numberarray(dif)+1;  
%            end;   
%        end;  
%    end; 
%    figure;plot(0,zeronumber,'k*');hold on;plot(numberarray,'r*'),title('DFD distribution');hold off;
%    ERR1=zeros(16,16);
%    for i=0:15    
%        for j=0:15   
%            ERR1(i+1,j+1)=round(sum(sum(ERR(i*blocksize+1:i*blocksize+blocksize,j*blocksize+1:j*blocksize+blocksize)))/(blocksize*blocksize));   
%        end
%    end
%    numberarray=0:1:255; 
%    for m=1:255    
%        numberarray(m+1)=0; 
%    end;  zeronumber=0; 
%    for n=1:16  
%        for m=1:16 
%            dif=abs(ERR1(m,n));  
%            if(dif==0)    
%                temp=zeronumber; 
%                zeronumber=temp+1;   
%            else
%                numberarray(dif)=numberarray(dif)+1; 
%            end;   
%        end; 
%    end; 
%    figure;plot(0,zeronumber,'k*');hold on;plot(numberarray,'r*'),title('DFD(block average) distribution');hold off;
%   figure; 
%   for i=1:16   
%       for j=1:16      
%           quiver(i,j,xrecord(i,j)/16,yrecord(i,j)/16); hold on; 
%       end
%   end
%   grid on;
%   figure;quiver(1:16,1:16,yrecord,xrecord); 
%   grid on;    
 % S_3SS: clear all;
  I1=imread('222.JPG'); 
  %read the first frame 
  I2=imread('222.JPG');
  %read the second frame 
  dm=7;
  I1=double(I1); 
  I2=double(I2); 
  [rownum, colnum]=size(I1);
  II=zeros(rownum+2*dm,colnum+2*dm); 
  II(dm+1:dm+rownum,dm+1:dm+co1num)=I1; 
  for i=1:dm    
      II(i,dm+1:dm+colnum)=II(dm+1,dm+1:dm+colnum);   
      II(rownum+dm+i,dm+1:dm+colnum)=II(dm+rownum,dm+1:dm+colnum); 
  end
  for j=1:dm    
      II(1:rownum+2*dm,j)=II(1:rownum+2*dm,dm+1);   
      II(1:rownum+2*dm,colnum+dm+j)=II(1:rownum+2*dm,dm+colnum);
  end
 
  blocksize=16; 
  rowblocks =rownum/blocksize; 
  colblocks =colnum/blocksize; 
  A=99999999999999999999;      
  % Eij=0; 
  xrecord=ones(16,16);    
  % yrecord=ones(16,16);  
  diff=zeros(256,256);         
  % tic          
  for x=0:(rowblocks-1)     
      %             
      row=x*blocksize;          
      for y=0:(colblocks-1)       
          %                      
          col=y*blocksize; %   tempx=x*blocksize+1:(x+1)*blocksize; %       tempy=y*blocksize+1:(y+1)*blocksize;
   for p1=-4:4:4   
       %??????      
       for q1=-4:4:4     %   
           Eij=0;                  
           Eij=sum(sum((I2(row+1:row+blocksize,col+1:col+blocksize)-II(row+dm+p1+1:row+dm+p1+blocksize,col+dm+q1+1:col+dm+q1+blocksize)).^2))/(blocksize^2);           
           if Eij<A           
               A=Eij;       
               xrecord(x+1,y+1)=p1;   
               yrecord(x+1,y+1)=q1;        
           end
       end
   end
   p1=xrecord(x+1,y+1);     
   q1=yrecord(x+1,y+1);         
   for p2=p1-2:2:p1+2     
       %??????                
       for q2=q1-2:2:q1+2      
           if p2~=p1 | q2~=q1      
               Eij=0;              
               Eij=sum(sum((I2(row+1:row+blocksize,col+1:col+blocksize)-II(row+dm+p2+1:row+dm+p2+blocksize,col+dm+q2+1:col+dm+q2+blocksize)).^2))/(blocksize^2);    
               if Eij<A      
                   A=Eij;        
                   xrecord(x+1,y+1)=p2;   
                   yrecord(x+1,y+1)=q2;    
               end
           end
       end
   end
   p2=xrecord(x+1,y+1);                
   q2=yrecord(x+1,y+1);                
   for p3=p2-1:1:p2+1        %??????    
       for q3=q2-1:1:q2+1               
           if p3~=p2 | q3~=q2           
               Eij=0;                   
               Eij=sum(sum((I2(row+1:row+blocksize,col+1:col+blocksize)-II(row+dm+p3+1:row+dm+p3+blocksize,col+dm+q3+1:col+dm+q3+blocksize)).^2))/(blocksize^2); 
               if Eij<A
    A=Eij;          
    xrecord(x+1,y+1)=p3;       
    yrecord(x+1,y+1)=q3;    
               end
           end
       end
       
   end
   A=999999999999999999;    
   for mx=1:blocksize         
       for ny=1:blocksize      
           diff(row+mx,col+ny)=I2(row+mx,col+ny)-II(row+mx+dm+xrecord(x+1,y+1),col+ny+dm+yrecord(x+1,y+1));     
       end
       
   end
      end
  end
  toc   
  figure,imshow(I1,[]);  
  title('the first frame'); 
  figure,imshow(I2,[]);         
  title('the second frame');    
  IIII=I2-I1;     
  figure,imshow(IIII,[]);     
  title('????????');          
  figure,imshow(diff,[]);     
  title('DFD');   
  
  for x=0:(rowblocks-1)     
      
      row=x*blocksize;     
      for y=0:(colblocks-1)   
          col=y*blocksize;      
          III(row+1:row+blocksize,col+1:col+blocksize)=II(row+dm+xrecord(x+1,y+1)+1:row+dm+xrecord(x+1,y+1)+blocksize,col+dm+yrecord(x+1,y+1)+1:col+dm+yrecord(x+1,y+1)+blocksize)+diff(row+1:row+blocksize,col+1:col+blocksize);    
      end
  end
  figure,imshow(III,[]);  
  title('恢复后的第二帧图像')  
  ERR=diff;
   numberarray=0:1:255; 
   for m=1:255  
       numberarray(m+1)=0;
   end;  
   zeronumber=0; 
   for n=1:rownum  
       for m=1:colnum  
           dif=abs(ERR(m,n)); 
           if(dif==0)    
               temp=zeronumber; 
               zeronumber=temp+1;    
           else
               numberarray(dif)=numberarray(dif)+1;
           end; 
       end;  
   end; 
   figure;plot(0,zeronumber,'k*');hold on;plot(numberarray,'r*'),title('DFD distribution');hold off; 
   ERR1=zeros(16,16); 
   for i=0:15  
       for j=0:15     
           ERR1(i+1,j+1)=round(sum(sum(ERR(i*blocksize+1:i*blocksize+blocksize,j*blocksize+1:j*blocksize+blocksize)))/(blocksize*blocksize));   
       end
   end
   numberarray=0:1:255;
   for m=1:255   
       numberarray(m+1)=0; 
   end;  
   zeronumber=0; 
   for n=1:16 
       for m=1:16     
           dif=abs(ERR1(m,n));  
           if(dif==0)      
               temp=zeronumber; 
               zeronumber=temp+1;   
           else
               numberarray(dif)=numberarray(dif)+1;   
           end;
             end; 
   end; 
   figure;plot(0,zeronumber,'k*');hold on;plot(numberarray,'r*'),title('DFD(block average) distribution');hold off;    
   %figure;mesh(diff);figure;contour(diff,15); 
   figure; 
   for i=1:16  
       for j=1:16    
           quiver(i,j,xrecord(i,j)/16,yrecord(i,j)/16); hold on;    
       end
   end
   grid on; figure;quiver(1:16,1:16,yrecord,xrecord); 
   grid on;
 
 
 