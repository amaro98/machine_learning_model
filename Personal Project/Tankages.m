clear; clc; close all;

%inputdat = xlsread('D:\00 Personal File\000 CAREER\New Tank 7245-7250 Data and Matlab code\Dataset_Tankages_T7245_100D new');
inputdat = xlsread('D:\00 Personal File\000 CAREER\New Tank 7245-7250 Data and Matlab code\Dataset_Tankages_T7246_100D new');
%inputdat = xlsread('D:\00 Personal File\000 CAREER\New Tank 7245-7250 Data and Matlab code\Dataset_Tankages_T7247_150D new');
%inputdat = xlsread('D:\00 Personal File\000 CAREER\New Tank 7245-7250 Data and Matlab code\Dataset_Tankages_T7248_150D new');
%inputdat = xlsread('D:\00 Personal File\000 CAREER\New Tank 7245-7250 Data and Matlab code\Dataset_Tankages_T7249_500D new');
%inputdat = xlsread('D:\00 Personal File\000 CAREER\New Tank 7245-7250 Data and Matlab code\Dataset_Tankages_T7250_500D new');

%inputdat = inputdat(1:10,:);
[row,col]=size(inputdat);
% remove 1st column and the last 3 rows (contains max and min only and
% dates)
inputdat = inputdat(1:row-3,1:col);
[row1,col1]=size(inputdat);

%% for shifting rows (no interpolation)
inputdatnew = zeros(row1-1,59);

k = 2;

for iii=1:row1-1;

 inputdatnew(k-1,:)=[inputdat(k,1:17) inputdat(k-1,18:38) inputdat(k,18:38)];

 k=k+1;

end
    


%% end of delete rows (no interpolation)

% assigning inputs and targets for machine learning development
dat_1 = inputdatnew(:,1:38); %input variables
dat_2 = inputdatnew(:,39:59); %output variables
[row2,col2]= size(dat_2);

% assigning outputs manually
y1 = dat_2(:,1);
y2 = dat_2(:,2);
y3 = dat_2(:,3);
y4 = dat_2(:,4);
y5 = dat_2(:,5);
y6 = dat_2(:,6);
y7 = dat_2(:,7);
y8 = dat_2(:,8);
y9 = dat_2(:,9);
y10 = dat_2(:,10);
y11 = dat_2(:,11);
y12 = dat_2(:,12);
y13 = dat_2(:,13);
y14 = dat_2(:,14);
y15 = dat_2(:,15);
y16 = dat_2(:,16);
y17 = dat_2(:,17);
y18 = dat_2(:,18);
y19 = dat_2(:,19);
y20 = dat_2(:,20);
y21 = dat_2(:,21);

%run nftool
nftool

%%

y1_opt = T7426_y1(dat_1)

plot(y1_opt)