%**********************************************************************
%THIS CODING IS FOR GENERATING TRAINING DATA FOR THE MACHINE LEARNING 
%**********************************************************************

clear all;
close all;
clc;

% 1.0   DEFINING THE PID PARAMETERS
%       -   this is for setting up the loop to simulate the outcome of the tuning
%       -   status if it is tuned at P = n1,n2,n3,...,n(n), I = n1,n2,n3,...,n(n) and
%       -   D = n1,n2,n3,...,n(n)
%       -   REF FROM MASTER STUDENTS:
%               for P= 0.015:0.03:0.75 % n = BIGGEST/INTERVAL + 1 (only + 1 if start from 0)
%               for I= 1:1:100

k = 1;
for D = 0
for P = 0.1:0.1:4    %0:1:3 40 COLUMN
for I = 0.1:0.1:4    %0:1:3 40 ROW
Ts=1;
sim('PID_ctrl_W6_03');

%setp_val(:,k) = sp;         %storing in an array of (:,k)
%outp_val(:,k) = op;
%proc_val(:,k) = pv;

gain(:,k) = P;
gain_int(:,k) = I;
gain_dev(:,k) = D;
IAE(:,k)=iae;
ISE(:,k)=ise;
ITAE(:,k)=itae;
S = stepinfo(pv);
OS(:,k)=S.Overshoot;
k = k + 1;

end
end
end


%7.0    THIS IS TO SQUARIFY DATA TO COMPLY AS THE FIGURE ILLUSTRATED ABOVE
%       -   Exporting to ExceL:
%           writematrix(M,'M.xls','Sheet',2,'Range','A3:E8')
%       -   Row = p
%       -   Col = i

%error = reshape(error(end_reading,:), [Row,Cok]); --> need adjust
% 81 = p
% 25 = I

IAE_X = reshape(IAE(end,:), [40,40]);
OS_X = reshape(OS,[40,40]);
ISE_X = reshape(ISE(end,:),[40,40]);
ITAE_X = reshape(ITAE(end,:),[40,40]);






% ......................CODE ENDS HERE..................................

