clc
clear
load('Pre-processed_variables_M_ano.mat')

for i = 1:27
    All_Variable_1_Period(:,i) = 2 * (All_Variable_1_Period(:,i) - min(All_Variable_1_Period(:,i))) / (range(All_Variable_1_Period(:,i))) - 1;
end

period = 1:3500;

var1 = All_Variable_1_Period(:,1);
var2 = All_Variable_1_Period(:,2);
var3 = All_Variable_1_Period(:,3);
var4 = All_Variable_1_Period(:,4);
var5 = All_Variable_1_Period(:,5);
var6 = All_Variable_1_Period(:,6);
var7 = All_Variable_1_Period(:,7);
var8 = All_Variable_1_Period(:,8);
var9 = All_Variable_1_Period(:,9);
var10 = All_Variable_1_Period(:,10);
var11 = All_Variable_1_Period(:,11);
var12 = All_Variable_1_Period(:,12);
var13 = All_Variable_1_Period(:,13);
var14 = All_Variable_1_Period(:,14);
var15 = All_Variable_1_Period(:,15);
var16 = All_Variable_1_Period(:,16);
var17 = All_Variable_1_Period(:,17);
var18 = All_Variable_1_Period(:,18);
var19 = All_Variable_1_Period(:,19);
var20 = All_Variable_1_Period(:,20);
var21 = All_Variable_1_Period(:,21);
var22 = All_Variable_1_Period(:,22);
var23 = All_Variable_1_Period(:,23);
var24 = All_Variable_1_Period(:,24);
var25 = All_Variable_1_Period(:,25);
var26 = All_Variable_1_Period(:,26);
var27 = All_Variable_1_Period(:,27);

hold on
plot(period,var1)
plot(period,var2)
plot(period,var3)
plot(period,var4)
plot(period,var5)
plot(period,var6)
plot(period,var7)
plot(period,var8)
plot(period,var9)
plot(period,var10)
plot(period,var11)
plot(period,var12)
plot(period,var13)
plot(period,var14)
plot(period,var15)
plot(period,var16)
plot(period,var17)
plot(period,var18)
plot(period,var19)
plot(period,var20)
plot(period,var21)
plot(period,var22)
plot(period,var23)
plot(period,var24)
plot(period,var25)
plot(period,var26)
plot(period,var27)

title('Mean-shifted anominal data, 27 variable, 1 period')
xlabel('Time')
ylabel('Magnitude')