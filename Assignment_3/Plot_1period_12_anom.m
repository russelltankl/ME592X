clc
clear
load('Pre-processed_variables_M_ano.mat')

for i = 1:12
    Selected_Variable_1_Period(:,i) = 2 * (Selected_Variable_1_Period(:,i) - min(Selected_Variable_1_Period(:,i))) / (range(Selected_Variable_1_Period(:,i))) - 1;
end

period = 1:3500;

var1 = Selected_Variable_1_Period(:,1);
var2 = Selected_Variable_1_Period(:,2);
var3 = Selected_Variable_1_Period(:,3);
var4 = Selected_Variable_1_Period(:,4);
var5 = Selected_Variable_1_Period(:,5);
var6 = Selected_Variable_1_Period(:,6);
var7 = Selected_Variable_1_Period(:,7);
var8 = Selected_Variable_1_Period(:,8);
var9 = Selected_Variable_1_Period(:,9);
var10 = Selected_Variable_1_Period(:,10);
var11 = Selected_Variable_1_Period(:,11);
var12 = Selected_Variable_1_Period(:,12);

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

title('Mean-shifted anominal data, 12 variable, 1 period')
xlabel('Time')
ylabel('Magnitude')