clc
clear
load('matlabLag.mat')

for i = 1:46
    datasourAno(:,i) = 2 * (datasourAno(:,i) - min(datasourAno(:,i))) / (range(datasourAno(:,i))) - 1;
end

period = 1:224870;

var1 = datasourAno(:,1);
var2 = datasourAno(:,2);
var3 = datasourAno(:,3);
var4 = datasourAno(:,4);
var5 = datasourAno(:,5);
var6 = datasourAno(:,6);
var7 = datasourAno(:,7);
var8 = datasourAno(:,8);
var9 = datasourAno(:,9);
var10 = datasourAno(:,10);
var11 = datasourAno(:,11);
var12 = datasourAno(:,12);
var13 = datasourAno(:,13);
var14 = datasourAno(:,14);
var15 = datasourAno(:,15);
var16 = datasourAno(:,16);
var17 = datasourAno(:,17);
var18 = datasourAno(:,18);
var19 = datasourAno(:,19);
var20 = datasourAno(:,20);
var21 = datasourAno(:,21);
var22 = datasourAno(:,22);
var23 = datasourAno(:,23);
var24 = datasourAno(:,24);
var25 = datasourAno(:,25);
var26 = datasourAno(:,26);
var27 = datasourAno(:,27);
var28 = datasourAno(:,28);
var29 = datasourAno(:,29);
var30 = datasourAno(:,30);
var31 = datasourAno(:,31);
var32 = datasourAno(:,32);
var33 = datasourAno(:,33);
var34 = datasourAno(:,34);
var35 = datasourAno(:,35);
var36 = datasourAno(:,36);
var37 = datasourAno(:,37);
var38 = datasourAno(:,38);
var39 = datasourAno(:,39);
var40 = datasourAno(:,40);
var41 = datasourAno(:,41);
var42 = datasourAno(:,42);
var43 = datasourAno(:,43);
var44 = datasourAno(:,44);
var45 = datasourAno(:,45);
var46 = datasourAno(:,46);

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
plot(period,var28)
plot(period,var29)
plot(period,var30)
plot(period,var31)
plot(period,var32)
plot(period,var33)
plot(period,var34)
plot(period,var35)
plot(period,var36)
plot(period,var37)
plot(period,var38)
plot(period,var39)
plot(period,var40)
plot(period,var41)
plot(period,var42)
plot(period,var43)
plot(period,var44)
plot(period,var45)
plot(period,var46)

title('Mean-shifted anominal data, 46 variable, all period')
xlabel('Time')
ylabel('Magnitude')