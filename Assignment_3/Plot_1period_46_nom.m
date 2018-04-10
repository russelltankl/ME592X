clc
clear
load('All46.mat')

for i = 1:46
    nom_46(:,i) = 2 * (nom_46(:,i) - min(nom_46(:,i))) / (range(nom_46(:,i))) - 1;
end

period = 1:3500;

var1 = nom_46(:,1);
var2 = nom_46(:,2);
var3 = nom_46(:,3);
var4 = nom_46(:,4);
var5 = nom_46(:,5);
var6 = nom_46(:,6);
var7 = nom_46(:,7);
var8 = nom_46(:,8);
var9 = nom_46(:,9);
var10 = nom_46(:,10);
var11 = nom_46(:,11);
var12 = nom_46(:,12);
var13 = nom_46(:,13);
var14 = nom_46(:,14);
var15 = nom_46(:,15);
var16 = nom_46(:,16);
var17 = nom_46(:,17);
var18 = nom_46(:,18);
var19 = nom_46(:,19);
var20 = nom_46(:,20);
var21 = nom_46(:,21);
var22 = nom_46(:,22);
var23 = nom_46(:,23);
var24 = nom_46(:,24);
var25 = nom_46(:,25);
var26 = nom_46(:,26);
var27 = nom_46(:,27);
var28 = nom_46(:,28);
var29 = nom_46(:,29);
var30 = nom_46(:,30);
var31 = nom_46(:,31);
var32 = nom_46(:,32);
var33 = nom_46(:,33);
var34 = nom_46(:,34);
var35 = nom_46(:,35);
var36 = nom_46(:,36);
var37 = nom_46(:,37);
var38 = nom_46(:,38);
var39 = nom_46(:,39);
var40 = nom_46(:,40);
var41 = nom_46(:,41);
var42 = nom_46(:,42);
var43 = nom_46(:,43);
var44 = nom_46(:,44);
var45 = nom_46(:,45);
var46 = nom_46(:,46);

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

title('Mean-shifted nominal data, 46 variable, 1 period')
xlabel('Time')
ylabel('Magnitude')