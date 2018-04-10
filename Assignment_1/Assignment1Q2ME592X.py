# Question 2 - Time Series
# Part 1: Plot whole period of energy consumption
import csv
import matplotlib.pyplot as plt

x = []
y = []
z = []
i = 1

with open('energydata_complete.csv') as csvfile:
 reader = csv.DictReader(csvfile)
 for row in reader:
  x.append(i)
  y.append(float(row['Energy Consumption']))
  z.append(float(row['Appliances']))
  i += 1

plt.figure(figsize=(20,10))
plt.plot(x,y, label='Energy Consumption')
plt.xlabel('Time')
plt.ylabel('Energy Consumption')
plt.title('Energy Consumption for Whole Period')
plt.legend()
plt.show()

plt.figure(figsize=(20,10))
plt.plot(x,z, label='Appliances consumption')
plt.xlabel('Time')
plt.ylabel('Appliances consumption')
plt.title('Appliances consumption for Whole Period')
plt.legend()
plt.show()

# One week consumption
one_week = 7*24*6

x1 = x[0:one_week]
y1 = y[0:one_week]
z1 = z[0:one_week]

plt.figure(figsize=(20,10))
plt.plot(x1,y1, label='Energy Consumption')
plt.xlabel('Time')
plt.ylabel('Energy Consumption')
plt.title("One Week's Energy Consumption")
plt.legend()
plt.show()

plt.figure(figsize=(20,10))
plt.plot(x1,z1, label='Appliances Consumption')
plt.xlabel('Time')
plt.ylabel('Appliances Consumption')
plt.title("One Week's Appliances Consumption")
plt.legend()
plt.show()

#----------------------------------------------------------------------
# Part 2: Heatmap
import csv
import numpy as np
import matplotlib.pyplot as plt
y = []
i = 0

with open('energydata_complete.csv') as csvfile:
 reader = csv.DictReader(csvfile)
 for row in reader:
  y.append(int(row['Appliances'])) #column of appliances

a = len(y)/6
a = int(a)
y1 = [] # y1 = average App in 1 hour
for c in range(0,a+1):
 y1.append(np.mean(y[i:i+6]))
 i += 6

j = 0
y2 = [] # y2 = 1 day's App in each hour
for c in range(0,7):
 d = y1[j:j+24]
 y2.append(d)
 j += 24

plt.figure(figsize=(5,10))
plt.imshow(np.transpose(y2))
# Transposed to get a 'tall' heatmap

plt.ylim([-0.5,23.5])
plt.xticks(np.arange(0, 7, 1.0))
plt.yticks(np.arange(0, 24, 1.0))
plt.colorbar()
plt.ylabel('Hours')
plt.xlabel('Days')
plt.title('Appliances Wh')
plt.show()

#----------------------------------------------------------------------
# Part 3: Historgram of Energy consumption of Appliances
import csv
import matplotlib.pyplot as plt

x = []
y = []

with open('energydata_complete.csv') as csvfile:
 reader = csv.DictReader(csvfile)
 for row in reader:
  y.append(int(row['Appliances']))
  x.append(float(row['Energy Consumption']))

plt.hist(y,color='blue')
plt.ylabel('Frequency')
plt.xlabel('Appliances')
plt.title('Appliances Histogram')
plt.show()

plt.hist(x,color='blue')
plt.ylabel('Frequency')
plt.xlabel('Energy Consumption')
plt.title('Energy Consumption Histogram')
plt.show()

#----------------------------------------------------------------------
# Part 4: Energy Consumption vs NSM
import csv
import matplotlib.pyplot as plt
import numpy as np

x = []
y = []

with open('energydata_complete.csv') as csvfile:
 reader = csv.DictReader(csvfile)
 for row in reader:
  y.append(float(row['Appliances']))
  x.append(float(row['Energy Consumption']))

NSM = (24*60*60)
#print(NSM) #86400 data points
x_NSM = np.linspace(0,NSM-1,NSM)
#print(len(x_NSM))
#print(x_NSM[0:100])
#print(y[42])

y_Appliances = []
i = 0
index = 42 #starting from first 00.00
for i in range (0,NSM,600):
    for count in range (600):
        y_Appliances.append(y[index])
        #print(y_Consumption)
        #print('i:',i)
    index = index + 1


plt.figure(figsize=(20,10))
plt.plot(x_NSM,y_Appliances, label='Line 1')
plt.xlabel('NSM')
plt.ylabel('Appliances Consumption')
plt.title('Appliances Consumption vs NSM')
plt.legend()
plt.show()

y_Energy = []
i = 0
index = 42 #starting from first 00.00
for i in range (0,NSM,600):
    for count in range (600):
        y_Energy.append(x[index])
        #print(y_Consumption)
        #print('i:',i)
    index = index + 1


plt.figure(figsize=(20,10))
plt.plot(x_NSM,y_Energy, label='Line 1')
plt.xlabel('NSM')
plt.ylabel('Energy Consumption')
plt.title('Energy Consumption vs NSM')
plt.legend()
plt.show()

#----------------------------------------------------------------------
# Part 5: Plot appliances energy consumption vs. Press mm Hg
import csv
import matplotlib.pyplot as plt

x = []
y = []
z = []

with open('energydata_complete.csv') as csvfile:
 reader = csv.DictReader(csvfile)
 for row in reader:
  x.append(float(row['Press_mm_hg']))
  y.append(float(row['Energy Consumption']))
  z.append(float(row['Appliances']))

plt.figure(figsize=(20,10))
plt.plot(x,y, label='Line 1')
plt.xlabel('Press_mm_hg')
plt.ylabel('Energy Consumption')
plt.title('Energy Consumption vs Press_mm_hg')
plt.legend()
plt.show()

plt.figure(figsize=(20,10))
plt.plot(x,z, label='Line 1')
plt.xlabel('Press_mm_hg')
plt.ylabel('Appliances Consumption')
plt.title('Appliances Consumption vs Press_mm_hg')
plt.legend()
plt.show()
# To show all graph at once, type plt.show() only in the end of the LAST plot

#------------------------------------------------------------------------
# Part 6:
print("From the graphs shown above, it can be seen that NSM is a good predictor of appliance energy consumption because the peaks of energy consumption can clearly be seen occuring right after midnight and between 11 am to 6 pm.")
print("From the pressure graph, it can be seen that pressure isn't a good indicator of appliance energy consumption because on the same pressure point/range, the energy consumption can be high or low. It can only be observed that most the data collected were in the range of 748 to 765 mmHg.")
