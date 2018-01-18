
# coding: utf-8

# In[23]:

# Q3
import csv
import numpy as np
import scipy.stats as sp

Freq = []
AoA = []
ChordL = []
Velo = []
Thick = []
Sound = []


# Import dat file & extract data
with open('airfoil_self_noise.dat') as datfile:
    for line in datfile:
        Freq.append(int(line.split()[0]))
        AoA.append(float(line.split()[1]))
        ChordL.append(float(line.split()[2]))
        Velo.append(float(line.split()[3]))
        Thick.append(float(line.split()[4]))
        Sound.append(float(line.split()[5]))


# In[24]:

def compute_stat(value):
    
    mean = np.mean(value)
    print('Mean:', mean)
    
    variance = np.var(value)
    print('Variance:', variance)
    
    median = np.median(value)
    print('Median:', median)
    
    kurt = sp.kurtosis(value)
    print('Kurtosis:', kurt)
    
    skew = sp.skew(value)
    print('Skewness:', skew)
    
    rang = np.ptp(value)
    print('Range:', rang)
            
        


# In[25]:

print('Descriptive Statistics of Frequency, in Hz')
compute_stat(Freq)
print()
print('Descriptive Statistics of Angle of attack, in degrees')
compute_stat(AoA)
print()
print('Descriptive Statistics of Chord length, in meters')
compute_stat(ChordL)
print()
print('Descriptive Statistics of Free-stream velocity, in meters per second')
compute_stat(Velo)
print()
print('Descriptive Statistics of Suction side displacement thickness, in meters')
compute_stat(Thick)
print()
print('Descriptive Statistics of Scaled sound pressure level, in decibels')
compute_stat(Sound)


# In[ ]:



