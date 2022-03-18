# -*- coding: utf-8 -*-
'''
Name: __RAJU BOTTA___________ Batch ID: __05102021_________
Topic: Exploratory Data Analysis

Q1) Calculate Skewness, Kurtosis using R/Python code & draw inferences on the following data.
Hint: [Insights drawn from the data such as data is normally distributed/not, outliers, measures like mean, median, mode, variance, std. deviation]

#  a.   Cars speed and distance  '''
   

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#create data frame using the given 2 inputs
lst = [4,4,7,7,8,9,10,10,10,11,11,12,12,12,12,13,13,13,13,14,14,14,14,15,15,15,16]
lst2 = [2,10,4,22,16,10,18,26,34,17,28,14,20,24,28,26,34,34,46,26,36,60,80,20,26,54,32]    

df = pd.DataFrame(list(zip(lst, lst2)), columns =['speed', 'distance'])
   
#measures of central tendency / 1st moment business decision
df.speed.mean()
df.distance.mean()
df.speed.median()
df.distance.median()
df.speed.mode()

df.distance.mode()
   
#Measures of Dispersion / 2nd moment business decision
df.speed.var()
df.distance.var()
df.speed.std()
df.distance.std()

#3rd moment business decision / skewness
df.speed.skew()
df.distance.skew()

#4th moment business decision / kurtosis
df.speed.kurt()
df.distance.kurt()

#visualization

df.shape
plt.bar(height=df.speed, x=np.arange(0,27))
 
plt.hist(df.distance)
# or
plt.hist(lst2,ec='red')
plt.xlabel('distance')
plt.ylabel('speed')
plt.title('cars')

plt.boxplot(df.distance)
 
# **** insights from the above data ****

'''data is not normally distributed (histogram)
data has outliers - upper whisker
data has outliers so can take median
skewness for speed is negative or left skewed
skewness for distance is positively skewed or right skewed
kurtosis for speed is mesokurtic distribution (near to zero)
kurtosis for distance is positive or leptokurtic '''
# ========================================================


#) b. Top Speed (SP) and Weight (WT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

list1 = [104.1854,105.4613,105.4613,133.4613,104.4613,113.1854,105.4613,102.5985,102.5985,115.6452,111.1854,117.5985,122.1051,111.1854,108.1854,111.1854,114.3693,117.5985,114.3693,118.4729,119.1051,110.8408,120.289,113.8291,119.1854,114.5985,120.7605,119.1051,99.56491,121.8408,113.4846,112.289,119.9211,121.3926]
list2 = [28.76206,30.46683,30.1936,30.63211,29.88915,29.59177,30.30848,15.84776,16.35948,30.92015,29.36334,15.75353,32.81359,29.37844,29.34728,29.60453,29.53578,16.19412,29.92939,33.51697,32.32465,34.90821,32.67583,31.83712,28.78173,16.04317,38.06282,32.83507,34.48321,35.54936,37.04235,33.23436,31.38004,37.57329]        
df1 = pd.DataFrame(list(zip(list1,list2)), columns = ['SP', 'WT'])
df1

#measures of central tendency / 1st moment business decision

df1.SP.mean()
df1.WT.mean()
df1.SP.median()
df1.WT.median()
df1.SP.mode()
df1.SP.mode()
   
#Measures of Dispersion / 2nd moment business decision
df1.SP.var()
df1.WT.var()
df1.SP.std()
df1.WT.std()

#3rd moment business decision / skewness
df1.SP.skew()
df1.WT.skew()

#4th moment business decision / kurtosis
df1.SP.kurt()
df1.WT.kurt()

#visualization
df1.shape
plt.bar(height=df1.WT, x=np.arange(0,34))
 
plt.hist(df1.SP)  #To know data is normally distributed or not
    
plt.boxplot(df1.SP) #to know outliers are there or not
 
#the distribution is normal

plt.boxplot(df1.WT)
 
#the distribution is not normal outliers are there at lower whiskers.

#insights

'''data is not normally distributed (histogram)
weight (WT) has outliers - upper whisker & lower whisker
top speed (SP) has outliers
data has outliers so can take median as its numerical
skewness for top speed(SP) is positively skewed or right skewed
skewness for weight (WT) is negative skewed or left skewed
kurtosis for top speed (SP) is mesokurtic distribution (near to zero)
kurtosis for distance is positive or leptokurtic '''

# ************* ================== ************************


#Q2) Draw inferences about the following boxplot & histogram. 

''' Histogram - given is not normally distributed

Boxplot- the distribution is not normal & outliers are there - lower & upper whisker
'''

#Q3)  Below are the scores obtained by a student in tests 

import statistics

stm = [34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56]

df2 = pd.DataFrame(stm)

x = statistics.mean(stm)
print("mean is:", x)

x1= statistics.median(stm)
print("median is:", x1)

x2 = statistics.variance(stm)
print('variance is:', x2)

x3 = statistics.stdev(stm)
print('standard deviation is:', x3)

x4 = df2.skew()
print("skewness is:", x4)

import matplotlib.pyplot as plt

plt.hist(df2) #To know data is normally distributed or not
plt.boxplot(df2) #to know outliers are there or not

#insights from above inferences
''' mean -41, median - 40.5, variance- 25.529411764705884, standard deviation 5.05266382858645
given data is not normal distribution,
data is right skewed or positively skewed,
outliers are there @ upper whisker'''

#Q5)  what is the nature of skewness when mean, median of data are equal
# when mean, median of data are equal then the distribution is symmetric the distribution has zero skewness

#Q6) What is the nature of skewness when mean > median?
# If the mean is greater than the median, the distribution is positively skewed.
 
#Q7) What is the nature of skewness when median > mean?
# If the mean is less than the median, the distribution is negatively skewed

#Q8) what does positive kurtosis value indicate for a data?
# Positive excess values of kurtosis (>3) indicate that a distribution is peaked and possess thick tails. Leptokurtic distributions have positive kurtosis values. A leptokurtic distribution has a higher peak (thin bell) and taller (i.e. fatter and heavy) tails than a normal distribution.

#Q9) what does negative kurtosis value indicate for a data?
# A negative kurtosis means that the distribution is flatter than a normal curve with the same mean and standard deviation. ... This means your distribution is platykurtic or flatter as compared with normal distribution with the same Mean and Standard Deviation. The curve would have very light tails.

#Q10) Box plot inferences
''' the data is not normal distribution and is lower whisker,
nature of skewness of data is negative skewed or left skewed,
IQR (Inter Quarter Range) of the data is in between 18 - 10 = 8 ... i.e, IQR = Q3-Q1 '''

# Q11) Box plot visualizations
''' the given plot is distributed normally,
    no outliers are present in the given plot,
    the distribution has zero skewness since the plot is symmetric distribution '''
   
# Q12) Box plot of variable 'x'

'''  1) IQR of given plot is Q3-Q1 = 18-5 = 13
The interquartile range is a measure of where the “middle fifty” is in a data set. IQR value is between 25th quartile and 75th quartile. i.e, Q3-Q1 = IQR
2) Given plot is left or negative skewed

'''

 # Q13)
 
''' 1) The mode is the data value that occurs the most often in a data set. ... The mode always occurs at the highest point of the peak. the mode of the dataset lie at 22
2) the histogram is right skewed or positively skewed

3) visualise histogram and box plots '''

