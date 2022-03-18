# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:10:26 2022

@author: LENOVO
"""

# 1 Sample Sign Test
# 1 Sample Z-Test
# Mann-Whitney test
# Paired T-Test
# Moods-Median Test
# 2 sample T-Test
# One - Way Anova
# 2-Proportion
# Chi-Square Test
# Tukey's Test

# to determine whether there is any significant difference in the diameter of the cutlet between two units
# Cutlets.csv

import pandas as pd
import numpy as np
import scipy
from scipy import stats
import statsmodels.stats.descriptivestats as sd
from statsmodels.stats import weightstats as stests


############ 1 Sample Sign Test ################
# Student Scores Data
cutlets = pd.read_csv("D:\datasets\Cutlets.csv")
cutlets.columns
cutlets.head()


# checking null values
cutlets.isna().sum()

# Normal Q-Q plot
import pylab

# Checking Whether data is normally distributed
stats.probplot(cutlets['Unit A'], dist="norm", plot=pylab)
stats.probplot(cutlets['Unit B'], dist="norm", plot=pylab)

unitA=pd.Series(cutlets.iloc[:,0])
unitA

unitB=pd.Series(cutlets.iloc[:,1])
unitB

# 2-sample 2-tail ttest:   stats.ttest_ind(array1,array2)     # ind -> independent samples
p_value=stats.ttest_ind(unitA,unitB)
p_value

p_value[1]     # 2-tail probability 

# compare p_value with α = 0.05 (At 5% significance level)
'''
Assume Null hyposthesis as Ho: μ1 = μ2 (There is no difference in diameters of cutlets between two units).

Thus Alternate hypothesis as Ha: μ1 ≠ μ2 (There is significant difference in diameters of cutlets between two units) 2 Sample 2 Tail test applicable
'''


# ======================================== *********************** =====================    
# 2. Objective: A hospital wants to determine whether there is any difference in the
'''average Turn Around Time (TAT) of reports of the laboratories on their
preferred list. They collected a random sample and recorded TAT for
reports of 4 laboratories. TAT is defined as sample collected to report
dispatch.
Analyze the data and determine whether there is any difference in
average TAT among the different laboratories at 5% significance level.'''

lab_tat = pd.read_csv('D:\datasets\lab_tat_updated.csv')
lab_tat.head()

# Anova ftest statistics: stats.f_oneway(column-1,column-2,column-3,column-4)
p_value = stats.f_oneway(lab_tat.iloc[:,0],lab_tat.iloc[:,1],lab_tat.iloc[:,2],lab_tat.iloc[:,3])
p_value

p_value[1]  # compare it with α = 0.05
'''
Anova ftest statistics: Analysis of varaince between more than 2 samples or columns Assume Null Hypothesis Ho as No Varaince: All samples TAT population means are same

Thus Alternate Hypothesis Ha as It has Variance: Atleast one sample TAT population mean is different '''

# ============================== ********************** =========================

# 3. Objective: Sales of products in four different regions is tabulated for males and females. Find if male-female buyer rations are similar across regions.
        
buyer_ratio= pd.read_csv('D:\datasets\BuyerRatio.csv')
buyer_ratio.head()

buyer_table = buyer_ratio.iloc[:,1:6]
buyer_table

buyer_table.values
val=stats.chi2_contingency(buyer_table)
val
type(val)

no_of_rows=len(buyer_table.iloc[0:2,0])
no_of_columns=len(buyer_table.iloc[0,0:4])
degree_of_f=(no_of_rows-1)*(no_of_columns-1)
print('Degree of Freedom=',degree_of_f)

Expected_value=val[3]
Expected_value

# chi2 test or chi  square test
from scipy.stats import chi2
chi_square=sum([(o-e)**2/e for o,e in zip(buyer_table.values,Expected_value)])
chi_square_statestic=chi_square[0]+chi_square[1]
chi_square_statestic

critical_value=chi2.ppf(0.95,3)
critical_value

if chi_square_statestic >= critical_value:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')

pvalue=1-chi2.cdf(chi_square_statestic,3)
pvalue

if pvalue <= 0.05:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')

# ==================== ************************* ===============================

# 4. Telecall uses 4 centers around the globe to process customer order forms.
'''They audit a certain % of the customer order forms. Any error in order form
renders it defective and must be reworked before processing. The manager
wants to check whether the defective % varies by center. Please analyze
the data at 5% significance level and help the manager draw appropriate
inferences '''

custom = pd.read_csv('D:\datasets\CustomerOrderform.csv')
custom.columns

print(custom['Phillippines'].value_counts(),custom['Indonesia'].value_counts(),custom['Malta'].value_counts(),custom['India'].value_counts())
observed=([[271,267,269,280],[29,33,31,20]])

from scipy.stats import chi2_contingency

stat, p, dof, expected = chi2_contingency([[271,267,269,280],[29,33,31,20]])
stat
p

print('dof=%d' % dof)
print(expected)

alpha = 0.05
prob=1-alpha
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
	print('Dependent (reject H0),variables are related')
else:
	print('Independent (fail to reject H0), variables are not related')

print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')

# =============================== ****************************** ====================

# 5. Fantaloons Sales managers commented that % of males versus females
'''walking into the store differ based on day of the week. Analyze the data
and determine whether there is evidence at 5 % significance level to
support this hypothesis. '''

######### 2-Proportion Test #########
import numpy as np

fantaloons = pd.read_csv("D:\datasets\Fantaloons.csv")
fantaloons.head()
fantaloons.isna().sum()

# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
fantaloons['Weekdays']= label_encoder.fit_transform(fantaloons['Weekdays'])
fantaloons['Weekend']= label_encoder.fit_transform(fantaloons['Weekend'])

# mean imputation
fantaloons['Weekdays'] = fantaloons['Weekdays'].fillna(fantaloons['Weekdays'].mode())
fantaloons['Weekend'] = fantaloons['Weekend'].fillna(fantaloons['Weekend'].mode())


from statsmodels.stats.proportion import proportions_ztest

tab1 = fantaloons.Weekdays.value_counts()
tab1
tab2 = fantaloons.Weekend.value_counts()
tab2

# crosstable table
pd.crosstab(fantaloons.Weekdays, fantaloons.Weekend)

count = np.array([167, 120])
nobs = np.array([66, 47])

stats, pval = proportions_ztest(count, nobs, alternative = 'two-sided') 
print(pval)
stats, pval = proportions_ztest(count, nobs, alternative = 'larger')
print(pval)  


                     ####### Chi-Square Test #########


count = pd.crosstab(fantaloons['Weekdays'], fantaloons['Weekend'])
count
Chisquares_results = scipy.stats.chi2_contingency(count)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square

 # ================================ end

