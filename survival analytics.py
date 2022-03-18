# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 14:22:04 2022

Name:	 _BOTTA RAJU____________
Batch Id: 05102021		     
Topic: Survival Analytics

"""

# 1. Objective: The following dataset contains patient ID, follow up, event type, and scenarios. Build a survival analysis model on the given data.

import lifelines
import pandas as pd
import numpy as np

# Loading the survival un-employment data
patient = pd.read_csv('D:\Data Set\patient.csv')
patient.describe()
patient.columns

patient.head()

patient.drop(columns='PatientID', axis=1, inplace=True)

# spell is reffering to time
T = patient.Followup

# Importing the KaplanMeierFitter model to fit the survival analysis
from lifelines import KaplanMeierFitter

# Initiating the KaplanMeierFitter model
kmf = KaplanMeierFitter()

# Fitting KaplanMeierFitting model
kmf.fit(T, event_observed=patient.Eventtype)

# Time-line estimations plot
kmf.plot()

# over multiple groups
# for each group, here group is scenario
patient.Scenario.value_counts()

# applying KaplanMeierFitter model on Time and Events for the group "1"
kmf.fit(T[patient.Followup==1], patient.Eventtype[patient.Followup==1], label='1')
ax = kmf.plot()

# applying KaplaneMeierFitter model on Time and Events for the group "0"
kmf.fit(T[patient.Followup==0], patient.Eventtype[patient.Followup==0], label='0')
ax1 = kmf.plot()

patient.Followup 
patient.Eventtype

# ================ ****************** ======================

# 2. Objective: Perform survival analysis on ECG of different age groups of people and provide your insights in the documentation

import pandas as pd
# loading the data
ecg_surv = pd.read_excel('/content/ECG_Surv.xlsx')
ecg_surv.head()
ecg_surv['survival_time_hr'].describe

# survival_time_hr Reffering to time
T = ecg_surv.survival_time_hr

# importing the kaplanMeierFitter model to fit the survival analysis
from lifelines import KaplaneMeierFitter

# initiating the KaplanMeierFitter model
kmf = KaplaneMeierFitter()

# Fitting KaplanMeierFitter model on time and events
kmf.fit(T, event_observed=ecg_surv.Eventtype)

# time-line estimations plot
kmf.plot()

# over multiple groups
# for each group here group is group
ecg_surv.group.value_counts()

# Applying KaplanMeierFitter model on Time and Events for the group "1"
kmf.fit(T[ecg_surv.group==1], ecg_surv.event[ecg_surv.group==1], label='1')
ax = kmf.plot()

# Applying KaplanMeierFitter model on Time and Events for the group "0"
kmf.fit(T[ecg_surv.group==0], ecg_surv.event[ecg_surv.group==0], label='0')
kmf.plot(ax=ax)

