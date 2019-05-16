#!/usr/bin/env python

import numpy as np
import pandas as pd

def get_sepsis_score(data, model):
#    df = pd.DataFrame({'HR':data[:,0],'O2Sat':data[:,1],'Temp':data[:,2],'SBP':data[:,3],'MAP':data[:,4],'DBP':data[:,5],'Resp':data[:,6],'EtCO2':data[:,7],'BaseExcess':data[:,8],'HCO3':data[:,9],'FiO2':data[:,10],'pH':data[:,11],'PaCO2':data[:,12],'SaO2':data[:,13],'AST':data[:,14],'BUN':data[:,15],'Alkalinephos':data[:,16],'Calcium':data[:,17],'Chloride':data[:,18],'Creatinine':data[:,19],'Bilirubin_direct':data[:,20],'Glucose':data[:,21],'Lactate':data[:,22],'Magnesium':data[:,23],'Phosphate':data[:,24],'Potassium':data[:,25],'Bilirubin_total':data[:,26],'TroponinI':data[:,27],'Hct':data[:,28],'Hgb':data[:,29],'PTT':data[:,30],'WBC':data[:,31],'Fibrinogen':data[:,32],'Platelets':data[:,33],'Age':data[:,34],'Gender':data[:,35],'Unit1':data[:,36],'Unit2':data[:,37],'HospAdmTime':data[:,38],'ICULOS':data[:,39]})

#    print(data)

    indices = [0,5,7,9,10,11,13,17,19,20,22,23,25,28,29,31,36,37,39]

    df = np.take(data[-1],indices)

#    print(df)
#    print(df)
#    print(type(df))
    df = np.nan_to_num(df)
#    print(df)
#    print(df)
#    print(df)
#    print(type(df))
#    df = pd.read_csv(df, sep='|')
#    x = df.loc[:,['HR', 'DBP', 'EtCO2', 'HCO3', 'FiO2', 'pH', 'SaO2', 'Calcium', 'Creatinine', 'Bilirubin_direct', 'Lactate', 'Magnesium', 'Potassium', 'Hct', 'Hgb', 'WBC', 'Unit1', 'Unit2', 'ICULOS']]
#    x.fillna(value=0, inplace=True)
#    x = np.array(x)
    pred = np.dot(df, model)
    score = 1/(1+np.exp(-pred))

    prob = score.copy()

#    print(prob)
#    print(type(score))

    if score>0.8:
        score = 1
    else:
        score = 0
#    score[score>0.8] = 1
#    score[score<=0.8] = 0

    return prob, score

def load_sepsis_model():
    coeff = np.array([ 0.01380884, -0.01723406,  0.04433786,  0.06454266,  2.19654009, -0.14405364,
		   0.01312669,  0.03321688,  0.13299622,  0.25669214, -0.04325157, -0.11289885,
		  -0.023059  ,  0.01526274, -0.07127261,  0.03342408, -0.1839546 , -0.49297878,
		   0.01532965])
    return coeff
