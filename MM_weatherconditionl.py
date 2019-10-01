# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:23:22 2019
https://www.datacamp.com/community/tutorials/markov-chains-python-tutorial
@author: lidon
"""

import random 
import numpy as np
import pandas as pd
from pandas import read_csv

df=read_csv("Phoneix_HMM_data.csv") 
x = df.loc[df['Conditions_Name'].isin(["Clear","Mostly Cloudy","Partly Cloudy","Scattered Clouds",'Overcast'])]
x = x.dropna()
x.Conditions_Name_1 = x.Conditions_Name_1.shift(-1)
x = x.dropna()
days=8760
real_data=x.iloc[-days:,4]
real_data=pd.DataFrame(real_data)
real_data.Conditions_Name.value_counts()
star= x.iloc[-(days+1),4]
# The statespace
states = ["Clear" ,"Mostly Cloudy","Partly Cloudy","Scattered","Overcast"]
# Possible sequences of events
transitionName = [['CC',"CM","CP","CS",'CO'],["MC","MM",'MP',"MS",'MO'],
                  ["PC","PM","PP",'PS','PO'],['SC','SM','SP','SS','SO'],
                  ['OC','OM','OP','OS','OO']]

# Probabilities matrix (transition matrix)
transitionMatrix = [[0.8955, 0.0036, 0.0869, 0.0139, 0.0001],
                     [0.0029, 0.8193, 0.0128, 0.1162, 0.0488],
                     [0.1208, 0.0213, 0.7501, 0.1074, 0.0004],
                     [0.0166, 0.1420, 0.1320, 0.7074, 0.0020],
                     [0.0007, 0.2446, 0.0015, 0.0153, 0.7379]]

def activity_forecast(days,star):
    # Choose the starting state
    activityToday = star
    activityList = [activityToday]
#    activityList = []
    i = 0
#    prob_prce=[]
    prob = 1
    while i != days:
        if activityToday == "Clear":
            change = np.random.choice(transitionName[0],replace=True,p=transitionMatrix[0])
            if change == "CC":
                prob = prob * 0.896
                activityList.append("Clear")
                pass
            elif change == "CM":
                prob = prob * 0.004
                activityToday = "Mostly Cloudy"
                activityList.append("Mostly Cloudy")
            elif change == "CP":
                prob = prob * 0.00
                activityToday = "Partly Cloudy"
                activityList.append("Partly Cloudy")
            elif change == "CS":
                prob = prob * 0.087
                activityToday = "Scattered"
                activityList.append("Scattered")            
            else:
                prob = prob * 0.013
                activityToday = "Overcast"
                activityList.append("Overcast")
        elif activityToday == "Mostly Cloudy":
            change = np.random.choice(transitionName[1],replace=True,p=transitionMatrix[1])
            if change == "MM":
                prob = prob * 0.820
                activityList.append("Mostly Cloudy")
                pass
            elif change == "MC":
                prob = prob * 0.003
                activityToday = "Clear"
                activityList.append("Clear")
            elif change == "MP":
                prob = prob * 0.049
                activityToday = "Partly Cloudy"
                activityList.append("Partly Cloudy")                
            elif change == "MS":
                prob = prob * 0.013
                activityToday = "Scattered"
                activityList.append("Scattered")                
            else:
                prob = prob * 0.115
                activityToday = "Overcast"
                activityList.append("Overcast")
        elif activityToday == "Partly Cloudy":
            change = np.random.choice(transitionName[2],replace=True,p=transitionMatrix[2])
            if change == "PP":
                prob = prob * 0.739
                activityList.append("Partly Cloudy")
                pass
            elif change == "PC":
                prob = prob * 0.001
                activityToday = "Clear"
                activityList.append("Clear")
            elif change == "PM":
                prob = prob * 0.244
                activityToday = "Partly Cloudy"
                activityList.append("Partly Cloudy")                
            elif change == "PS":
                prob = prob * 0.001
                activityToday = "Scattered"
                activityList.append("Scattered")                
            else:
                prob = prob * 0.015
                activityToday = "Overcast"
                activityList.append("Overcast")                            
        elif activityToday == "Scattered":
            change = np.random.choice(transitionName[3],replace=True,p=transitionMatrix[3])
            if change == "SS":
                prob = prob * 0.752
                activityList.append("Scattered")
                pass
            elif change == "SC":
                prob = prob * 0.12
                activityToday = "Clear"
                activityList.append("Clear")
            elif change == "SM":
                prob = prob * 0.021
                activityToday = "Mostly Cloudy"
                activityList.append("Mostly Cloudy")                
            elif change == "SP":
                prob = prob * 0.00
                activityToday = "Partly Cloudy"
                activityList.append("Partly Cloudy")                
            else:
                prob = prob * 0.107
                activityToday = "Overcast"
                activityList.append("Overcast")                         
        elif activityToday == "Overcast":
            change = np.random.choice(transitionName[4],replace=True,p=transitionMatrix[4])
            if change == "OO":
                prob = prob * 0.709
                activityList.append("Scattered")
                pass
            elif change == "OC":
                prob = prob * 0.017
                activityToday = "Clear"
                activityList.append("Clear")
            elif change == "SM":
                prob = prob * 0.141
                activityToday = "Mostly Cloudy"
                activityList.append("Mostly Cloudy")                
            elif change == "SP":
                prob = prob * 0.002
                activityToday = "Partly Cloudy"
                activityList.append("Partly Cloudy")                
            else:
                prob = prob * 0.131
                activityToday = "Scattered"
                activityList.append("Scattered")   
        i += 1
#        prob_prce.append(prob)
#    return activityList,prob_prce
    return activityList
# possible state sequrence:
a=activity_forecast(days,star)
a=pd.DataFrame(a)
a.columns=["weather"]
a.weather.value_counts()

#############

# To save every activityList
list_activity = []
#star="Clear"
cycling=5
for iterations in range(0,cycling):
        list_activity.append(activity_forecast(days,star))



C_prob=[]
for i in range(1,days+1):
    count_C = 0
    percent = 0
    for smaller_list in list_activity:
        if(smaller_list[i] == "Clear"):
            count_C += 1
            percent=(count_C/days) 
    C_prob.append(percent)
    print("The probability of Clear state ending at %d day with starting state:%s = %.3f" % (i+1,star,percent))

M_prob=[]
for i in range(1,days+1):
    count_M =0
    percent=0
    for smaller_list in list_activity:
        if(smaller_list[i] == "Mostly Cloudy"):
            count_M += 1
            percent=(count_M/days)
    M_prob.append(percent)
    print("The probability of Mostly cloudy ending at %d day with state:'%s' Percentage= %.3f" % (i+1, star, percent))


P_prob=[]
for i in range(1,days+1):
    count_p =0
    percent=0
    for smaller_list in list_activity:
        if(smaller_list[i] == "Partly Cloudy"):
            count_p += 1
            percent=(count_p/days) 
    P_prob.append(percent)
    print("The probability of Mostly cloudy ending at %d day with state:'%s' Percentage= %.3f" % (i+1, star, percent))


S_prob=[]
for i in range(1,days+1):
    count_s =0
    percent=0
    for smaller_list in list_activity:
        if(smaller_list[i] == "Scattered"):
            count_s += 1
            percent=(count_s/days) 
    S_prob.append(percent)
    print("The probability of Mostly cloudy ending at %d day with state:'%s' Percentage= %.3f" % (i+1, star, percent))

O_prob=[]
for i in range(1,days+1):
    count_o =0
    percent=0
    for smaller_list in list_activity:
        if(smaller_list[i] == "Overcast"):
            count_o += 1
            percent=(count_o/days) 
    O_prob.append(percent)
    print("The probability of Mostly cloudy ending at %d day with state:'%s' Percentage= %.3f" % (i+1, star, percent))

data_tuples = list(zip(C_prob,M_prob,P_prob,S_prob,O_prob))
run_result=pd.DataFrame(data_tuples, columns=['Clear','MC','PC','Scattered','Overcast'])
pred=run_result.idxmax(axis=1)

tuples = list(zip(real_data,pred))
run_result=pd.DataFrame(tuples, columns=['Label','pred'])
count_pred=run_result.pred.value_counts()
count_real=run_result.Label.value_counts()


###save file
"""
run_result.to_csv("MM_result_8760.csv")
"""


####
acc=0
for i in range(len(real_data)):
    if pred[i]==real_data[i]:
        acc+=1
accuracy=acc/len(real_data)

print(accuracy)