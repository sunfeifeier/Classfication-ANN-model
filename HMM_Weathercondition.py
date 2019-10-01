# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 15:51:53 2019
https://medium.com/@kangeugine/hidden-markov-model-7681c22f5b9
http://www.cs.jhu.edu/~jason/papers/#eisner-2002-tnlp
https://github.com/aldengolab/hidden-markov-model
viterbi

@author: lidon
"""
from pandas import read_csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



states = ("Clear" ,"Mostly Cloudy","Partly Cloudy","Scattered Clouds","Overcast")
start_p = {"Clear":0.288, "Mostly Cloudy":0.251,"Partly Cloudy":0.217,
           "Scattered Clouds":0.195,"Overcast":0.049}

trans_p = {"Clear":{"Clear":0.8955, "Mostly Cloudy":0.0036,"Partly Cloudy": 0.0869,"Scattered Clouds": 0.0139, "Overcast":0.0001},
             "Mostly Cloudy":{"Clear":0.0029, "Mostly Cloudy":0.8193,"Partly Cloudy":0.0128, "Scattered Clouds":0.1162, "Overcast":0.0488},
             "Partly Cloudy":{"Clear":0.1208, "Mostly Cloudy":0.0213,"Partly Cloudy":0.7501, "Scattered Clouds":0.1074, "Overcast":0.0004},
             "Scattered Clouds":{"Clear":0.0166, "Mostly Cloudy":0.1420,"Partly Cloudy": 0.1320, "Scattered Clouds":0.7074, "Overcast":0.0020},
             "Overcast":{"Clear":0.0007, "Mostly Cloudy":0.2446,"Partly Cloudy":0.0015, "Scattered Clouds":0.0153, "Overcast":0.7379}}

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = V[t-1][states[0]]["prob"]*trans_p[states[0]][st]
            prev_st_selected = states[0]
            for prev_st in states[1:]:
                tr_prob = V[t-1][prev_st]["prob"]*trans_p[prev_st][st]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st                  
            max_prob = max_tr_prob * emit_p[st][obs[t]]
            V[t][st] = {"prob": max_prob, "prev": prev_st_selected}
    pro_line=[]
    for line in dptable(V):
        print (line)
        pro_line.append(line)
    opt = []
    # The highest probability
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            opt.append(st)
            previous = st
            break
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]
    print ('The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob)
    return opt, max_prob,pro_line

def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)



"""
for i in range(len(a)):
    if a[i] <= -20.2:
        c="q_1"
    elif a[i]>-20.2 and a[i]<=-9.4:
        c="q_2"
    elif a[i]>-9.4 and a[i]<=1.4:  
        c="q_3"
    elif a[i]>1.4 and a[i]<=12.2: 
        c="q_4"
    elif a[i]>12.2 and a[i]<=23: 
        c="q_5"
    elif a[i]>23 and a[i]<=33.8: 
        c="q_6"
    elif a[i]>33.8 and a[i]<=44.6: 
        c="q_7"   
    elif a[i]>44.6 and a[i]<=55.4: 
        c="q_8"
    elif a[i]>55.4 and a[i]<=66.2: 
        c="q_9"
    else: 
        c="q_10"
    discrete_dewp.append(c)

emit_p = {"Clear":{"q_1":0.0001,"q_2":0.0005,"q_3":0.0095,"q_4":0.0518,"q_5":0.1705,
            "q_6":0.323,"q_7":0.3152,"q_8":0.1058,"q_9":0.0226,"q_10":0.0011},                       
  "Mostly Cloudy":{"q_1":0,"q_2":0,"q_3":0.0015,"q_4":0.0187,"q_5":0.0954,
                 "q_6":0.2194,"q_7":0.2423,"q_8":0.1865,"q_9":0.1919,"q_10":0.0443},  
  "Partly Cloudy":{"q_1":0,"q_2":0.001,"q_3":0.0052,"q_4":0.0374,"q_5":0.1432,
               "q_6":0.2913,"q_7":0.2844,"q_8":0.1493,"q_9":0.0819,"q_10":0.0062}, 
 "Scattered Clouds":{"q_1":0,"q_2":0,"q_3":0.0022,"q_4":0.0172,"q_5":0.0859,
               "q_6":0.211,"q_7":0.2529,"q_8":0.2257,"q_9":0.1854,"q_10":0.0197},
 "Overcast":{"q_1":0,"q_2":0.0002,"q_3":0.0026,"q_4":0.0188,"q_5":0.0752,
            "q_6":0.2009,"q_7":0.2816,"q_8":0.2365,"q_9":0.1209,"q_10":0.0634}}

"""


emit_p = {"Clear":{"q_1":0.0005,"q_2":0.0406,"q_3":0.4046,"q_4":0.5007,"q_5":0.0537},                       
  "Mostly Cloudy":{"q_1":0.0001,"q_2":0.101,"q_3":0.2372,"q_4":0.3981,"q_5":0.2963},  
  "Partly Cloudy":{"q_1":0.0008,"q_2":0.0266,"q_3":0.3585,"q_4":0.4641,"q_5":0.1499}, 
 "Scattered Clouds":{"q_1":0.0001,"q_2":0.01,"q_3":0.1995,"q_4":0.3768,"q_5":0.2710},
 "Overcast":{"q_1":0.0002,"q_2":0.0114,"q_3":0.2138,"q_4":0.5138,"q_5":0.2608}}


step=5

obs=x.iloc[-step:,-2].tolist()
real_data=x.iloc[-step:,4]

aaa=viterbi(obs,
        states,
        start_p,
        trans_p,
        emit_p)






"""
emit_p = {"Clear":{"q_1":0.0001,"q_2":0.0005,"q_3":0.0095,"q_4":0.0518,"q_5":0.1705,
            "q_6":0.323,"q_7":0.3152,"q_8":0.1058,"q_9":0.0226,"q_10":0.0011},                       
  "Mostly Cloudy":{"q_1":0,"q_2":0,"q_3":0.0015,"q_4":0.0187,"q_5":0.0954,
                 "q_6":0.2194,"q_7":0.2423,"q_8":0.1865,"q_9":0.1919,"q_10":0.0443},  
  "Partly Cloudy":{"q_1":0,"q_2":0.001,"q_3":0.0052,"q_4":0.0374,"q_5":0.1432,
               "q_6":0.2913,"q_7":0.2844,"q_8":0.1493,"q_9":0.0819,"q_10":0.0062}, 
 "Scattered Clouds":{"q_1":0,"q_2":0,"q_3":0.0022,"q_4":0.0172,"q_5":0.0859,
               "q_6":0.211,"q_7":0.2529,"q_8":0.2257,"q_9":0.1854,"q_10":0.0197},
 "Overcast":{"q_1":0,"q_2":0.0002,"q_3":0.0026,"q_4":0.0188,"q_5":0.0752,
            "q_6":0.2009,"q_7":0.2816,"q_8":0.2365,"q_9":0.1209,"q_10":0.0634}}

"""

####3state of air pressure
discrete_pressure=[]
a=x["Sea_Level_PressureIn_N"].tolist()
for i in range(len(a)):
    if a[i] <= 29.98:
        c="q_1"
    elif a[i]>29.304 and a[i]<=29.368:
        c="q_2"
    elif a[i]>29.368 and a[i]<=29.432:  
        c="q_3"
    elif a[i]>29.432 and a[i]<=28.496: 
        c="q_4"
    elif a[i]>29.496 and a[i]<=29.56: 
        c="q_5"
    elif a[i]>29.56 and a[i]<=29.624: 
        c="q_6"
    elif a[i]>29.624 and a[i]<=29.688: 
        c="q_7"   
    elif a[i]>29.688 and a[i]<=29.752: 
        c="q_8"
    elif a[i]>29.752 and a[i]<=29.816: 
        c="q_9"
    elif a[i]>29.816 and a[i]<=29.88: 
        c="q_10"
    elif a[i]>29.88 and a[i]<=29.944: 
        c="q_11"
    elif a[i]>29.944 and a[i]<=30.072: 
        c="q_12"
    elif a[i]>30.072 and a[i]<=30.136: 
        c="q_13"
    elif a[i]>30.136 and a[i]<=30.2: 
        c="q_14"
    elif a[i]>30.2 and a[i]<=30.264: 
        c="q_15"
    elif a[i]>30.264 and a[i]<=30.328: 
        c="q_16"
    elif a[i]>30.328 and a[i]<=30.392: 
        c="q_17"
    elif a[i]>30.392 and a[i]<=30.456: 
        c="q_18"
    else: 
        c="q19"
    discrete_pressure.append(c)

se = pd.Series(discrete_pressure)    
x['discrete_pressure'] = se.values
x.groupby(['Conditions_Name', 'discrete_pressure']).count()

##three obs_level
emit_p = {
        "Clear":       {"High":0.222,"Normal":0.562,"Low":0.216},
       "Mostly Cloudy":{"High":0.111,"Normal":0.666,"Low":0.223},
       "Partly Cloudy":{"High":0.116,"Normal":0.754,"Low":0.130},
       "Scattered cloud": {"High":0.178,"Normal":0.591,"Low":0.231},
       "Overcast":     {"High":0.100,"Normal":0.583,"Low":0.317}}

obs = ("Low","Normal","Normal")
viterbi(obs,
        states,
        start_p,
        trans_p,
        emit_p)



