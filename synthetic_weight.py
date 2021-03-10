# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def Histogram_Plot(data):
    
    nTrains = 8
    source1 = data[0,:]
    source2 = data[1,:]
    source3 = data[2,:]
    
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.weight'] = 'normal'
    matplotlib.rcParams['font.size'] = 20
    plt.figure(figsize=(13,5))#figsize=(5,6)
    
    b=np.ones(nTrains)
    for i in range(b.shape[0]): 
        b[i] = i
    
    bar_width = 0.5
    istart = 1
    isource1 = istart + 4*bar_width*b
    isource2 = istart + 1*bar_width+4*bar_width*b
    isource3 = istart + 2*bar_width+4*bar_width*b
    
    color = ['r','tomato','C1','C4','C5','C6']
    plt.bar(isource1, source1, width=bar_width,fc=color[0],edgecolor = 'k', label = 'Source1')
    plt.bar(isource2, source2, width=bar_width,fc=color[1],edgecolor = 'k', label = 'Source2')
    plt.bar(isource3, source3, width=bar_width,fc=color[2],edgecolor = 'k', label = 'Source3')

    plt.xlabel('Rules')
    plt.ylabel('Weight')
    plt.legend(loc = 'upper right')
    plt.xticks(isource2,['1','2','3','4','5','6','7','8'])
    
    ymin = 0
    ymax = 1
    plt.ylim(ymin,ymax)
    a = np.linspace(ymin,ymax,6)
    plt.yticks(a)

    plt.show()
    plt.savefig('synthetic_weight.png',dpi=600,bbox_inches='tight')

rule_weight = pd.read_csv('./Result/rule_weight.csv', header=None).values
Histogram_Plot(rule_weight)