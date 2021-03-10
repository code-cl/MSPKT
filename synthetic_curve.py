import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import MSPKT

plt.figure()
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16

#Target Data
target = pd.read_csv('./Target/target.csv',header=None).values
x_target = target[:,0:1]
y_target = target[:,1:2]
plt.scatter(x_target,y_target,s=40,marker='o',c='b',label='Target Data')

#Source Data
Source_Path = './Source'
source_dirs = os.listdir(Source_Path)
nSources = len(source_dirs)

d = 1
xtr = np.empty((nSources),dtype=object)
ytr = np.empty((nSources),dtype=object)
Xtr = np.zeros((1,d))

color=['r','tomato','C1','C5','C6','C7']
for i in range (nSources):
    data = pd.read_csv(Source_Path + '/' + source_dirs[i],header=None).values
    xtr[i] = data[:,0:d]
    ytr[i] = data[:,d].reshape((data.shape[0],1))
    plt.plot(xtr[i],ytr[i],c = color[i], linewidth = 2.5, linestyle = '-',label = 'Source'+np.str(i+1))
    Xtr = np.r_[Xtr,xtr[i]]
Xtr = np.delete(Xtr,0,0)

#Test data
xmin = 0
xmax = 8
xtest = np.linspace(xmin,xmax,100).reshape((100,1))

#Represent multiple source domains as the collection of partial knowledge
X = np.r_[Xtr,x_target]
nRules = 8
centers,delta = MSPKT.fcm(X,nRules)
CCs = np.zeros((nSources,nRules*(d+1)))
I = np.ones((100,1))
color1 = ['C9','C7','C3','C5','C6','C7']
for i in range(nSources):
    _,Ps = MSPKT.TSK(xtr[i],ytr[i],xtr[i],ytr[i],centers,delta,0.0001)
    CCs[i,:] = Ps.T
    
    #show the output function of each TSK rule
    for j in range (nRules):
        x = np.linspace(centers[j]-0.5,centers[j]+0.5,100)
        X = np.c_[x,I]
        y = np.dot(X,Ps[(2*j):(2*j + 2), 0].reshape((2,1)))
        plt.plot(x,y,c = color1[i], linewidth = 2.0, linestyle = '--')

#Construct the target model by MSPKT
lamda = 8
alpha = 0.3
rule_weight,Pt,ypre = MSPKT.Model(CCs,x_target,y_target,xtest,centers,delta,lamda,alpha)
plt.plot(xtest,ypre,c = 'navy', linewidth = 2.5, linestyle = '-',label = 'MSPKT')

#Save the rule weight
df = pd.DataFrame(rule_weight[:,np.argsort(centers[:,0])])
df.to_csv('./Result/rule_weight.csv',header = 0,index = 0)

#other plot setting
plt.xlabel('x')
plt.ylabel('y')
ymin = -1.5
ymax = 2.5
a = np.linspace(ymin,ymax,9)
b = np.linspace(xmin,xmax,9)
plt.xticks(b)
plt.yticks(a)
plt.xlim(xmin - 0.2,xmax + 0.2)
plt.ylim(ymin,ymax)
plt.legend(loc = 'upper right')
plt.show()
plt.savefig('synthetic_curve.png',dpi = 600,bbox_inches = 'tight')