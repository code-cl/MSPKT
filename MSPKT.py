# -*- coding: utf-8 -*-
"""
Created on 2020
@author: chenlu
Multiple source partial knowledge transfer for regression considering the transferability
"""
import numpy as np
import skfuzzy as fuzz

#Determine the clustering centers by FCM and compute variance of each feature 
def fcm(data, nRules):
    '''
    input:
      data   : n_Samples * n_Features
      nRules : number of TSK rules
    output: 
      centers : clustering centers
      delta   : variance of each feature
    '''
    n_samples, n_features = data.shape
    centers, mem, _, _, _, _, _ = fuzz.cmeans(
            data.T, nRules, 2.0, error=1e-5, maxiter=200)
    
    #compute the variance of each feature
    delta = np.zeros([nRules, n_features])
    for i in range(nRules):
        d = (data - centers[i, :]) ** 2
        delta[i, :] = np.sum(d * mem[i, :].reshape(-1, 1),
             axis=0) / np.sum(mem[i, :])
    return centers, delta

#Compute the membership degree using Gaussian membership function
def get_x_p(data, centers, delta):
       
    '''
    input:
        data    : n_Samples * n_Features
        centers : clustering centers, nRules * n_Features
        delta   : variance of each feature, nRules * n_Features
    output:
        data_fs : the X_h in Eq.(7), n_Samples * [nRules * (n_Features+1)]
        mu_a    : the membership degree of each data
    '''
    nRules = centers.shape[0]
    n_samples = data.shape[0]
    
    mu_a = np.zeros([n_samples, nRules])
    for i in range(nRules):
        tmp_k = 0 - np.sum((data - centers[i, :]) ** 2 /
                           delta[i, :], axis=1)
        mu_a[:, i] = np.exp(tmp_k)  # exp max 709
        
    # norm
    mu_a = mu_a / np.sum(mu_a, axis=1, keepdims=True)
    
    # print(np.count_nonzero(mu_a!=mu_a))
    data_1 = np.concatenate((data, np.ones([n_samples, 1])), axis=1)
    
    zt = []
    for i in range(nRules):
        zt.append(data_1 * mu_a[:, i].reshape(-1, 1))
    data_fs = np.concatenate(zt, axis=1)
    data_fs = np.where(data_fs != data_fs, 1e-5, data_fs)
    
    return data_fs, mu_a

def TSK(X_train,Y_train,X_test,Y_test,centers, delta,C=0.0001):
    
    n, d = X_train.shape
    
    # compute x_p: as euqation(3) in the paper
    X_p_s   , _  = get_x_p(X_train, centers, delta) # X_p_s as euqation(3)  
    X_p_test, _  = get_x_p(X_test,   centers, delta)
    
    X_p_s1 = np.dot(X_p_s.T, X_p_s)
    Ps = np.linalg.pinv(X_p_s1 + C * np.eye(X_p_s1.shape[0])).dot(X_p_s.T).dot(Y_train)
    
    Y_predict = X_p_test.dot(Ps) 
    
    return Y_predict,Ps

#The similarity measurement, i.e. Eq.(12)
def Similarity (CCs,X_target,Y_target,X_center,delta):   
    #计算隶属度误差
    nSources = CCs.shape[0]
    nRules = X_center.shape[0]
    nt , D = X_target.shape
    
    _, Membership = get_x_p(X_target, X_center, delta)
    
    SM = np.zeros((nSources,nRules))
    x = np.zeros((X_target.shape[0],1))
    Xt = np.c_[X_target,x]
    for i in range(nSources):
        Ps =  CCs[i,:].reshape((nRules,D+1))
        Ytemp = np.dot(Xt,Ps.T)
        Errtemp = np.abs(Ytemp - Y_target)
        SM[i,:] = np.sum(np.multiply(Membership,Errtemp),axis=0)/nt
        
    return SM

#The reliability measurement, i.e. Eq.(13)
def Reliability (CCs,nRules):   
    #计算欧式距离
    nSources = CCs.shape[0]
    D = int(CCs.shape[1]/nRules-1)
    RM = np.zeros((nSources,nRules))
    Htemp = np.zeros(((nSources,nSources,((D+1)*nRules))))
    Ftemp = np.zeros(((nSources,nSources,nRules)))
    for i in range(nSources):
        for j in range(nSources):
            Htemp[i,j,:] = CCs[i,:] - CCs[j,:]
    Htemp_pf = np.multiply(Htemp,Htemp)
    for i in range(nSources):
        for j in range(nSources):
            for k in range(nRules):
                Ftemp[i,j,k] = np.sqrt(np.sum(Htemp_pf[i,j,(k*(D+1)):((k+1)*(D+1)-1)]))
        RM[i,:] = Ftemp[i,:,:].sum(axis=0)
    
    return RM

#Compute the transferability combining the similarity and reliability
#and the weight coefficients of source TSK rules
def RulesWeight (CCs,SM,RM,lam=5,alpha=0.5):   
   
    nSources,nRules = SM.shape
    D = int(CCs.shape[1] / nRules-1)
    
    SM_norm = np.zeros((nSources,nRules))
    RM_norm = np.zeros((nSources,nRules))
    for i in range(nRules):
        SM_norm[:,i] = SM[:,i]/np.max(SM[:,i])
        RM_norm[:,i] = RM[:,i]/np.max(RM[:,i])
    
    #compute the transferability, i.e. Eq.(15)
    TM = (1-alpha)*RM_norm + alpha*SM_norm 
    
    #the transferability function, i.e. Eq.(16)
    DOT = np.exp(-lam*TM) 
    
    #calculate the weight coefficient by Eq.(17)
    weight = np.zeros((nSources,nRules))
    for i in range(nRules):
        weight[:,i] = DOT[:,i] / np.sum(DOT[:,i])
    
    #obtain the weighted rules by Eq.(18)
    W_weight = np.zeros((nRules*(D+1),1))
    Wtemp = np.zeros((nSources,(D+1)*nRules))
    for i in range(nSources):
        for k in range(nRules):
            Wtemp[i,k*(D+1):(k+1)*(D+1)] = weight[i,k]*CCs[i,k*(D+1):(k+1)*(D+1)]
    W_weight[:,0] = Wtemp.sum(axis=0)
    
    return weight, W_weight

#ResTL: append a constant bias to each TSK rule
def FuzzyResidual (W_weight,X_target,Y_target,X_test,X_center,delta): 

    nRules,D = X_center.shape
    
    X_p_target, Member = get_x_p(X_target, X_center, delta)
    X_p_test, _ = get_x_p(X_test, X_center, delta)
    
    #compute the bias by Eq.(19)
    err_tar = Y_target - X_p_target.dot(W_weight) 
    residual = np.dot(Member.T,err_tar) / (Member.T.sum(axis = 1)).reshape((nRules,1))

    #append the bias to each TSK rule
    P_weight = np.zeros((nRules*(D + 1),1))
    P_weight[:,0] = W_weight[:,0]
    for j in range(nRules):     
        P_weight[D + (D + 1)*j,0] = W_weight[D + (D + 1)*j,0] + residual[j,0]
    
    Y_predict = X_p_test.dot(P_weight) 
    
    return P_weight,Y_predict

#Multiple source partial knowledge transfer
def Model (CCs,x_target,y_target,x_test,X_center,delta,lam=5,alpha=0.4): 
    
    '''
    input
      CCs : nSources*(nRules*(D+1))
      x_target : Ntr*D
      y_target : Ntr*1
      x_test   : Nte*D
      X_center : nRules*D
      Sigma    : nRules*1
    output
      Pt : (nRules*(D + 1))*1
      y_predict: Nte*1
    '''
    nRules=X_center.shape[0]
    
    RM = Reliability(CCs,nRules)
    SM = Similarity(CCs,x_target,y_target,X_center,delta)
    
    rule_weight,W_weight =  RulesWeight(CCs,SM,RM,lam,alpha)
    
    Pt,y_predict = FuzzyResidual(W_weight,x_target,y_target,x_test,X_center,delta)
        
    return rule_weight,Pt,y_predict
    
    
    
    

