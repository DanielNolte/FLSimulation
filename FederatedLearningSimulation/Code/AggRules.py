# -*- coding: utf-8 -*-
"""
@author: Daniel Nolte
Adapted from DanielNolte/FederatedDeepRegressionForests: https://github.com/DanielNolte/FederatedDeepRegressionForests
"""
import numpy as np
    
def fed_avgDRF(clientUpdates,clientAccs,clientWeights):
    #Calc total weight
    totalWeight = np.sum(clientWeights)
    aggregatedUpdate = {}
    #For each client, for each layer in the model, take the weighted average
    for ii, (name, value) in enumerate(clientUpdates[0].items()):
                aggregatedUpdate[name] = value*(clientWeights[0]/totalWeight)
    for i in range(len(clientWeights)-1):
            i+=1
            for ii, (name, value) in enumerate(clientUpdates[i].items()):
                aggregatedUpdate[name] += value*(clientWeights[i]/totalWeight)
    #Calculate weighted losses
    weightedAccs = np.array(clientAccs)*(clientWeights/totalWeight)
    aggregatedAcc = np.sum(weightedAccs)
    del weightedAccs, totalWeight
    print(aggregatedAcc)
    #Return data weighted average of updates and losses
    return aggregatedUpdate,aggregatedAcc
    

#FedAvg: Data Weighted Average of all updates
def fed_avg(clientUpdates,clientAccs,clientWeights):
    #Calc total weight
    totalWeight = np.sum(clientWeights)
    aggregatedUpdate = [0]*len(clientUpdates[0])
    #For each client, for each layer in the model, take the weighted average
    for i in range(len(clientWeights)):
            for ii in range(len(clientUpdates[i])):
                aggregatedUpdate[ii] += clientUpdates[i][ii]*(clientWeights[i]/totalWeight)
    #Calculate weighted losses
    print(clientAccs)
    weightedAccs = np.array(clientAccs)*clientWeights/totalWeight
    aggregatedAcc = np.sum(weightedAccs)
    print(aggregatedAcc)
    #Return data weighted average of updates and losses
    return aggregatedUpdate,aggregatedAcc