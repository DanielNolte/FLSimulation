# -*- coding: utf-8 -*-
"""
@author: Daniel Nolte
Adapted from DanielNolte/FederatedDeepRegressionForests: https://github.com/DanielNolte/FederatedDeepRegressionForests
"""
from torch.utils.data import Dataset
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch


#%%
class NCI60FedData:
    def __init__(self,cell,opt):
        self.opt=opt
        name = '/NCI60_2_'+cell+'.pickle'
        path = opt.data_path+'/'+opt.dataset
        with open(path+name, 'rb') as handle:
            Xrest = pickle.load(handle)
            Yrest = pickle.load(handle)
        
        print(Yrest.size)
        print("Data Loaded")
        print()
        self.X = Xrest
        self.Y = Yrest 

   
    def split_data(self,numClients,fracTestData,seed, numSamples,beta=None):
        XYtest = {}
        XYtrain = {}
        centralizedData = {}
        # Set data
        drugCellX = self.X
        drugCellY = self.Y
        drugCellX = drugCellX.reset_index(drop=True)
        drugCellY = drugCellY.reset_index(drop=True)
        
        #Split data
        X_train, X_test, y_train, y_test = train_test_split(drugCellX, drugCellY, train_size=numSamples,random_state=seed)
        X_Val, X_test, y_Val, y_test = train_test_split(X_test, y_test, train_size=2000,random_state=seed)
        
        
        np.random.seed(seed)
        if beta is not None:           
            bins=[]
            for bin1 in X_train.groupby(pd.qcut(y_train.rank(method='first').values.ravel(),numClients,duplicates='drop')):
                bins.append(bin1)
            if beta == 0:
                splitData = [[] for _ in range(numClients)]
                splitTarget = [[] for _ in range(numClients)]
                for i in range(numClients):
                    splitData[i] = bins[i]
                    splitTarget[i]= y_train.loc[bins[i].index]
                    splitData[i] = {'X':splitData[i].to_numpy(),'Y':splitTarget[i].to_numpy()}
                    splitData[i] = FedNCI60DataSet(splitData[i])
            else:
                trigger= True
                
                while trigger:
                    trigger= False
                    splitData = [[] for _ in range(numClients)]
                    splitTarget = [[] for _ in range(numClients)]
                    
                    dataPerClient= np.random.dirichlet([beta for x in range(numClients)],numClients)
                    
                    for i in range(numClients):
                        BinPer = dataPerClient[i]
                        clientSamples = BinPer*len(bins[i][1])
                        clientSamples=clientSamples.astype(int)
                        idx=0
                        for client,size in enumerate(clientSamples):
                            temp = bins[i][1][idx:idx+size]
                            splitData[client].append(temp)
                            splitTarget[client].append(y_train.loc[temp.index])
                            idx +=size
                            
                    for i in range(numClients):
                        splitData[i] = pd.concat(splitData[i])
                        splitTarget[i]= pd.concat(splitTarget[i])
                        if len(splitData[i])<0.05*numSamples:
                            trigger= True
                        splitData[i] = {'X':splitData[i].to_numpy(),'Y':splitTarget[i].to_numpy()}
                        splitData[i] = FedNCI60DataSet(splitData[i])
        else:
            dataPerClient = round(len(y_train)/numClients)
            splitData = [0] * numClients
            for i in range(numClients):
                clientX = X_train[i*dataPerClient:(i+1)*dataPerClient]
                clientY = y_train[i*dataPerClient:(i+1)*dataPerClient]
                splitData[i] = {'X':clientX.to_numpy(),'Y':clientY.to_numpy()}
                splitData[i] = FedNCI60DataSet(splitData[i])
        print("Data splitting is done!")
        XYtrain = splitData
        centralizedData = {'X':X_train.to_numpy(),'Y':y_train.to_numpy()}
        XYval = {'X':X_Val.to_numpy(),'Y':y_Val.to_numpy()}
        XYtest = {'X':X_test.to_numpy(),'Y':y_test.to_numpy()}
        
        
        
        print('Train n'+str(len(X_train)))
        print('Val n'+str(len(X_Val)))
        print('Test n'+str(len(X_test)))
        print('Cent n'+str(len(centralizedData['X'])))
    
        centralizedData = FedNCI60DataSet(centralizedData)
        XYval = FedNCI60DataSet(XYval)
        XYtest = FedNCI60DataSet(XYtest)      
        
        del X_test,y_test,splitData,self.Y,self.X,drugCellX,drugCellY,X_train,y_train
        return XYtrain,XYtest,centralizedData,XYval
#%%

class FedNCI60DataSet(Dataset):
    def __init__(self, data):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.X = data['X']
        self.Y = data['Y']

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
      
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(self.Y[idx]).float()
        sample = {'data': x, 'target': y}

        return sample