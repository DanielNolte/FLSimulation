# -*- coding: utf-8 -*-
"""
@author: Daniel Nolte
Adapted from DanielNolte/FederatedDeepRegressionForests: https://github.com/DanielNolte/FederatedDeepRegressionForests
"""
import utilsNCI as utils
from Client import Client
import numpy as np
import torch
from  AggRules import fed_avgDRF
from sklearn.ensemble import RandomForestRegressor
from FedDRF import FedDRF
from resnetNCI60 import ANN
from utilsNCI import NRMSE
import time
import pandas as pd

class Server:
    def __init__(self,numClients,trainingRounds,fracTestData,opt,cell):
        self.n = numClients
        self.opt = opt
        self.clientValLossHist = []
        self.clientTrainLossHist = []
        self.serLossHist = []
        # Set GPU
        if self.opt.cuda:
            torch.cuda.set_device(self.opt.gpuid)
        
        #Select aggRule to use
        self.aggRule = fed_avgDRF
        
        from NCI60FedData import NCI60FedData
        fedDatset = NCI60FedData(cell,opt)
        self.train,self.test,self.central,self.val= fedDatset.split_data(numClients,fracTestData,
                                    self.opt.seed, opt.numSamples,opt.beta)
        del fedDatset
              
        #If evalCentralized = True, then train and test the centralized model
        #for baseline performance
        if self.opt.trainCent:
            if self.opt.model == 'ANNFull':
                self.centralizedModel = ANN(self.opt)
                t = time.time()
                self.centralizedModel,lossHistory = self.centralizedModel.train1(self.central,self.val,False)
                elapsed = time.time() - t
                utils.save_model(self.centralizedModel, self.opt,'Central')
                utils.save_losses_time(lossHistory,elapsed,self.opt,'Central')
            elif self.opt.model == 'FedDRF':
                self.centralizedModel = FedDRF(self.opt)
                t = time.time()
                self.centralizedModel,lossHistory = self.centralizedModel.train1(self.central,self.val,False)
                elapsed = time.time() - t
                utils.save_model(self.centralizedModel, self.opt,'Central')
                utils.save_losses_time(lossHistory,elapsed,self.opt,'Central')
            elif self.opt.model == 'FedRF':
                params = {'random_state': opt.seed,
                                  'n_jobs': -1,
                                  'min_samples_leaf':5,
                                  'max_features':0.5}
                self.centralizedModel = RandomForestRegressor(n_estimators=500,**params)
                self.centralizedModel.fit(self.central.X,self.central.Y)
                utils.save_model(self.centralizedModel, self.opt,'Central')
                
            
            

        print(self.opt.cuda)

        if self.opt.evalCent:
            save_dir = utils.get_save_dir(self.opt,'Central')
            if self.opt.model == 'FedRF':
                self.centralizedModel = torch.load(save_dir,map_location=torch.device('cpu')) 
                FedTrainResults,FedValResults,FedTestResults = self.fed_eval_rf(self.centralizedModel)                
            else:
                if self.opt.cuda:
                    self.centralizedModel = torch.load(save_dir)
                else:
                    self.centralizedModel = torch.load(save_dir,map_location=torch.device('cpu')) 
                    self.centralizedModel.opt.cuda = 0
                FedTrainResults,FedValResults,FedTestResults = self.fed_eval(self.centralizedModel)
            CentInitResults = pd.DataFrame(index=['MSE','PCC'])
            CentInitResults['CentTrain'+str(self.opt.seed)] = FedTrainResults
            CentInitResults['CentVal'+str(self.opt.seed)] = FedValResults
            CentInitResults['CentTest'+str(self.opt.seed)] = FedTestResults
            CentInitResults=CentInitResults.T
            print(CentInitResults)
            


        if (self.opt.model == 'ANNFull')& (self.opt.trainFed):
            self.model = ANN(self.opt)
            self.model.opt = self.opt
            self.clients = [Client(self.train[client],self.val,ANN,self.opt) 
                        for client in range(numClients)]
        elif (self.opt.model == 'FedDRF') & (self.opt.trainFed):
            self.model = FedDRF(self.opt)
            self.model.opt = self.opt
            self.clients = [Client(self.train[client],self.val,FedDRF,self.opt) 
                        for client in range(numClients)]
        else:
            self.model = FedDRF(self.opt)
            self.model.opt = self.opt
            self.clients = [Client(self.train[client],self.val,FedDRF,self.opt) 
                        for client in range(numClients)]


        
        
    def server_agg(self,clientUpdates,clientAccs,clientWeights):
        ##Use agg rule, clientUpdates, and clientWeights for
        ##aggregated update
        update,acc = self.aggRule(clientUpdates,clientAccs,clientWeights)
        return update,acc  
        
    def fed_eval(self,model):
        FedTrainResults = model.evaluate_model(self.central)
        FedTestResultsMSE,FedPCC =FedTrainResults
        print('Train')
        print(FedTestResultsMSE)
        print(FedPCC)
        FedValResults = model.evaluate_model(self.val)
        FedTestResultsMSE,FedPCC = FedValResults
        print('Val')
        print(FedTestResultsMSE)
        print(FedPCC)
        FedTestResults = model.evaluate_model(self.test)
        FedTestResultsMSE,FedPCC = FedTestResults
        print('Test')
        print(FedTestResultsMSE)
        print(FedPCC)
        return FedTrainResults,FedValResults,FedTestResults
  
    def fed_eval_rf(self,model):
       FedTrainResults = NRMSE(self.central.Y,model.predict(self.central.X))
       FedTestResultsMSE,FedPCC =FedTrainResults
       print('Train')
       print(FedTestResultsMSE)
       print(FedPCC)
       FedValResults = NRMSE(self.val.Y,model.predict(self.val.X))
       FedTestResultsMSE,FedPCC = FedValResults
       print('Val')
       print(FedTestResultsMSE)
       print(FedPCC)
       FedTestResults = NRMSE(self.test.Y,model.predict(self.test.X))
       FedTestResultsMSE,FedPCC = FedTestResults
       print('Test')
       print(FedTestResultsMSE)
       print(FedPCC)
       return FedTrainResults,FedValResults,FedTestResults   
    def run_one_round(self):
        #Get current server weights
        serverUpdate = self.model.get_model_weights()
        clientsForRound = self.clients
        #Call all selected clients for update using serverWeight 
        #and zip results
        
        clientUpdates,clientTrainResults,clientValResults,clientWeights = zip(
            *[client.client_update(serverUpdate) 
              for client in clientsForRound])
      
        self.clientTrainLossHist.append(clientTrainResults)
        self.clientValLossHist.append(clientValResults)

        update,acc = self.server_agg(clientUpdates,pd.DataFrame(clientValResults)[:][1],clientWeights)
        del clientUpdates,clientValResults,clientTrainResults,clientWeights,serverUpdate
        self.model = self.model.update_model(update)

        #If test data is avaliable, run server eval
        if self.opt.eval:
                print('Server Val')
                NRMSE,R2 = self.model.evaluate_model(self.val)
                print(NRMSE)
        self.serLossHist.append([NRMSE,R2])
            #return the update, loss, and test results    
        return update,acc,NRMSE,self.model,self.clientValLossHist,self.clientTrainLossHist,self.serLossHist
    def train_FedRF(self,params):
        reg = RandomForestRegressor(n_estimators=500,**params)
        for i,client in enumerate(self.clients):
            client_reg = client.train_RF(params)
            if i==0:
                reg.estimators_ = client_reg.estimators_
            else:
                reg.estimators_[100*(i):100*(i+1)] = client_reg.estimators_
            
        reg.n_outputs_=client_reg.n_outputs_
        self.RandomForest = reg
        FedTrainResults,FedValResults,FedTestResults    = self.fed_eval_rf(reg)
        utils.save_model(reg, self.opt,'bestFed'+str(self.n))
        return FedTrainResults,FedValResults,FedTestResults   

        
