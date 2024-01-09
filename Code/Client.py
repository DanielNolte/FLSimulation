# -*- coding: utf-8 -*-
"""
@author: Daniel Nolte
Adapted from DanielNolte/FederatedDeepRegressionForests: https://github.com/DanielNolte/FederatedDeepRegressionForests
"""
import torch
from sklearn.ensemble import RandomForestRegressor
class Client:
    #Initialize client with given parameters
    def __init__(self,data,valData,model,opt):
        self.clientIteration = 0
        self.data = data
        opt.lr = opt.clientlr
        self.model = model(opt)
        #Epochs controlled by Training rounds
        self.model.opt.epochs=1
        self.val = valData
        
     
    #method for client level evaluation on given batch of data      
    def client_eval(self,batch):   
        testResults = self.model.evaluate_model(batch['X'],batch['Y'])
        #return evaluation results
        return testResults
            
    def reduce_lr(self):
        # self.model.opt.lr = self.model.opt.lr*0.25
        for g in self.model.optim.param_groups:
            print('LearningRate: '+str(g['lr']*0.1))
            g['lr'] =  g['lr']*0.1
    #Given the serverModelWeights, Initialize model with weights, and 
    #train on the batched data
    def client_update(self,serverModelParams):
        clientWeight = 0
        #update client model with provided server weights
        self.model.update_model(serverModelParams)
        self.model,_ = self.model.train1(self.data,self.val,clientTrain=True)
        #Set client update to be the model weights after training
        clientUpdate = self.model.get_model_weights()
        torch.cuda.empty_cache()
        #evalute on the current batch of data
        trainResults = self.model.evaluate_model(self.data) 
        valResults = self.model.evaluate_model(self.val) 

        #update client weight with number of samples from current batch
        clientWeight = len( self.data)
        #increase clientIteration by 1
        self.clientIteration+=1
        #Return client update,evalResults and the clientWeight
        return clientUpdate,trainResults,valResults,clientWeight
    def train_RF(self,params):
        reg = RandomForestRegressor( **params,n_estimators=100)
        reg.fit(self.data.X,self.data.Y)
        return reg
        
        
    
    
        
    