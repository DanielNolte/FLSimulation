# -*- coding: utf-8 -*-
"""
@author: Daniel Nolte
Adapted from DanielNolte/FederatedDeepRegressionForests: https://github.com/DanielNolte/FederatedDeepRegressionForests
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  DataLoader  
import numpy as np 

class ANNShallowDRF(nn.Module):
    def __init__(self,opt):
        super(ANNShallowDRF, self).__init__()
        self.opt = opt
        self.dense1 = nn.Linear(672,128)
        self.densebn1 = nn.BatchNorm1d(128)
        self.dense2 = nn.Linear(128,opt.num_output)
        
        self.act = nn.ReLU(inplace=True)
        if self.opt.cuda:
            self.cuda()

    def forward(self,x):
        x = (self.act(self.densebn1(self.dense1(x))))
        x = self.dense2(x)
        reg_loss = 0
        return x,reg_loss
    

class ANN(nn.Module):
    def __init__(self,opt):
        super(ANN, self).__init__()
        self.opt = opt
        self.bestRMSE = 1000
        self.numBadEps=0
        self.dense1 = nn.Linear(672, 1000)
        self.densebn1 = nn.BatchNorm1d(1000)
        self.dense2 = nn.Linear(1000, 800)
        self.densebn2 = nn.BatchNorm1d(800)
        self.dense3 = nn.Linear(800, 500)
        self.densebn3 = nn.BatchNorm1d(500)
        self.dense4 = nn.Linear(500, 200)
        self.densebn4 = nn.BatchNorm1d(200)
        self.dense5 = nn.Linear(200, 100)
        self.densebn5 = nn.BatchNorm1d(100)
        self.dense6 = nn.Linear(100, 1)
        self.act = nn.ReLU()
        if self.opt.cuda:
            self.cuda()
        else:
            self
        params = [ p for p in self.parameters() if p.requires_grad]    
        self.optim  = torch.optim.Adam(params, lr = self.opt.lr,eps=1e-7)
        self.sche = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim ,
                                                            mode='min',
                                                            factor=0.1,
                                                            patience=self.opt.LRsched,
                                                            verbose=True,
                                                            min_lr=0.000001)        
    def forward(self,x):
        x1 = self.act(self.densebn1(self.dense1(x)))
        x1 = self.act(self.densebn2(self.dense2(x1)))
        x1 = self.act(self.densebn3(self.dense3(x1)))
        x1 = self.act(self.densebn4(self.dense4(x1)))
        x1 = self.act(self.densebn5(self.dense5(x1)))
        x1 = self.dense6(x1)
        reg_loss = 0
        return x1,reg_loss  
    def get_model_weights(self):
        param = self.state_dict()
        return param
    
    def update_model(self,update):
        self.load_state_dict(update)
        del update
        return self
    
    def train1(self,train1,val,clientTrain=False):
        """
        Args:
            model: the model to be trained
            optim: pytorch optimizer to be used
            db : prepared torch dataset object
            opt: command line input from the user
            exp_id: experiment id
        """
                
        params = {'batch_size': self.opt.batch_size,
          'shuffle': True,
          'num_workers': self.opt.num_threads}
        training_generator = DataLoader(train1,pin_memory=False, **params)
        losses = []
        for epoch in range(1, self.opt.epochs + 1):            
            # Train neural decision forest
            # set the model in the training mode
            self.train()
            print("Epoch %d : Update Neural Weights"%(epoch))
            for idx,sample in enumerate(training_generator):
                
                data = sample['data']
                target = sample['target']
                if len(target)<2:
                    continue
                target = target.view(len(target), -1)
                if self.opt.cuda:
                    with torch.no_grad():
                        # move to GPU
                        data = data.cuda()
                        target = target.cuda()                  
                
                # erase all computed gradient        
                self.optim.zero_grad()
             
                # forward pass to get prediction
                prediction, reg_loss = self(data)
                loss = F.mse_loss(prediction, target)

                # compute gradient in the computational graph
                loss.backward()
    
                # update parameters in the model 
                self.optim.step()
                # torch.cuda.empty_cache()
                del data, target
                if self.opt.eval and idx+1 == len(training_generator):
                #     # evaluate model
                    self.eval()
                    print('Val')
                    NRMSE,R2 = self.evaluate_model(val)
                    losses.append([NRMSE,R2])
                    # if self.opt.cuda:
                    #     self = self.cuda()
                    print(NRMSE)
                    # # update learning rate
                    if clientTrain == False:
                        self.sche.step(NRMSE)
                    
                    # Eary NRMSE
                    if (self.bestRMSE-self.opt.EStol)<=NRMSE:
                        self.numBadEps +=1
                    else:
                        self.numBadEps = 0
                        self.bestRMSE = NRMSE 
                        
                    if self.numBadEps >= self.opt.ESpat:
                        print('Stopping Early')
                        if ~clientTrain:
                            return self,losses
                    print(self.numBadEps)
                    del  NRMSE,R2
                   
                    self.train() 

        return self,losses
    def evaluate_model(self,test):
            self.eval()
            
            with torch.no_grad():
                if self.opt.cuda:
                    self.cuda()
                params = {'batch_size': self.opt.eval_batch_size,
                  'shuffle': True,
                  'num_workers': self.opt.num_threads}
                validation_generator = DataLoader(test, pin_memory=True,**params)
                PCC = 0
                count = 0
                Ybar = 0
                for batch_idx, batch in enumerate(validation_generator):
                    Y = batch['target']
                    if self.opt.cuda:
                        Y = Y.cuda()
                    Ybar += torch.sum(Y)
                    count +=len(Y)
                Ybar /= count
                
                count = 0
                Ypred=[]
                Target =[]
                for batch_idx, batch in enumerate(validation_generator):
                    X = batch['data']
                    Y = batch['target']
                    if len(Y)<2:
                        continue
                    if self.opt.cuda:
                        X = X.cuda()
                        Y = Y.cuda()
                    Y = Y.view(len(Y), -1)
                    prediction, reg_loss = self(X)  
                    Yprediction = prediction.view(len(prediction), -1)
                    Ypred.extend(Yprediction.cpu().squeeze().tolist())
                    Target.extend(Y.cpu().squeeze().tolist())
                    count += len(Y)
                del X,Y,validation_generator
                PCC = np.corrcoef(np.asarray(Ypred),np.asarray(Target),rowvar=False)[0][1]
                MSE = np.mean((np.asarray(Ypred) - np.asarray(Target))**2);  
            return MSE,PCC
