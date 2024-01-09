# -*- coding: utf-8 -*-
"""
@author: Daniel Nolte
Adapted from DanielNolte/FederatedDeepRegressionForests: https://github.com/DanielNolte/FederatedDeepRegressionForests
"""
import gc
import numpy as np
import ndf
import torch

import torch.nn.functional as F
import logging       
from torch.utils.data import  DataLoader
seed_value= 10
 
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

FLT_MIN = float(np.finfo(np.float32).eps)
class FedDRF():
    def __init__(self,opt):
        # super(FedDRF, self).__init__()
        # RNDF consists of two parts:
        #1. a feature extraction model using residual learning
        #2. a neural decision forst
        self.opt = opt
        self.feat_layer = ndf.FeatureLayer(opt)
        self.forest = ndf.Forest(opt.n_tree, opt.tree_depth, opt.num_output,opt, opt.cuda)
        self.model = ndf.NeuralDecisionForest(self.feat_layer, self.forest)  
        if self.opt.cuda:
            self.model.cuda(device=self.opt.gpuid)
        # else:
        #     raise NotImplementedError
        self.optim, self.sche = self.prepare_optim(self.model, opt)
        self.numBadEps = 0
        self.bestRMSE = 1000
        
    def get_model_weights(self):
        param = self.model.state_dict()
        return param
    def update_model(self,update):
        self.model.load_state_dict(update)
        del update
        return self
        
    def prepare_optim(self,model, opt):
        params = [ p for p in model.parameters() if p.requires_grad]
        if opt.optim_type == 'adam':
            optimizer = torch.optim.Adam(params, lr = opt.lr, 
                                      weight_decay = opt.weight_decay,amsgrad =False)
        elif opt.optim_type == 'sgd':
            optimizer = torch.optim.SGD(params, lr = opt.lr, 
                                    momentum = opt.momentum,
                                    weight_decay = opt.weight_decay)    
    # scheduler with pre-defined learning rate decay
    # automatically decrease learning rate
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            mode='min',
                                                            factor=0.1,
                                                            patience=opt.LRsched-1,
                                                            verbose=True,
                                                            min_lr=0.000001)
        return optimizer, scheduler   
    def evaluate_model(self,test):
        self.model.eval()
        
        with torch.no_grad():
            if self.opt.cuda:
                self.model.cuda(device=self.opt.gpuid)
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
                # Ybar = torch.mean(Y)
                prediction, reg_loss = self.model(X)  
                Yprediction = prediction.view(len(prediction), -1)
                Ypred.extend(Yprediction.cpu().squeeze().tolist())
                Target.extend(Y.cpu().squeeze().tolist())
                count += len(Y)
            del X,Y,validation_generator
            PCC = np.corrcoef(np.asarray(Ypred),np.asarray(Target),rowvar=False)[0][1]
            MSE = np.mean((np.asarray(Ypred) - np.asarray(Target))**2);  
        return MSE,PCC
    def prepare_batches(self,model,train):
    # prepare some feature batches for leaf node distribution update
        with torch.no_grad():
            target_batches = []
            params = {'batch_size': self.opt.leaf_batch_size,
                      'shuffle': True,
                      'num_workers': self.opt.num_threads}
            training_generator = DataLoader(train,pin_memory=True, **params)
            num_batch = int(np.ceil(self.opt.label_batch_size/self.opt.leaf_batch_size))
            for batch_idx, sample in enumerate(training_generator):
                if batch_idx == num_batch:
                    break
                data = sample['data']
                
                target = sample['target']
                target = target.view(len(target), -1)
                if len(target)<2:
                    continue
                if self.opt.cuda:
                    with torch.no_grad():
                        # move to GPU
                        data = data.cuda()
                        target = target.cuda()                          
                # Get feats
                feats, _ = model.feature_layer(data)
                # release data Tensor to save memory
                del data
                for tree in model.forest.trees:
                    mu = tree(feats)
                    # add the minimal value to prevent some numerical issue
                    mu += FLT_MIN # [batch_size, n_leaf]
                    # store the routing probability for each tree
                    tree.mu_cache.append(mu)
                # release memory
                del feats
                # the update rule will use both the routing probability and the 
                # target values
                target_batches.append(target)   
            del target
        return target_batches

     
    def train1(self,train,val,clientTrain=False):
        """
        Args:
            model: the model to be trained
            optim: pytorch optimizer to be used
            db : prepared torch dataset object
            opt: command line input from the user
            exp_id: experiment id
        """        
        best_model_dir = os.path.join(self.opt.save_dir)
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        
        params = {'batch_size': self.opt.batch_size,
          'shuffle': True,
          'num_workers': self.opt.num_threads}
        losses = []
        for epoch in range(1, self.opt.epochs + 1):
            # At each epoch, train the neural decision forest and update
            # the leaf node distribution separately 
            
            # Train neural decision forest
            # set the model in the training mode
            self.model.train()
            training_generator = DataLoader(train,pin_memory=True, **params)
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
                prediction, reg_loss = self.model(data)
                
                loss = F.mse_loss(prediction, target) + reg_loss
                
                # compute gradient in the computational graph
                loss.backward()
    
                # update parameters in the model 
                self.optim.step()
                # torch.cuda.empty_cache()
                del data, target,sample

                # Update the leaf node estimation    
                if idx+1 == len(training_generator):
                    logging.info("Epoch %d : Update leaf node prediction"%(epoch))
                    print("Epoch %d : Update leaf node prediction"%(epoch))
                    target_batches = self.prepare_batches(self.model, train)
                    print('Batches Prepared')
                    for i in range(self.opt.label_iter_time):
                        # prepare features from the last feature layer
                        # some cache is also stored in the forest for leaf node
                        for tree in self.model.forest.trees:
                            tree.update_label_distribution(target_batches)
                    # release cache
                    del target_batches
                    for tree in self.model.forest.trees:   
                        del tree.mu_cache
                        tree.mu_cache = []
                            
                if self.opt.eval and idx+1 == len(training_generator):
                #     # evaluate model
                    self.model.eval()
                    print('Val')
                    NRMSE,R2 = self.evaluate_model(val)
                    losses.append([NRMSE,R2])
                    if self.opt.cuda:
                        self.model = self.model.cuda(device=self.opt.gpuid)
                    print(NRMSE)
                    # update learning rate
                    if clientTrain == False:
                        self.sche.step(NRMSE)
                    
                    # Eary Stopping
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
                    del NRMSE,R2
                   
        del training_generator,val,train
           
        torch.cuda.empty_cache()
        gc.collect()
        logging.info('Training finished.')
        return self,losses
        

        