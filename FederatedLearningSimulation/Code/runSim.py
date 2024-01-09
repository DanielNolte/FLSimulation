# -*- coding: utf-8 -*-
"""
@author: Daniel Nolte
Adapted from DanielNolte/FederatedDeepRegressionForests: https://github.com/DanielNolte/FederatedDeepRegressionForests
"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from Server import Server
import numpy as np
import utilsNCI as utils
import torch
import time
import pandas as pd

if __name__ == '__main__':

    #Parameters  
    n = 5
    numLocalSteps = 100
    trainingRounds = 400
    fracTestData = 0.2
    cell = 'A549'
    
    opt = utils.parse_arg()
    assert opt.beta in [100,1,0.1,None],'Beta only tested with the following: [100 (Nearly Random splitting),1(Mild Heterogeneity),0.1(Severe Heterogeneity),None], None results in a random split'
    assert opt.model in ['FedDRF','FedRF','ANNFull'],'Model must be one of the following: [\'FedDRF\',\'FedRF\',\'ANNFull\']'
    assert opt.numSamples <=40000,'numSamples must be less than 40,000'
    allResults = []   
    results = pd.DataFrame(index=['MSE','PCC'])
    

    save_dir = opt.save_dir
    opt.save_dir = save_dir+str(opt.seed)+'/'+cell+'_'+str(opt.numSamples)+'_'+str(opt.beta)
    opt.batch_size=int(np.min([np.max([np.ceil(opt.numSamples/numLocalSteps),8]),256]))
    #Init server with defined parameters
    ser = Server(n,trainingRounds,fracTestData,opt,cell)
    # ser.train_Fed_svm()
    
            
    #Early Stopping Params
    bestRMSE = 1000
    numBadEps = 0
    LRdelay = 0
    if opt.trainFed:
        #Run the FL Process
        if opt.model != 'FedRF':
            t = time.time()
            for rnd in range(trainingRounds):
                print('Training Rounnd: ',rnd+1)
                LRdelay += 1
                #for each round, call run_one_round and save the outputs
                round_output = ser.run_one_round()
                if ((bestRMSE-opt.EStol)<=round_output[2]):
                    numBadEps +=1
                else:
                    numBadEps = 0
                    bestRMSE = round_output[2] 
                    best = round_output[3]
                    
                if (numBadEps >= opt.LRsched) & (LRdelay >= opt.LRsched):
                    print('###Reducing LR####')
                    LRdelay = 0
                    for c in ser.clients:
                        c.reduce_lr()
                if (numBadEps >= opt.ESpat) :#& (rnd>25):
                    print('###Stopping Early###')
                    break
                print(numBadEps)
                print(LRdelay)
            elapsed = time.time() - t
            utils.save_losses_time(round_output[4],elapsed,opt,'FedValClients'+str(n))
            utils.save_losses_time(round_output[5],elapsed,opt,'FedTrainClients'+str(n))
            utils.save_losses_time(round_output[6],elapsed,opt,'FedSer'+str(n))
            utils.save_model(round_output[3], opt,'Fed'+str(n))
            utils.save_model(best, opt,'bestFed'+str(n))
            print('Finished Training')
        else:
            params = {'random_state': opt.seed,
                              'n_jobs': -1,
                              'min_samples_leaf':5,
                              'max_features':0.5}
            ser.train_FedRF(params)
           
    elif opt.evalFed:
        save_dir = utils.get_save_dir(opt,'bestFed'+str(n))
        print(opt.cuda)  
        if opt.cuda:
            fedModel = torch.load(save_dir)
            ser.opt.cuda = 1
        else:
            fedModel = torch.load(save_dir,map_location=torch.device('cpu'))   
        print('Federated Eval')         
        
        if opt.model =='FedDRF':
            print('FedDRF')
            FedTrainResults,FedValResults,FedTestResults = ser.fed_eval(fedModel)
            results['TrainDRF'+str(n)+'_'+str(opt.beta)] = FedTrainResults
            results['ValDRF'+str(n)+'_'+str(opt.beta)] = FedValResults
            results['TestDRF'+str(n)+'_'+str(opt.beta)] = FedTestResults
        elif opt.model =='FedRF':
            print('FedRF')
            fedModel = torch.load(save_dir,map_location=torch.device('cpu')) 
            FedTrainResults,FedValResults,FedTestResults = ser.fed_eval_rf(fedModel)
            results['TrainFedRandomForest'+str(n)+'_'+str(opt.beta)] = FedTrainResults
            results['ValFedRandomForest'+str(n)+'_'+str(opt.beta)] = FedValResults
            results['TestFedRandomForest'+str(n)+'_'+str(opt.beta)] = FedTestResults
        else:
            print('FedANN')
            FedTrainResults,FedValResults,FedTestResults = ser.fed_eval(fedModel)
            results['TrainANN'+str(n)+'_'+str(opt.beta)] = FedTrainResults
            results['ValANN'+str(n)+'_'+str(opt.beta)] = FedValResults
            results['TestANN'+str(n)+'_'+str(opt.beta)] = FedTestResults
    del ser

            
                
                