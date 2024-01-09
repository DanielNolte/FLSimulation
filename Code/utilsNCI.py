# -*- coding: utf-8 -*-
"""
@author: Daniel Nolte

Adapted from Nicholasli1995/VisualizingNDF: https://github.com/Nicholasli1995/VisualizingNDF
& DanielNolte/FederatedDeepRegressionForests: https://github.com/DanielNolte/FederatedDeepRegressionForests
miscellaneous utility functions
"""
import argparse
import time
import logging
import pandas as pd
import numpy as np
from torch import save
from os import path as path
import os 

# function for parsing input arguments
def parse_arg():
    parser = argparse.ArgumentParser(description='runSim.py')
    ## paths
    parser.add_argument('-data_path', type=str, default='../data')
    parser.add_argument('-dataset', type=str, default='NCI60')
    parser.add_argument('-save_dir', type=str, default='../models/')
    ## Data settings
    parser.add_argument('-beta', type=float, default=None,help="Beta of the dirichlet distribution that controls the heterogeneity of the federated dataset across clients")
    parser.add_argument('-numSamples', type=int, default=1000,help="number of smaples to include in the federation")
    ##-----------------------------------------------------------------------##
    ## model settings
    parser.add_argument('-save_name', type=str, default='model') 
    parser.add_argument('-model', type=str, default='FedDRF', help="Choose from [FedDRF,FedRF,ANNFull]") #'FedDRF' FedRF ANNFull
    parser.add_argument('-n_tree', type=int, default=6)
    parser.add_argument('-tree_depth', type=int, default=4)
    parser.add_argument('-num_output', type=int, default=128) # only used for coupled routing functions   
    ## training settings
    parser.add_argument('-trainCent', action='store_true', default=False, help="Boolean to train centralized alt") 
    parser.add_argument('-trainFed',  action='store_true', default=False, help="Boolean to train Federate model")
    parser.add_argument('-evalCent', action='store_true', default=False, help="Boolean to evaluate centralized alt")
    parser.add_argument('-evalFed',action='store_true', default=False, help="Boolean to evaluate Federate model")
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-eval_batch_size', type=int, default=128)
    parser.add_argument('-leaf_batch_size', type=int, default=128) 
    parser.add_argument('-label_batch_size', type=int, default=40000)
    # random seed 
    parser.add_argument('-seed', type=int, default=2020)    
    parser.add_argument('-num_threads', type=int, default=0)
    parser.add_argument('-label_iter_time', type=int, default=3)
    parser.add_argument('-cuda', type=int, default=1)
    parser.add_argument('-gpuid', type=int, default=0)
    parser.add_argument('-epochs', type=int, default=200)
    parser.add_argument('-eval', type=bool, default=True)
    ##-----------------------------------------------------------------------##    
    # Optimizer settings
    parser.add_argument('-optim_type', type=str, default='adam')
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-clientlr', type=float, default=0.001)
    parser.add_argument('-EStol', type=float, default=.0001)
    parser.add_argument('-ESpat', type=float, default=10)
    parser.add_argument('-LRsched', type=float, default=5)
    parser.add_argument('-weight_decay', type=float, default=0.001/(2*250))
    parser.add_argument('-momentum', type=float, default=0.9, help="sgd: 0.9")
    opt = parser.parse_args()
    return opt

def get_save_dir(opt, modelType,str_type=None):
    if str_type == 'his':
        root = opt.save_his_dir
    else:
        root = opt.save_dir
    if not os.path.exists(root):
        os.makedirs(root)
    save_name = path.join(root, opt.save_name) 
    save_name += '_model_type_' 
    save_name += opt.model
    save_name += '_RNDF_'
    save_name += '_depth{:d}_tree{:d}_output{:d}'.format(opt.tree_depth, opt.n_tree, opt.num_output)
    save_name += modelType
    # save_name += time.asctime(time.localtime(time.time()))
    save_name += '.pth'
    return save_name
def get_log_dir(opt, modelType,str_type=None):
    if str_type == 'his':
        root = opt.save_his_dir
    else:
        root = opt.save_dir
    if not os.path.exists(root):
        os.makedirs(root)
    save_name = path.join(root, opt.save_name) 
    save_name += '_model_type_' 
    save_name += opt.model
    save_name += '_RNDF_'
    save_name += '_depth{:d}_tree{:d}_output{:d}'.format(opt.tree_depth, opt.n_tree, opt.num_output)
    save_name += modelType
    # save_name += time.asctime(time.localtime(time.time()))
    save_name += '.xlsx'
    return save_name

def save_losses_time(lossHistory,elapsed,opt,modelType):
    LOG_FILENAME = get_log_dir(opt,modelType)
    hist = pd.DataFrame(lossHistory)
    hist = hist.append([elapsed])
    hist.to_excel(LOG_FILENAME)
    
def save_model(model, opt,modelType):
    save_name = get_save_dir(opt,modelType)
    save(model, save_name)
    return

def save_best_model(model, path):
    save(model, path)
    return

def update_log(best_model_dir, MAE,):
    text = time.asctime(time.localtime(time.time())) + ' '
    text += "Current MAE: " + MAE[0]
    text += "Best MAE: " + MAE[1] + "\r\n"
    with open(os.path.join(best_model_dir, "log.txt"), "a") as myfile:
        myfile.write(text)
    return

def save_history(train_his, eval_his, opt):
    save_name = get_save_dir(opt, 'his')
    train_his_name = save_name +'train_his_stage' 
    eval_his_name = save_name + 'eval_his_stage' 
    if not path.exists(opt.save_his_dir):
        os.mkdir(opt.save_his_dir)
    save(train_his, train_his_name)
    save(eval_his, eval_his_name)
    
def split_dic(data_dic):
    img_path_list = []
    age_list = []
    for key in data_dic:
        img_path_list += data_dic[key]['path']
        age_list += data_dic[key]['age_list']
    img_path_list = np.array(img_path_list)
    age_list = np.array(age_list)
    total_imgs = len(img_path_list)
    random_indices = np.random.choice(total_imgs, total_imgs, replace=False)
    num_train = int(len(img_path_list)*0.8)
    train_path_list = list(img_path_list[random_indices[:num_train]])
    train_age_list = list(age_list[random_indices[:num_train]])
    valid_path_list = list(img_path_list[random_indices[num_train:]])
    valid_age_list = list(age_list[random_indices[num_train:]])
    train_dic = {'path':train_path_list, 'age_list':train_age_list}
    valid_dic = {'path':valid_path_list, 'age_list':valid_age_list}
    return train_dic, valid_dic

def check_split(train_dic, eval_dic):
    train_num = len(train_dic['path'])
    valid_num = len(eval_dic['path'])
    logging.info("Image split: {:d} training, {:d} validation".format(train_num, valid_num))
    logging.info("Total unique image num: {:d} ".format(len(set(train_dic['path'] + eval_dic['path']))))
    return
def NRMSE(Y_Target, Y_Predict):
    Y_Target = np.array(Y_Target); Y_Predict = np.array(Y_Predict);
    Y_Target = Y_Target.reshape(len(Y_Target),1);    Y_Predict = Y_Predict.reshape(len(Y_Predict),1);    
    Y_Bar = np.mean(Y_Target)
    Nom = np.sum((Y_Predict - Y_Target)**2);    Denom = np.sum((Y_Bar - Y_Target)**2)
    MSE = np.mean((Y_Predict - Y_Target)**2);   NRMSE = np.sqrt(Nom/Denom)
    PCC = np.corrcoef(np.asarray(Y_Predict),np.asarray(Y_Target),rowvar=False)[0][1]
    return MSE, PCC
