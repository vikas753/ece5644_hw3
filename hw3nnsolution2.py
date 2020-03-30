import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import csv

from sklearn.mixture import GaussianMixture

""" find the error with the number of selected components """
def gmm_geterror(data_train,n_comps,labels):
    gmm = GaussianMixture(n_components=n_comps).fit(data_train)
    labels_out = gmm.predict(data_train)
    num_error = 0
    
    length_list = len(labels)
    
    for i in range(length_list):
        test_label     = int(labels[i])
        test_label_out = int(labels_out[i])
        change_label   = test_label - test_label_out
        if change_label != 1:
            num_error = num_error + 1    
    
    return num_error

""" Get the error with below training data X and Y , with labels """
def gmm_training(dataX,dataY,labels):
    data_train = np.vstack((dataX,dataY)).T
    min_error = gmm_geterror(data_train,1,labels)
    optimum_num_comps = 1
    for i in range(4):
        error = gmm_geterror(data_train,(i+1),labels)
        if error < min_error:
            min_error = error
            optimum_num_comps = i+1
    
    return optimum_num_comps

""" find the error of number of components with GMM validation """
def gmm_validate(dataX,dataY,labels,num_components_training):
    data_train = np.vstack((dataX,dataY)).T
    return gmm_geterror(data_train,num_components_training,labels)

""" K-fold validation procedure for GMM mixture modeling """
def KfoldCrossValidation(dataX,dataY,labels,kfold):
    group_len = len(dataX)
    group_factor = int(group_len / kfold)
    
    min_loss = group_len
    num_opt_components = 1
    loss_array = []
    
    for i in range(1,kfold-1,1):
        group_index_start = i*group_factor
        group_index_end   = (i+1)*group_factor
        dataX_train = dataX
        dataY_train = dataY
        labels_train = labels
        dataX_list  = dataX_train[0:group_index_start] + dataX_train[group_index_end:group_len]
        dataY_list  = dataY_train[0:group_index_start] + dataY_train[group_index_end:group_len]
        labels_list = labels_train[0:group_index_start] + labels_train[group_index_end:group_len] 
        dataX_test   = dataX[group_index_start:group_index_end]
        dataY_test   = dataY[group_index_start:group_index_end]
        labels_test  = labels[group_index_start:group_index_end]
        loss = gmm_validate(dataX_test,dataY_test,labels_test,i) 
        print(" Loss : " , loss , " num_components_training : " , i)
        loss_array = loss_array + [loss]
        if loss < min_loss:
            min_loss = loss        
            num_opt_components = i
    
    plt.plot(loss_array , linewidth=2.0)
    plt.figure()
    
    print(" min_loss " , min_loss)
    print(" num_opt_components " , num_opt_components)
    return num_opt_components

def Experiment(): 
    with open('dataX10k.csv', newline='') as f:
        reader = csv.reader(f)
        filedataX = [tuple(row) for row in reader]

    with open('dataY10k.csv', newline='') as f:
        reader = csv.reader(f)
        filedataY = [tuple(row) for row in reader]
    
    with open('labels10k.csv', newline='') as f:
        reader = csv.reader(f)
        labels = [tuple(row) for row in reader]


    n = len(filedataX[0])
    
    filedataX = [float(i) for i in filedataX[0]]    
    filedataY = [float(j) for j in filedataY[0]]
    labels    = [float(k) for k in labels[0]]
    
    kFold = 10
    
    validationLength = int(n / kFold)
    
    data_train = np.vstack((filedataX,filedataY)).T
    num_components = KfoldCrossValidation(filedataX[0:validationLength],filedataY[0:validationLength],labels,10)    
    plt.scatter(filedataX, filedataY, c=labels, s=40, cmap='viridis')
    plt.figure()    
    gmm = GaussianMixture(n_components=4).fit(data_train)
    labels_out = gmm.predict(data_train)
    num_error = 0
    
    length_list = len(labels)
    
    for i in range(length_list):
        test_label     = int(labels[i])
        test_label_out = int(labels_out[i])
        change_label   = test_label - test_label_out
        if change_label != 1:
            num_error = num_error + 1    
        
        
    print(" Error : " , num_error , " total samples : " , n)    
    plt.scatter(filedataX, filedataY, c=labels_out, s=40, cmap='viridis');
    plt.show()
    probs = gmm.predict_proba(X)

Experiment()