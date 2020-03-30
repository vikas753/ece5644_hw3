import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import csv

class Net(nn.Module):
    def __init__(self,num_perceptrons,num_inputs):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_inputs,num_perceptrons)
        self.fc2 = nn.Linear(num_perceptrons,num_inputs)
    def forward(self, x):
        dataV = self.fc2(F.relu(self.fc1(x)))
        y = F.softmax(dataV,dim=0)
        return dataV


def normal(x,mu,sigma):
    return ( 2.*np.pi*sigma**2. )**-.5 * np.exp( -.5 * (x-mu)**2. / sigma**2. )


def nntraining(num_inputs,num_perceptrons,x,y,labels):
    """plt.plot(x,y,".")"""
     
    criterion = nn.CrossEntropyLoss()
     
    net = Net(num_perceptrons,num_inputs)

    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(100):
        X, Y = Variable(torch.FloatTensor([x]), requires_grad=True), Variable(torch.FloatTensor([y]), requires_grad=False)
        labels = Variable(torch.FloatTensor([labels]), requires_grad=False)
        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        """print("Epoch {} - loss: {}".format(epoch, loss.item()))"""


    return net

def nnvalidate(nnet,dataX,dataY,labels):
    X, Y = Variable(torch.FloatTensor([dataX]), requires_grad=True), Variable(torch.FloatTensor([dataY]), requires_grad=False)
    labels = Variable(torch.FloatTensor([labels]), requires_grad=False)
    criterion = nn.CrossEntropyLoss()
    outputs = nnet(X)
    loss = criterion(outputs, labels)
    return loss.item()     

def KfoldCrossValidation(dataX,dataY,labels,kfold,num_perceptrons):
    group_len = len(dataX)
    group_factor = int(group_len / kfold)
    num_zeros = group_len - 2*group_factor
    
    min_loss = 10.0
    trainingNN = nntraining(group_len,num_perceptrons,dataX,dataY,labels)
    bestNNModel = nntraining(group_len,num_perceptrons,dataX,dataY,labels)
    
    for i in range(0,kfold,1):
        group_index_start = i*group_factor
        group_index_end   = (i+1)*group_factor
        dataX_train = dataX
        dataY_train = dataY
        labels_train = labels
        dataX_list  = dataX_train.tolist()
        dataY_list  = dataY_train.tolist()
        labels_list = labels_train.tolist() 
        dataX_test   = dataX[group_index_start:group_index_end]
        dataY_test   = dataY[group_index_start:group_index_end]
        labels_test  = labels[group_index_start:group_index_end]
        del dataX_list[group_index_start:group_index_end]
        del dataY_list[group_index_start:group_index_end]
        del labels_list[group_index_start:group_index_end]
        dataX_test_list  = dataX_test.tolist() + test_append
        dataY_test_list  = dataY_test.tolist() + test_append
        labels_test_list = labels_test.tolist() + test_append 
        trainingNN = nntraining(len(dataX_list),num_perceptrons,dataX_list,dataY_list,labels_list)
        loss = nnvalidate(trainingNN,dataX_test_list,dataY_test_list,labels_test_list)
        if loss < min_loss:
            min_loss = loss        
            bestNNModel = trainingNN
    
    return bestNNModel
 
def Experiment(num_perceptrons): 


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
    dataSampleLength = int(len(filedataX)/10)
    Kfolds = 10
    filedataX = np.array(filedataX)
    filedataY = np.array(filedataY)
    
    x = filedataX 
    y = filedataY
    trainedNN = KfoldCrossValidation(x[0:dataSampleLength],y[0:dataSampleLength],labels[0:dataSampleLength],Kfolds,num_perceptrons)

    batchLength = dataSampleLength
 
    indexRange = int(len(x)/batchLength)
    lossAvg = 0

    for i in range(indexRange-1):
        dataSetStart = int(batchLength * (i+1))
        dataSetEnd   = int(dataSetStart + batchLength)
        dataVar = Variable(torch.FloatTensor(x[dataSetStart:dataSetEnd]), requires_grad=True)
        Y = Variable(torch.FloatTensor(y[dataSetStart:dataSetEnd]), requires_grad=False)
        out = trainedNN(dataVar)
        criterion = nn.MSELoss()
        loss = criterion(out, Y)   
        lossAvg = lossAvg + loss.item()

    lossMean = lossAvg / indexRange
    return lossMean
    
for i in range(40):
    lossMean = Experiment(i+1)
    print(" Loss mean with num_perceptrons : " , i , " loss mean : " , lossMean)
    

