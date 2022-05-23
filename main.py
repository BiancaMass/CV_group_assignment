#!/usr/bin/env python3

CLEAN=True
DATASET_FOLDER = 'dataset'
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

import os
from utils import *
import torch.optim as optim
from random import shuffle

if __name__ == '__main__':

    data = ImageData(preload=True)
    print('smoting')
    #smote_output = SMOTE_balancing(n_percent=1).compute(data)
    dev = 'cpu'
    model = Net().to(dev)
    print(model)

    megalist = []
    for em in EMOTIONS:
        for i in range(len(data.data_struct['train'][em])):
            input, target = data.get('train',em,i)
            megalist.append((input,target))

    shuffle(megalist)

    train_data = megalist[0:int(len(megalist)*0.64)]
    eval_data =  megalist[int(len(megalist)*0.64):int(len(megalist)*0.8)]
    test_data =  megalist[int(len(megalist)*0.8)::]

    assert len(train_data) + len(test_data) + len(eval_data) == len(megalist)

    crit = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    ans = train_model(model,crit,optimizer,megalist,num_epochs=20)





























    '''for thing in megalist:
        optimizer.zero_grad()
        output = model(input)
        loss = crit(output, target)
        loss.backward()
        optimizer.step()

    #eval
    model.eval()
    loss = 0
    with torch.no_grad():
        for em in EMOTIONS:
            for i in range(len(data.data_struct['test'][em])):
                input, target = data.get('test',em,i)
                output = model(input)
                loss += crit(output,target)
                _, predictions = torch.max(output.data, dim=1)
    print('loss',loss)'''
