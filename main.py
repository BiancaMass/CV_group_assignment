#!/usr/bin/env python3

CLEAN=True
DATASET_FOLDER = 'dataset'
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

acc_train = []
loss_train = []
acc_eval = []
loss_eval = []

import os
from utils import *
import torch.optim as optim
from random import shuffle


if __name__ == '__main__':

    data = ImageData(preload=True)
    print('smoting')
    smote_output = SMOTE_balancing(n_percent=1).compute(data)
    dev = 'cpu'
    model = Net().to(dev)
    print(model)

    megalist = []
    #for em in EMOTIONS:
    #    for i in range(len(data.data_struct['train'][em])):
    #        input, target = data.get('train',em,i)
    #        megalist.append((input,target))

    for i in range(len(smote_output.X_train)):
        megalist.append(smote_output.get(i))


    shuffle(megalist)

    train_data = megalist[0:int(len(megalist)*0.64)]
    eval_data =  megalist[int(len(megalist)*0.64):int(len(megalist)*0.8)]
    test_data =  megalist[int(len(megalist)*0.8)::]

    assert len(train_data) + len(test_data) + len(eval_data) == len(megalist)

    crit = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    ans, to_plot = train2(model,crit,optimizer,train_data, eval_data,num_epochs=25)
    #ans = load_model('model_history/1.pt')
    save_model(ans,'model_history/3.pt')
    acc = eval2(ans, test_data)























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
