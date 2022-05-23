import glob
import torch
import cv2
import sys
import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import torch.optim as optim
import torch.nn.functional as F
import time
import copy

class TestImport:
    def __init__(self):
        pass
    def test(self):
        print('ok')

class ImageData:
    def __init__(self,dir=os.getcwd()+'/dataset',preload=True):
        self.data_struct = {}
        self.preload = preload
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        for t in glob.glob(dir+'/*'):
            self.data_struct[t.split('/')[-1]] = {}
            for e in glob.glob(dir+'/{}/*'.format(t.split('/')[-1])):
                self.data_struct[t.split('/')[-1]][e.split('/')[-1]] = []
                for i in glob.glob(dir+'/{}/{}/*'.format(t.split('/')[-1],e.split('/')[-1])):
                    if self.preload:
                        sys.stderr.write('reading image {} into memory preload=True\n'.format(i))
                        self.data_struct[t.split('/')[-1]][e.split('/')[-1]].append(cv2.equalizeHist(cv2.imread(i,cv2.IMREAD_GRAYSCALE)))
                    else:
                        sys.stderr.write('locating image {} preload=False\n'.format(i))
                        self.data_struct[t.split('/')[-1]][e.split('/')[-1]].append(i)

    def get(self,set,label,index):
        exp_idx = self.emotions.index(label)
        target = [0,0,0,0,0,0,0]
        target[exp_idx] = 1
        target = torch.Tensor(target).float()
        target = target.view(1,-1)
        try:
            if self.preload:
                return (torch.reshape(torch.from_numpy(self.data_struct[set][label][index]).float(),(1,1,48,48)), target)
            else:
                return (torch.reshape(torch.from_numpy(cv2.imread(self.data_struct[set][label][index])).float(),(1,1,48,48)), target)
        except Exception as e:
            sys.stderr.write('key not in dict! {}\n'.format(e))
            return None

class SMOTE_balancing:
    def __init__(self,n_percent=1):
        self.n_perc = n_percent
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.X_smote = None
        self.y_smote = None
        self.X_train = None


    def compute(self,dataset):
        df_agg = pd.DataFrame(data=[], columns=["label", "image"])
        for em in self.emotions:
            current_emotion_data = dataset.data_struct["train"][em]
            len_to_keep = int(self.n_perc*len(current_emotion_data))
            for i in range(len_to_keep):
                current_image = current_emotion_data[i]
                current_row =  pd.DataFrame({"label":[em], "image":[current_image]})
                df_agg = df_agg.append(current_row, ignore_index=True)
        smote = SMOTE(random_state=123)
        y = df_agg['label']
        X = np.stack(df_agg['image'])
        n_samples, height, width = [X.shape[index] for index in range(3)]
        X_reshaped = X.reshape(n_samples, height*width)
        self.X_smote, self.y_smote = smote.fit_resample(X_reshaped, y)
        self.X_train = self.X_smote.reshape(len(self.X_smote), 48, 48)
        return self



def get_device():
    print('cuda is_available: ',torch.cuda.is_available())
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # kernel
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=4)
        # an affine operation: y = Wx + b
        self.conv2 = torch.nn.Conv2d(6,16,6)
        self.fc1 = torch.nn.Linear(16 * 8 * 8, 120)  # 5*5 from image dimension
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 7)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(model, crit, opt, train_data, eval_data,num_epochs=10):
    start_time = time.time()
    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f'epoch {epoch+1}/{num_epochs}')
        print()
        for phase in ['train','eval']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corr = 0.0


            for input, label in train_data:


                opt.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    output = model(input)
                    _, preds = torch.max(output, 1)
                    loss = crit(output,label)

                    if phase == 'train':
                        loss.backward()
                        opt.step()

                running_loss += loss.item()*input.size(0)
                running_corr += torch.sum(preds == label)


            if phase == 'train':
                epoch_loss = running_loss/len(train_data)
                epoch_acc = running_corr.double() / len(train_data)
            else:
                epoch_loss = running_loss/len(eval_data)
                epoch_acc = running_corr.double() / len(eval_data)

            print(f'{phase} loss={epoch_loss} acc={epoch_acc}')

            if phase == 'eval' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - start_time
    print(f'done in {time_elapsed}\nbest_acc={best_acc}')

    model.load_state_dict(best_wts)
    return model

def evaluation(model,test_data):
    model.eval()
    accuracy = 0.0
    for input, label in test_data:
        output = model(input)
        equality = (label.data == output.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()
        test_accuracy = accuracy / len(test_data)
    print(f'test_acc={test_accuracy}')

def save_model(model,path=os.getcwd()+'model.pt'):
    torch.save(model, path)

def load_model(path=os.getcwd()+'model.pt'):
    model = torch.load(path)
    return model
