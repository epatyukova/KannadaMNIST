import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime

from model import CNN
from utils import reset_weights, test_func

from utils import TrainDataset, TestDataset
from torch.utils.data import DataLoader
from torchvision import transforms


from torch.optim.lr_scheduler import StepLR



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Supervisor:
    def __init__(self,trainset, testset, device ,**kwargs):
        self.trainset=trainset
        self.testset=testset
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._train_kwargs = kwargs.get('train')
        self._eval_kwargs = kwargs.get('eval')
        self._seed_kwargs = kwargs.get('seed')
        self.val_frac=self._data_kwargs.get('val_frac')
        self.batch_size=self._data_kwargs.get('batch_size')
        self.rotation=self._data_kwargs.get('rotation_angle')
        self.affine=self._data_kwargs.get('rotation_angle')
        self.resizecrop_scale=float(self._data_kwargs.get('resizecrop_scale'))
        self.resizecrop_ratio=float(self._data_kwargs.get('resizecrop_ratio'))
        self.epochs=int(self._train_kwargs.get('epochs',1))
        self.use_testset=bool(self._train_kwargs.get('use_testset'))
        self.num_iter=int(self._train_kwargs.get('num_iter_on_testset',1))
        self.patience=int(self._train_kwargs.get('patience',1))
        self.test_freq=int(self._train_kwargs.get('test_every_n_epochs',1))
        self.ensemble=bool(self._eval_kwargs.get('ensemble'))
        self.fix_random_seed=bool(self._seed_kwargs.get('fix_random_seed'))
        self.seed=int(self._seed_kwargs.get('random_seed'))
        
        if(self.fix_random_seed):
	        torch.manual_seed(self.seed)
        
        
    def train(self):
              
        val_set = self.trainset.sample(frac = self.val_frac)
        train_set_new = self.trainset.drop(val_set.index)
        val_set=val_set.reset_index(drop=True)
        train_set_new=train_set_new.reset_index(drop=True)
        test_set_new=self.testset
        
        trainset_new=TrainDataset(train_set_new,transform=transforms.Compose([transforms.RandomRotation(self.rotation),
                                                            transforms.RandomAffine(self.affine),
                                                            transforms.RandomResizedCrop(size=28, scale=(self.resizecrop_scale,1.0), ratio=(self.resizecrop_ratio,1.0/self.resizecrop_ratio))]))
        valset=TrainDataset(val_set)
        testset_new=TestDataset(test_set_new)
        
        
        train_dataloader = DataLoader(trainset_new, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(valset, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(testset_new, batch_size=1, shuffle=False)
        
        model=CNN()
    
        
        print('device:', device, ', model training...')
        if(self.use_testset==False):
            self._train(train_dataloader,val_dataloader,model)
        else:
            for run in range(self.num_iter):
                print('training round:', run, ' ------------')
                self._train(train_dataloader,val_dataloader,model,run)
                if(run<self.num_iter-1):
                    train_set_new, test_set_new=self._testset_split(test_dataloader, run, train_set_new, test_set_new)
                    trainset_new=TrainDataset(train_set_new,transform=transforms.Compose([transforms.RandomRotation(self.rotation),
                                                            transforms.RandomAffine(self.affine),
                                                            transforms.RandomResizedCrop(size=28, scale=(self.resizecrop_scale,1.0), ratio=(self.resizecrop_ratio,1.0/self.resizecrop_ratio))]))
                    testset_new=TestDataset(test_set_new)
                    train_dataloader = DataLoader(trainset_new, batch_size=self.batch_size, shuffle=True)
                    test_dataloader = DataLoader(testset_new, batch_size=1, shuffle=False)
                    
                                   
                        
    def _train(self,train_dataloader,val_dataloader,model,run=None):  
        wait=0
        min_los=100
        train_loss_list=[]
        val_loss_list=[]
        accur_train_list=[]
        accur_val_list=[]
        
        size = len(train_dataloader.dataset)
        model.apply(reset_weights)
        model.to(device)
        model.train()
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=2e-2)
        scheduler = StepLR(optimizer,step_size=4,gamma=0.2)
        for epoch in range(self.epochs):
            print('epoch:',epoch,'   #############')
            for batch, (X, y) in enumerate(train_dataloader):
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            scheduler.step()
            if epoch % self.test_freq == 0 or epoch==self.epochs-1:
                acc,los=test_func(train_dataloader, model, device, loss_fn, mode='train')
                accur_train_list.append([epoch,acc])
                train_loss_list.append([epoch,los])
                acc,los=test_func(val_dataloader, model, device, loss_fn, mode='test')
                accur_val_list.append([epoch,acc])
                val_loss_list.append([epoch,los])
                if(los < min_los):
                    min_los=los
                    wait=0
                    if(self.use_testset):
                        torch.save(model.state_dict(), "models/model-round"+str(run)+".pth")
                    else:
                        torch.save(model.state_dict(), "models/model.pth")
                else:
                    wait+=self.test_freq
                    if(wait>self.patience):
                        print('Patience is over!')
                        if(self.use_testset==False):
                            self._save_log(train_loss_list, accur_train_list,'log/train_data.csv')
                            self._save_log(val_loss_list, accur_val_list,'log/val_data.csv')
                        else:
                            self._save_log(train_loss_list, accur_train_list,'log/train_data_round'+str(run)+'.csv')
                            self._save_log(val_loss_list, accur_val_list,'log/val_data_round'+str(run)+'.csv')
                        break
            if epoch==self.epochs-1:
                if(self.use_testset==False):
                    self._save_log(train_loss_list, accur_train_list,'log/train_data.csv')
                    self._save_log(val_loss_list, accur_val_list,'log/val_data.csv')
                else:
                    self._save_log(train_loss_list, accur_train_list,'log/train_data_round'+str(run)+'.csv')
                    self._save_log(val_loss_list, accur_val_list,'log/val_data_round'+str(run)+'.csv')
   
    def evaluate(self):
        testset_new=TestDataset(self.testset)
        size = len(self.testset)
        test_dataloader = DataLoader(testset_new, batch_size=1, shuffle=False)
        
        if(self.ensemble == True and self.use_testset == True and self.num_iter>1):
            probs=np.zeros((size,10))
            ids=np.zeros(size)
            b=np.ones((size,10))*1.0/float(self.num_iter)
            for run in range(self.num_iter):
                model=CNN()
                model.load_state_dict(torch.load("models/model-round"+str(run)+".pth"))
                model.to(device)
                model.eval()
                with torch.no_grad():
                    for ind, (X, y) in enumerate(test_dataloader): 
                        X, y = X.to(device), y.to(device)
                        probs[ind,:]+=F.softmax(model(X),dim=1).cpu().detach().numpy()[0,:]
                        ids[ind]=int(y[0].item())

                                    
            probs=probs*b
            predicted = probs.argmax(1)
            data=np.concatenate((ids.reshape((size,1)),predicted.reshape((size,1))),axis=1)
            prediction_df=pd.DataFrame(columns=['id','label'],data=data.astype(int))
            prediction_df.to_csv("submission.csv",index=False)
        else:
            probs=np.zeros((size,10))
            ids=np.zeros(size)
            model=CNN()
            if(self.use_testset == True and self.num_iter>1):
                model.load_state_dict(torch.load("models/model-round"+str(self.num_iter-1)+".pth"))
            elif(self.use_testset == False):
                model.load_state_dict(torch.load("models/model.pth"))
            else:
                model.load_state_dict(torch.load("models/model-round0.pth"))
            model.to(device)
            model.eval()
            with torch.no_grad():
                for ind, (X, y) in enumerate(test_dataloader): 
                    X, y = X.to(device), y.to(device)
                    probs[ind,:]+=F.softmax(model(X),dim=1).cpu().detach().numpy()[0,:]
                    ids[ind]=int(y[0].item())
            predicted = np.array(probs.argmax(1),dtype=int)
            data=np.concatenate((ids.reshape((size,1)),predicted.reshape((size,1))),axis=1)
            prediction_df=pd.DataFrame(columns=['id','label'],data=data.astype(int))
            prediction_df.to_csv("submission.csv",index=False)
        
        
        return
    
    def _save_log(self,loss_list, acc_list,name):
        loss_list=np.array(loss_list)
        acc_list=np.array(acc_list)
        epoch=loss_list[:,0]
        loss=loss_list[:,1]
        acc=acc_list[:,1]
        df=pd.DataFrame(columns=['epoch','loss','accuracy'],data=np.transpose([epoch,loss,acc]))
        df.to_csv(name,index=False)
        
    def _testset_split(self,test_dataloader, run, train_set_new, test_set_new):
        
        model=CNN()
        model.load_state_dict(torch.load("models/model-round"+str(run)+".pth"))
        

        new_labels=[]
        count=0
        model.to(device)
        model.eval()
        with torch.no_grad():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                prob = F.softmax(pred,dim=1)

        
                if(prob[0,pred[0].argmax(0)]>0.95):
                    new_labels.append(pred[0].argmax(0).item())
                else:
                    new_labels.append(10)
                    count+=1
        
        test_set_new['label']=new_labels
        test_to_train = test_set_new[test_set_new.label != 10]
        test_to_test=test_set_new[test_set_new.label == 10]
        test_set_new=test_to_test.drop(['label'], axis=1)
        test_to_train=test_to_train.drop(['id'], axis=1)
        cols = test_to_train.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        test_to_train = test_to_train[cols]
        train_set_new=pd.concat([train_set_new, test_to_train], ignore_index=True)
        
        train_set_new=train_set_new.reset_index(drop=True)
        test_set_new=test_set_new.reset_index(drop=True)
        
        return train_set_new, test_set_new
        
        
        
        
        
        
        
        
        
        
        
