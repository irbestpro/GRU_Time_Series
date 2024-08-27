import numpy as np
import pandas as pd
from torch.optim import Adam
import torch
import torch.nn as nn
import math
import pickle
import asyncio

#_______________GRU Model________________________

class GRU_Model(nn.Module):
    def __init__(self , hidden_size , num_layers):
        super().__init__()
        self.hidden_size , self.num_layers = hidden_size , num_layers
        self.GRU = nn.GRU(input_size = 1 , hidden_size = hidden_size , num_layers = num_layers , bidirectional = False , batch_first = True) # GRU Layer
        self.FC = nn.Sequential(nn.Linear(hidden_size , 1)) # Fully Connected Layer

    def forward(self , X):
        out , _ = self.GRU(X)
        out = self.FC(out).squeeze(1) if len(X.size()) == 2 else self.FC(out.unsqueeze(0)).squeeze(0).squeeze(2) # prepare data for batch prediction 
        return out

class Time_Series_Model():

    #____________Init Phase________________________

    def __init__(self , DatasetName , * , epoch = 100, lag = 4 , train_size = 0.8 , test_size = 0.2 , learning_rate = 0.0001 , num_layers = 8 , hidden_size = 32):
        
        self.model = None
        self.DatasetName = DatasetName
        self.Data = pd.read_csv(f'./TempFiles/{self.DatasetName}' , header=0 , skip_blank_lines = True)['count'].to_numpy(dtype = np.float32) # Load the dataset and convert it's data to numpy array
        #self.Data = pd.read_csv(f'./TempFiles/{self.DatasetName}' , header=0 , skiprows=3 , skip_blank_lines = True)['count'].to_numpy(dtype = np.float32) # Load the dataset and convert it's data to numpy array
        self.Data = self.Data[~np.isnan(self.Data)] # remove nan values
        self.epoch = epoch
        self.lag = lag
        self.train_size = train_size
        self.test_size = test_size
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    #___________Prediction Phase___________________

    async def Predict(self):
        
        mean_value = self.Data.mean(axis = 0)
        self.Data = self.Data / mean_value # count field normalization
        self.model = GRU_Model(hidden_size = self.hidden_size , num_layers = self.num_layers) # the model object instance
        optimizer = Adam(self.model.parameters() , lr = self.learning_rate) # model optimizer
        Loss = nn.MSELoss() # mse loss function

        #________________Train Step________________________
        
        self.model.train()
        best_loss_value = math.inf
        best_model = None
        for epoch in range(1 , self.epoch):
            temp_loss = 0
            steps = 0
            for batch in range(0 , math.floor(self.Data.shape[0] * self.train_size - self.lag)): 
                counts = torch.from_numpy(self.Data[batch :  batch + self.lag]).unsqueeze(1) # get count values at time t - lag 
                goals = torch.as_tensor(self.Data[batch + 1 : batch + self.lag + 1]) # get goal value at time t
                out = self.model(counts) # model ouput
                loss = Loss(out , goals) # loss value at time t
                temp_loss += loss.item()
                steps += 1

                self.model.zero_grad() # equals to optimizer.zero_grad()
                loss.backward() # backpropagation step
                optimizer.step() # new optimizer step
        
        #__________________Test Step_________________________

            self.model.eval() # switch to evaluation mode
            with torch.no_grad():
                test_temp_loss = 0
                test_steps = 0
                for batch in range(math.floor(self.Data.shape[0] * self.train_size) - self.lag + 1 , self.Data.shape[0] - self.lag): 
                    test_series = torch.from_numpy(self.Data[batch :  batch + self.lag]).unsqueeze(1) # get test series
                    expected = torch.as_tensor(self.Data[batch + 1 : batch + self.lag + 1]) # get expected value at time t
                
                    predicted = self.model(test_series) # get predicted count at time t
                    test_loss = Loss(predicted , expected) # evaluate the results 
                    test_temp_loss += test_loss.item() 
                    test_steps += 1

                    #yield str([(i.data , j) for i,j in zip(predicted * mean_value , expected * mean_value)])

                best_loss_value , best_model = ((test_temp_loss / test_steps) , self.model) if best_loss_value > (test_temp_loss / test_steps) else (best_loss_value , best_model)
                pickle.dump(best_model , open (f'./CheckPoints/{self.DatasetName}.mdl' , 'wb'))
                yield f'Epoch # {epoch} - Train loss : {temp_loss / steps} and Test loss : {test_temp_loss / test_steps} \n '
                await asyncio.sleep(0.1)
                
        
        yield f'The Optimized Loss value is {best_loss_value} \n'
        yield '____________________________________________________________________________\n\n'
        yield f'The model with the best checkpoint is saved! [ Model-Name:{self.DatasetName}.mdl]'







