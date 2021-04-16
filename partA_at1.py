# %%
import pandas as pd
from sklearn.linear_model import LogisticRegression
import math
import numpy as np
import matplotlib.pyplot as plt

class LogistiRegression:
    def __init__(self, dataPath, lr=0.01, iterations=10000, train_test_split=0.7):
        self.dataPath = dataPath
        self.xTrain = None
        self.xTest = None
        self.yTrain = None
        self.yTest = None
        self.weights = None
        self.bias = 0
        self.learning_rate = lr
        self.n_iters =  iterations
        self.train_test_split = 0.7
        self.n_train_rows = 0
        self.n_test_rows = 0
    
    def setData(self):
        df = pd.read_csv(self.dataPath)
        df.sample(frac=1).reset_index(inplace=True)
        cols = df.columns
        n_rows = df.shape[0]
        self.n_train_rows = math.floor(self.train_test_split * n_rows)
        self.n_test_rows = n_rows - n_train_rows
        self.xTrain = df[cols[:-1]].head(self.n_train_rows).to_numpy()
        self.xTest = df[cols[:-1]].tail(self.n_test_rows).to_numpy()
        self.yTrain = df[cols[-1]].head(self.n_train_rows).to_numpy()
        self.yTest = df[cols[-1]].tail(self.n_test_rows).to_numpy()
        self.weights = np.zeros(self.xTrain.shape[1])

    def fit(self):
        for i in range(self.n_iters):
            yPred = 1 / (1 + np.exp(- (self.xTrain.dot(self.weights) + self.bias)))
            diff = (self.yPred - self.yTrain).reshape(self.n_train_rows)
            
            delW = np.dot(self.xTrain.T, diff) / self.n_train_rows
            delb = np.sum(diff) / self.n_train_rows

            loss = - self.yTest
            if i % 50 == 0:
                plt.plot()
    
    def getLoss(self):
        yPred = - self.yTest 

    
        
# %%
dataLocation = r'E:\BITS\Yr 3 Sem 2\BITS F464 Machine Learning\Assignments\Assignment 2\2A\dataset_LR.csv'
model = LogistiRegression(dataLocation)
model.setData()

# %%
dff = pd.DataFrame({'animal': ['monkey',
                               'tiger',
                               'rat',
                               'cat',
                               'human',
                               'cheetah'],
                    'names': ['Danish',
                              'Huma',
                              'Anirudh',
                              'Dheeraj',
                              'Farzana',
                              'Utkarsh'],
                    'surnames': ['Mohammad',
                                 'Shaikh',
                                 'Jha',
                                 'Sood',
                                 'Ranka',
                                 'Mussolini']})
print(dff[['animal', 'surnames']].head(3))


