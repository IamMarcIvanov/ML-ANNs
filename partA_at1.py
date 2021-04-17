# %%
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, 
                 dataPath, 
                 lr=0.01, 
                 iterations=1000, 
                 train_test_split=0.7,
                 batch_size=100,
                 n_epochs=10):
        
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
        self.loss  = []
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.fscore = []
        self.batch_size = batch_size
        self.n_epochs = n_epochs
    
    def setData(self):
        df = pd.read_csv(self.dataPath)
        df.sample(frac=1).reset_index(inplace=True)
        cols = df.columns
        n_rows = df.shape[0]
        self.n_train_rows = math.floor(self.train_test_split * n_rows)
        self.n_test_rows = n_rows - self.n_train_rows
        self.xTrain = df[cols[:-1]].head(self.n_train_rows).to_numpy()
        self.xTest = df[cols[:-1]].tail(self.n_test_rows).to_numpy()
        self.yTrain = df[cols[-1]].head(self.n_train_rows).to_numpy()
        self.yTest = df[cols[-1]].tail(self.n_test_rows).to_numpy()
        self.weights = np.zeros(self.xTrain.shape[1])

    def fit(self, sgd=False):
        if sgd:
            self.SGDfit()
        else:
            self.GDfit()
    
    def GDfit(self):
        for i in range(self.n_iters):
            yPred = 1 / (1 + np.exp(- (self.xTrain.dot(self.weights) + self.bias)))
            diff = (yPred - self.yTrain).reshape(self.n_train_rows)
            
            delW = np.dot(self.xTrain.T, diff) / self.n_train_rows
            delb = np.sum(diff) / self.n_train_rows

            self.weights = self.weights - self.learning_rate * delW
            self.bias = self.bias - self.learning_rate * delb
            
            if i % 50 == 0:
                self.addMetrics()
        self.addMetrics()
    
    def SGDfit(self):
        for i in range(self.n_epochs):
            indices = np.random.choice(self.xTrain.shape[0], self.batch_size, replace=False)
            xtrain = self.xTrain[indices]
            ytrain = self.yTrain[indices]
            for j in range(self.batch_size):
                yPred = 1 / (1 + np.exp(- (xtrain.dot(self.weights) + self.bias)))
                diff = (yPred - ytrain).reshape(self.batch_size)

                delW = np.dot(xtrain.T, diff) / self.batch_size
                delb = np.sum(diff) / self.batch_size

                self.weights = self.weights - self.learning_rate * delW
                self.bias = self.bias - self.learning_rate * delb

                if (i * self.batch_size + j) % 50 == 0:
                    self.addMetrics()
        self.addMetrics()
    
    def addMetrics(self):
        h_xTest = 1 / (1 + np.exp(- (self.xTest.dot(self.weights) + self.bias)))
        loss = np.dot(- self.yTest,  np.log(h_xTest)) - np.dot(1 - self.yTest, np.log(1 - h_xTest))
        yPred = np.where(h_xTest > 0.5, 1, 0)
        tp, fp, fn, tn = 0, 0, 0, 0
        for i in range(self.n_test_rows):
            if self.yTest[i] == 1 and yPred[i] == 1:
                tp += 1
            elif self.yTest[i] == 0 and yPred[i] == 1:
                fp += 1
            elif self.yTest[i] == 1 and yPred[i] == 0:
                fn += 1
            elif self.yTest[i] == 0 and yPred[i] == 0:
                tn += 1
        self.loss.append(loss)
        self.accuracy.append((tp + tn) / (tp + tn + fp + fn))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        self.fscore.append(2 * precision * recall / (precision + recall))
        self.precision.append(precision)
        self.recall.append(recall)
    
    def plotter(self, 
                acc=True, 
                loss=True, 
                fscore=True, 
                precision=True, 
                recall=True,
                sgd=False):
        
        iters = [50 * i for i in range(len(self.loss) - 1)]
        if sgd:
            if loss:
                plt.plot(iters, self.loss[:-1], color='r', label='Loss')
                plt.title('Stochastic Gradient Descent ; Loss ; lr=' + str(self.learning_rate))
                plt.show()
            if acc:
                plt.plot(iters, self.accuracy[:-1], color='g', label='Accuracy')
            if precision:
                plt.plot(iters, self.precision[:-1], color='b', label='Precision')
            if recall:
                plt.plot(iters, self.recall[:-1], color='y', label='Recall')
            if fscore:
                plt.plot(iters, self.fscore[:-1], color='r', label='Fscore')
            plt.legend()
            plt.title('Stochastic Gradient Descent ; lr=' + str(self.learning_rate))
            plt.show()
        else:
            if loss:
                plt.plot(iters, self.loss[:-1], color='r', label='Loss')
                plt.title('Gradient Descent ; Loss ; lr=' + str(self.learning_rate))
                plt.show()
            if acc:
                plt.plot(iters, self.accuracy[:-1],
                         color='g', label='Accuracy')
                plt.title('Gradient Descent ; Accuracy ; lr=' + str(self.learning_rate))
            if precision:
                plt.plot(iters, self.precision[:-1], color='b', label='Precision')
            if recall:
                plt.plot(iters, self.recall[:-1], color='y', label='Recall')
            if fscore:
                plt.plot(iters, self.fscore[:-1], color='r', label='Fscore')
            plt.legend()
            plt.title('Gradient Descent ; lr=' + str(self.learning_rate))
            plt.show()
    
        
dataLocation = r'E:\BITS\Yr 3 Sem 2\BITS F464 Machine Learning\Assignments\Assignment 2\2A\dataset_LR.csv'
sgd_acc_sum, gd_acc_sum, sgd_loss_sum, gd_loss_sum = 0, 0, 0, 0
for i in range(10):
    model = LogisticRegression(dataLocation, lr=1)
    model.setData()
    model.fit(sgd=False)
    if i == 0:
        model.plotter(sgd=False)
    agd, lgd = model.accuracy[-1], model.loss[-1]
    gd_acc_sum += agd
    gd_loss_sum += lgd

    model = LogisticRegression(dataLocation, lr=1)
    model.setData()
    model.fit(sgd=True)
    if i == 0: 
        model.plotter(sgd=True)
    asgd, lsgd = model.accuracy[-1], model.loss[-1]
    sgd_acc_sum += asgd
    sgd_loss_sum += lsgd
    
    if i == 0:
        print('{:^5}{:^30}{:^30}{:^30}'.format('No.', 'Stat Name', 'Gradient Descent','Stochastic Gradient Descent'))
     
    print('{:^5}{:^30}{:^30}{:^30}'.format(i + 1, 'Accuracy', agd, asgd))
    print('{:^5}{:^30}{:^30}{:^30}'.format('', 'Loss', lgd, lsgd))

print('{:^5}{:^30}{:^30}{:^30}'.format('All', 'Avg Accuracy', '{:.6f}'.format(gd_acc_sum / 10), '{:.6f}'.format(sgd_acc_sum / 10)))
print('{:^5}{:^30}{:^30}{:^30}'.format('All', 'Avg Loss', '{:.6f}'.format(gd_loss_sum / 10), '{:.6f}'.format(sgd_loss_sum / 10)))
