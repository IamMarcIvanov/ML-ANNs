# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
import timer


class ANN:
    def __init__(self, n_inputs, n_outputs, hidden_sizes=[3]):
        self.nx = n_inputs
        self.ny = n_outputs
        self.nh = len(hidden_sizes)
        self.sizes = [self.nx] + hidden_sizes + [self.ny]
        self.lr = 0.01
        self.hidden_layers = hidden_sizes[0]
        self.W = {
            i+1: np.random.randn(self.sizes[i], self.sizes[i+1]) for i in range(self.nh + 1)}
        self.B = {i+1: np.zeros((1, self.sizes[i+1])) for i in range(self.nh + 1)}
        self.dW = None
        self.H = None

    def forward_pass(self, x):
        A = {}
        self.H = {}
        self.H[0] = x.reshape(1, -1)
        for i in range(self.nh):
            A[i+1] = np.matmul(self.H[i], self.W[i+1]) + self.B[i+1]
            A[i + 1] = [self.H[i+1]if A[i + 1] > 100000000
            self.H[i+1] = 1.0 / (1.0 + np.exp(- A[i + 1]))
        A[self.nh+1]=np.matmul(self.H[self.nh],
                                    self.W[self.nh+1]) + self.B[self.nh+1]
        self.H[self.nh + 1] = np.exp(A[self.nh + 1]) / np.sum(np.exp(A[self.nh + 1]))
        return self.H[self.nh + 1]

    def predict(self, X):
        Y_pred=[]
        for x in X:
            y_pred=self.forward_pass(x)
            Y_pred.append(y_pred)
        return np.array(Y_pred).squeeze()

    def cross_entropy(self, label, pred):
        yl=np.multiply(pred, label)
        yl=yl[yl != 0]
        yl=-np.log(yl)
        yl=np.mean(yl)
        return yl

    def grad(self, x, y):
        self.forward_pass(x)
        self.dW={}
        self.dB={}
        dH={}
        dA={}
        L=self.nh + 1
        dA[L]=(self.H[L] - y)
        for k in range(L, 0, -1):
            self.dW[k]=np.matmul(self.H[k-1].T, dA[k])
            self.dB[k]=dA[k]
            dH[k-1]=np.matmul(dA[k], self.W[k].T)
            dA[k-1]=np.multiply(dH[k-1], self.H[k-1] * (1 - self.H[k-1]))

    def fit(self, X, Y, epochs=10000, learning_rate=0.01, display_loss=False):
        self.lr=learning_rate
        if display_loss:
            loss={}

        for epoch in range(epochs):
            dW={}
            dB={}
            for i in range(self.nh+1):
                dW[i+1]=np.zeros((self.sizes[i], self.sizes[i+1]))
                dB[i+1]=np.zeros((1, self.sizes[i+1]))
            for x, y in zip(X, Y):
                self.grad(x, y)
                for i in range(self.nh+1):
                    dW[i+1] += self.dW[i+1]
                    dB[i+1] += self.dB[i+1]

            m=X.shape[1]
            for i in range(self.nh+1):
                self.W[i+1] -= learning_rate * (dW[i+1]/m)
                self.B[i+1] -= learning_rate * (dB[i+1]/m)

            if display_loss:
                Y_pred=self.predict(X)
                loss[epoch]=self.cross_entropy(Y, Y_pred)

        if display_loss:
            plt.plot(loss.values())
            plt.xlabel('Epochs')
            plt.ylabel('Cross Entropy Loss')
            plt.title('Hidden Layers: ' + str(self.hidden_layers) +
                      ' Learning Rate: ' + str(self.lr))
            plt.show()


def process(data):
    X=data.iloc[:, :-1]
    y=data.iloc[:, -1]
    X=(X-X.mean(axis=0)) / X.std(axis=0)

    return X, y


def one_hot(arr):
    encoding=[]
    vals=sorted(set([a[0] for a in arr]))
    m, k=min(vals), len(vals)
    for ind, a in enumerate(arr):
        curr=np.zeros(k)
        curr[a[0] - m]=1.0
        encoding.append(curr)
    return np.asarray(encoding)

data=pd.read_csv(
    r'E:\BITS\Yr 3 Sem 2\BITS F464 Machine Learning\Assignments\Assignment 2\2B\dataset_NN.csv')
X, y=process(data)

def test_train_split(x, y, test_size=0.25):
    x=x.sample(frac=1, random_state=32).reset_index(inplace=False)
    y=y.sample(frac=1, random_state=32).reset_index(inplace=False, drop=True)
    n_rows=len(x)
    n_train=int(n_rows * test_size)
    n_test=n_rows - n_train
    x_train=x.head(n_train).to_numpy()
    x_test=x.tail(n_test).to_numpy()
    y_train=y.head(n_train).to_numpy()
    y_test=y.tail(n_test).to_numpy()
    return x_train, x_test, y_train, y_test

X_train, X_test, y_train, y_test=test_train_split(X, y, test_size=0.3)

y_OH_train=one_hot(np.expand_dims(y_train, 1))
y_OH_test=one_hot(np.expand_dims(y_test, 1))


def accuracy_score(y_pred, y_true):
    correct=0
    for ind, pred in enumerate(y_pred):
        if pred == y_true[ind]:
            correct += 1
    return correct / len(y_true)


for n_hidden_layers in [1, 2]:
    for rate in [0.1, 0.01, 0.001]:
        nn=ANN(X_train.shape[1], len(
            y.unique()), hidden_sizes=[n_hidden_layers])
        nn.fit(X_train, y_OH_train, display_loss=True, learning_rate=rate)

        predictions=nn.predict(X_test)
        predictions=predictions.astype(int).ravel()
        y_OH_test=y_OH_test.astype(int).ravel()
        accuracy_test=accuracy_score(predictions, y_OH_test)
        print('Accuracy:', accuracy_test,
              'Hidden Layers:', n_hidden_layers,
              'Learning Rate:', rate)

# %%
