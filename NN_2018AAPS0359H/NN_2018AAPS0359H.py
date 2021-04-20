import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def oneHotEncoding(y, n_labels):
	n_samples = y.shape[0]
	encoded_y = np.zeros((n_labels, n_samples))
	for j in range(n_samples):

		classindx = int(y[j])-1
		encoded_y[classindx, j] = 1

	return encoded_y


def parametersSetting(num_x, hidden_sizes, num_y):
	H_1, H_2 = hidden_sizes[:]
	W1 = np.random.normal(0, 1, size=(H_1, num_x))
	b1 = np.zeros((H_1, 1))
	if(H_2 != 0):
		W2 = np.random.normal(0, 1, size=(H_2, H_1))
		b2 = np.zeros((H_2, 1))
		W3 = np.random.normal(0, 1, size=(num_y, H_2))
		b3 = np.zeros((num_y, 1))
		return np.array([W1, b1, W2, b2, W3, b3], dtype=object)

	W2 = np.random.normal(0, 1, size=(num_y, H_1))
	b2 = np.zeros((num_y, 1))
	return np.array([W1, b1, W2, b2], dtype=object)


def loss_fn(calc, real):
    n_samples = real.shape[0]
    logp = - np.log(calc[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    return loss


def feedforward(X, mult_param, hidden_sizes):
	H_1, H_2 = hidden_sizes[:]
	if(H_2 == 0):
		W1, b1, W2, b2 = mult_param
		A1 = 1 / (1 + np.exp(- W1 @ X + b1))
		Z2 = W2@A1+b2
		A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)
		mult_nodes = np.array([W1 @ X + b1, A1, Z2, A2], dtype=object)
	else:
		W1, b1, W2, b2, W3, b3 = mult_param
		A1 = 1 / (1 + np.exp(- W1@X+b1))
		Z2 = W2@A1+b2
		A2 = 1 / (1 + np.exp(- Z2))
		Z3 = W3@A2+b3
		A3 = np.exp(Z3) / np.sum(np.exp(Z3), axis=0)
		mult_nodes = np.array([W1@X+b1, A1, Z2, A2, Z3, A3], dtype=object)

	return mult_nodes


def grad_fn(mult_nodes, mult_param, X, y, hidden_sizes):
	H_1, H_2 = hidden_sizes[:]
	m = X.shape[1]
	if(H_2 == 0):
		W1, b1, W2, b2 = mult_param
		Z1, A1, Z2, A2 = mult_nodes
		dz2 = (A2-y)
		dw2 = dz2@(A1.T)/m
		db2 = np.sum(dz2, axis=1, keepdims=True)/m
		da1 = (W2.T)@dz2
		dz1 = da1*(A1*(1-A1))
		dw1 = dz1@(X.T)/m
		db1 = np.sum(dz1, axis=1, keepdims=True)/m
		grad_param = np.array([dw1, db1, dw2, db2], dtype=object)

	else:
		W1, b1, W2, b2, W3, b3 = mult_param
		Z1, A1, Z2, A2, Z3, A3 = mult_nodes
		dw3 = (A3-y)@(A2.T)/m
		db3 = np.sum((A3-y), axis=1, keepdims=True)/m
		da2 = (W3.T)@(A3-y)
		dz2 = da2*(A2*(1-A2))
		dw2 = dz2@(A1.T)/m
		db2 = np.sum(dz2, axis=1, keepdims=True)/m
		da1 = (W2.T)@dz2
		dz1 = da1*(A1*(1-A1))
		dw1 = dz1@(X.T)/m
		db1 = np.sum(dz1, axis=1, keepdims=True)/m
		grad_param = np.array([dw1, db1, dw2, db2, dw3, db3], dtype=object)

	return grad_param


def grad_desc_fn(X, y, eta, epochs, hidden_sizes, batch_size):
	H_1, H_2 = hidden_sizes[:]
	n_labels = y.shape[0]
	errors = np.array([])
	mult_accuracy = np.array([])
	mult_param = parametersSetting(X.shape[0], hidden_sizes, n_labels)

	for j in range(epochs):
		mult_nodes = feedforward(X, mult_param, hidden_sizes)
		grad_param = grad_fn(mult_nodes, mult_param, X, y, hidden_sizes)
		mult_param -= eta*grad_param
		calc = mult_nodes[-1]

		loss = loss_fn(calc.T, y.T)
		errors = np.append(errors, loss)
		calc_labels = np.argmax(calc, axis=0) + 1
		acc = np.sum(calc_labels == train_labels) / len(train_labels) * 100
		mult_accuracy = np.append(mult_accuracy, acc)

	return mult_param, errors, mult_accuracy


data = pd.read_csv(
    'dataset_NN.csv')
data = data.to_numpy()
np.random.shuffle(data)


eta = 0.5
epochs = 1000
n_labels = 10
len_train = int(0.7 * len(data))


train = data[0: len_train]
x_train, y_train = (train[:, :-1].T, train[:, -1].T)
train_labels = y_train
x_train = (x_train - np.mean(x_train, axis=1, keepdims=True)) / \
    np.std(x_train, axis=1, keepdims=True)
y_train = y_train.reshape(len(y_train), 1)
y_train = oneHotEncoding(y_train, n_labels)


test = data[len_train: len(data)]
x_test, y_test = (test[:, :-1].T, test[:, -1].T)
test_labels = y_test
x_test = (x_test - np.mean(x_test, axis=1, keepdims=True)) / \
    np.std(x_test, axis=1, keepdims=True)
y_test = y_test.reshape(len(y_test), 1)
y_test = oneHotEncoding(y_test, n_labels)


err = [0, 0]
acc = [0, 0]
for ind, hsz in enumerate([[20, 0], [20, 20]]):
    print('Number of Hidden Layers: {:^3}'.format(ind + 1))
    mp, err[ind], acc[ind] = grad_desc_fn(x_train, y_train, eta, epochs, hsz, int(0.5 * x_train.shape[1]))
    tr = np.argmax(feedforward(x_train, mp, hsz)[-1], axis=0) + 1
    a = np.sum(tr == train_labels) / len(train_labels) * 100
    print("Training Accuracy {:.2f}%".format(a))
    
    ts = np.argmax(feedforward(x_test, mp, hsz)[-1], axis=0) + 1
    a = np.sum(ts == test_labels) / len(test_labels) * 100
    print("Testing Accuracy {:.2f}%\n".format(a))

i = 1
for e, ma in zip(err, acc):
	plt.plot(e)
	plt.title('Number of Hidden Layers: {:^3}'.format(i))
	plt.xlabel('Number of Epochs')
	plt.ylabel('Loss')
	plt.show()

	plt.plot(ma)
	plt.title('Number of Hidden Layers: {:^3}'.format(i))
	plt.xlabel('Number of Epochs')
	plt.ylabel('Accuracy')
	plt.show()
	i += 1
