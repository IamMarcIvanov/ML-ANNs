# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

df=pd.read_csv(r'E:\BITS\Yr 3 Sem 2\BITS F464 Machine Learning\Assignments\Assignment 2\2C\dataset_comb.csv')
df.drop(['id'], axis=1, inplace=True)
X=df.iloc[:,:10]
y=df.iloc[:,10:]

models=[GaussianNB(),
        SVC(kernel='linear', tol=1e-6),
        LinearDiscriminantAnalysis(),
        Perceptron(tol=1e-6),
        LogisticRegression(max_iter=1000),
        MLPClassifier(hidden_layer_sizes=(10), max_iter=1000)]
modelName=["GaussianNB",
           "SVC",
           "LDA",
           "Perceptron",
           "LogisticRegression",
           "MLPClassifier"]


scores=[]
print('{:<20}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}'.format('Model',
                                                                      'Round 1',
                                                                      'Round 2',
                                                                      'Round 3',
                                                                      'Round 4',
                                                                      'Round 5',
                                                                      'Round 6',
                                                                      'Round 7',
                                                                      'Median'))
for i, model in enumerate(models):
    score = cross_val_score(model, X, y.values.ravel(), cv=7)
    score = [round(acc, 6) for acc in score]
    scores.append(score)
    print('{:<20}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}'.format(modelName[i],
                                                                          score[0],
                                                                          score[1],
                                                                          score[2],
                                                                          score[3],
                                                                          score[4],
                                                                          score[5],
                                                                          score[6],
                                                                          round(np.median(score), 6)))

accDF = pd.DataFrame(np.array(scores).T, columns=modelName)
fig, ax = plt.subplots(figsize=(20, 15))
ax = accDF.boxplot(sym='')
ax.set_title('7-fold Cross Val Boxplot', fontsize=20)
ax.set_ylabel("Accuracy", fontsize=14)
plt.show()

folds = [1, 2, 3, 4, 5, 6, 7]
plt.plot(folds, scores[0], label=modelName[0])
plt.plot(folds, scores[1], label=modelName[1])
plt.plot(folds, scores[2], label=modelName[2])
plt.plot(folds, scores[3], label=modelName[3])
plt.plot(folds, scores[4], label=modelName[4])
plt.plot(folds, scores[5], label=modelName[5])
plt.legend()
plt.title('Variation of Accuracy over the folds', fontsize=20)
plt.xlabel('Fold Number')
plt.ylabel('Accuracy')
plt.show()

print('Maximum Accuracy:', max(max(scores)))

# %%
