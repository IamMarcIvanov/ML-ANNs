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
        SVC(),
        LinearDiscriminantAnalysis(),
        Perceptron(),
        LogisticRegression(max_iter=1000),
        MLPClassifier(max_iter=1000)]
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
                                                                      'Average'))
for i, model in enumerate(models):
    score = cross_val_score(model, X, y.values.ravel(), cv=7)
    score=[round(acc, 3) for acc in score]
    scores.append(score)
    print('{:<20}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}'.format(modelName[i],
                                                                          score[0],
                                                                          score[1],
                                                                          score[2],
                                                                          score[3],
                                                                          score[4],
                                                                          score[5],
                                                                          score[6],
                                                                          round(np.average(score), 3)))

plt.figure(figsize=(10,8))
bp = plt.boxplot(scores, sym='', widths=0.5,showmeans=True)
plt.setp(bp['boxes'], color='#66ddaa',linewidth=2)
plt.setp(bp['whiskers'], color='indianred')
plt.setp(bp['caps'], color='black')
plt.setp(bp['means'], color='#654321')
plt.xticks(range(1,len(modelName)+1), modelName, rotation ='vertical')
plt.savefig("plot.png",dpi=1500)
plt.show()

maxx=-1
for i in scores:
    if maxx<=max(i):
        maxx=max(i)
print('Maximum Accuracy:', max(scores[4]))