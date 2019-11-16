import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import svm, metrics
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import matplotlib as mpl
from matplotlib import colors
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,precision_score, recall_score, f1_score, fbeta_score
import warnings
def iris_type(t):
    targets =[]
    for i in range(0,len(t)):
        # print(row)
        # print(t.iloc[i])


        if t.iloc[i] == 'Iris-setosa':
            # print(t.iloc[i][4])
            # t.iloc[i] = 0
            targets.append(0)
        if t.iloc[i]== 'Iris-versicolor':
            # t.iloc[i] = 1
            targets.append(1)
        if t.iloc[i]== 'Iris-virginica':
             # t.iloc[i] = 2
            targets.append(2)
    return targets


data=pd.read_csv('iris.data.txt',header=None)
x = data.drop(4,axis=1)
y = data[4]

targets = np.array(iris_type(y))
# plt.figure(figsize=(10,10))

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20)
# svclassifier = svm.SVC(kernel='linear')
svclassifier = svm.SVC(kernel='rbf',gamma='auto')

svclassifier.fit(x_train, y_train)

y_hat_train = svclassifier.predict(x_train)


y_hat_test = svclassifier.predict(x_test)

x=np.array(x)
x_train = np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)

accuracy_train = []
accuracy_test = []

for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
    clf = svm.SVC(kernel=kernel, gamma=10)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_train)
    # print(y_pred)

    atrain = accuracy_score(y_train, y_pred)
    atest = accuracy_score(y_test,clf.predict(x_test))
    # m = accuracy_score(targets, y_pred)
    accuracy_train.append(atrain)
    accuracy_test.append(atest)
    print('Train dataset, Accuracy (',kernel,'): ',atrain)

    print('Test dataset, Accuracy (',kernel,'): ',atest)



    plt.figure('Task 2.1,'+kernel)
    plt.clf()
    plt.scatter(x[:, 0], x[:, 1], c=targets, zorder=10, cmap=plt.cm.Paired,
                edgecolor='k', s=20)

    # Circle out the test data
    plt.scatter(x_test[:, 0], x_test[:, 1], s=80, facecolors='none',
                zorder=10, edgecolor='k')

    plt.axis('tight')
    x_min = x[:, 0].min()
    x_max = x[:, 0].max()
    y_min = x[:, 1].min()
    y_max = x[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    plt.title(kernel)

    plt.savefig('Task21'+kernel+'.png')


tick_label = ['linear', 'rbf','poly']
fig = plt.figure('Task 2.1, Accuracy')
x =list(range(len(accuracy_train)))
total_width, n = 0.8, 2
width = total_width / n
plt.bar(x,accuracy_train,width=width,label='train set')
for i in range(len(x)):
    x[i] = x[i] + width



plt.bar(x,accuracy_test, width=width,label='test set',tick_label=tick_label)
plt.legend()
plt.savefig('Task21accuracy.png')
plt.show()


