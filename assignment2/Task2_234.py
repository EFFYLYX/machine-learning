import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import svm, metrics
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
def iris_type(t):
    targets =[]
    for i in range(0,len(t)):
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

def task2_2():
    data = pd.read_csv('iris.data.txt', header=None)
    x = data.drop(4, axis=1)
    y = data[4]


    targets = iris_type(y)
    targets = np.array(targets)
    t = iris_type(y)

    X = np.array(x)

    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(X)

    print('PCA variance ratio',pca.explained_variance_ratio_.sum())

    fig = plt.figure('Task 2.2')


    ax = Axes3D(fig, elev=-150, azim=110)

    ax.scatter(principalComponents[:, 0], principalComponents[:, 1], principalComponents[:, 2], c=t, cmap=plt.cm.Set1,
               edgecolor='k', s=40)
    ax.set_title('Task 2.2')
    ax.set_xlabel('1st eigenvector')
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel('2nd eigenvector')
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel('3rd eigenvector')
    ax.w_zaxis.set_ticklabels([])

    plt.savefig('Task22Figure'+'.png')


    return [principalComponents,targets,t]


def retreiveComponent(x, pc1):
    result =[]

    for list in x:
        r = []
        # print(list)
        r.append(list[pc1])
        # r.append(list[pc2])
        result.append(r)
    return np.array(result)

def svm_task2_3(principalComponents, targets, pc_x):
    X = retreiveComponent(principalComponents, pc_x)

    x_train, x_test, y_train, y_test = train_test_split(X, targets, test_size=0.20)

    models = (svm.SVC(kernel='linear',gamma=10),
              # svm.LinearSVC(C=C),
              svm.SVC(kernel='rbf', gamma=10),
              svm.SVC(kernel='poly', gamma=10))
    models = (clf.fit(X, np.ravel(targets)) for clf in models)

    titles = ('linear',
              # 'LinearSVC (linear kernel)',
              'rbf',
              'poly')


    accuracy = []
    accuracy_test = []
    mse = []


    print('=============================')
    print('SVM, PC', str(pc_x))

    for clf, title in zip(models,titles):
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_train)
        # atrain = accuracy_score(targets, y_pred)
        atrain = accuracy_score(y_train, y_pred)


        # m = accuracy_score(targets, y_pred)
        accuracy.append(atrain)
        # mse.append(m)
        atest = accuracy_score(y_test, clf.predict(x_test))
        accuracy_test.append(atest)

        # print(y_pred)
        print('Train dataset, Accuracy (', title, '): ', atrain)
        print('Test dataset, Accuracy (', title, '): ', atest)

        # print('MSE (', title, '): ', m)
        # plt.clf()
        # plt.scatter(X[:, 0], c=targets, zorder=10, cmap=plt.cm.Paired,
        #             edgecolor='k', s=20)

    tick_label = titles
    plt.savefig('Task23Figure' + str(pc_x) + '.png')

    fig = plt.figure('Task 2.3, Accuracy Train')

    x = list(range(len(accuracy)))
    total_width, n = 0.8, 3
    width = total_width / n
    # plt.bar(x, accuracy, width=width, label='homogeity')
    for i in range(len(x)):
        x[i] = x[i] + width * pc_x

    plt.bar(x, accuracy, width=width, label='PC' + str(pc_x) , tick_label=tick_label)
    plt.title('Task 2.3, Accuracy Train')
    plt.legend()
    plt.savefig('Task23accuracytrain.png')

    fig = plt.figure('Task 2.3, Accuracy Test')

    x = list(range(len(accuracy_test)))
    total_width, n = 0.8, 3
    width = total_width / n
    # plt.bar(x, accuracy, width=width, label='homogeity')
    for i in range(len(x)):
        x[i] = x[i] + width * pc_x

    plt.bar(x, accuracy_test, width=width, label='PC' + str(pc_x), tick_label=tick_label)
    plt.title('Task 2.3, Accuracy Test')
    plt.legend()
    plt.savefig('Task23accuracytest.png')

def retreiveComponent2(x, pc1,pc2):
    result =[]

    for list in x:
        r = []
        # print(list)
        r.append(list[pc1])
        r.append(list[pc2])
        result.append(r)
    return np.array(result)


def meshgrid(x,y,h=.02):
    x_min, x_max = x.min()-1, x.max()+1
    y_min, y_max = y.min()-1, y.max()+1
    xx, yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min,y_max,h))
    return xx,yy

def contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # print(Z)
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx,yy,Z,**params)



    return out

def svm_task2_4(principalComponents, targets, pc_x, pc_y):
    # targets=np.array(targets)
    X = retreiveComponent2(principalComponents, pc_x, pc_y)
    x_train, x_test, y_train, y_test = train_test_split(X, targets, test_size=0.20)

    # C = 1.0  # SVM regularization parameter
    # models = (svm.SVC(kernel='linear', C=C),
    #           svm.LinearSVC(C=C),
    #           svm.SVC(kernel='rbf', gamma=0.7, C=C),
    #           svm.SVC(kernel='poly', degree=3, C=C))
    models = (svm.SVC(kernel='linear', gamma=10),
              # svm.LinearSVC(C=C),
              svm.SVC(kernel='rbf', gamma=10),
              svm.SVC(kernel='poly', gamma=10))
    models = (clf.fit(X, targets) for clf in models)



    titles = ('linear',
              # 'LinearSVC (linear kernel)',
              'rbf',
              'poly')



    fig, sub = plt.subplots(2, 2)



    plt.subplots_adjust(wspace=0.4, hspace=0.4)



    X0, X1 = X[:, 0], X[:, 1]
    # xx, yy = meshgrid(X0, X1)

    print('==============================')
    print('SVM, PC',str(pc_x),' PC'+str(pc_y))

    accuracy =[]
    accuracy_test =[]

    mse = []

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = meshgrid(X0, X1)
    for clf, title, ax in zip(models, titles, sub.flatten()):
        contours(ax, clf, xx, yy,
                 cmap=plt.cm.coolwarm, alpha=0.8)

        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_train)

        # atrain = accuracy_score(targets,y_pred)
        atrain = accuracy_score(y_train,y_pred)

        # m = accuracy_score(targets,y_pred)
        accuracy.append(atrain)
        # mse.append(m)

        # print(y_pred)
        atest = accuracy_score(y_test, clf.predict(x_test))
        accuracy_test.append(atest)

        # print(y_pred)
        print('Train dataset, Accuracy (', title, '): ', atrain)
        print('Test dataset, Accuracy (', title, '): ', atest)
        # print('MSE (', title, '): ',m)


        ax.scatter(X0, X1, c=targets, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        x_label='PC'+str(pc_x)
        y_label='PC'+str(pc_y)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        ax.set_xticks(())
        ax.set_yticks(())

        t = 'Task 2.4 \n'+title
        ax.set_title(t)

    plt.savefig('Task24Figure' + str(pc_x)+str(pc_y) + '.png')


    tick_label = titles



    fig = plt.figure('Task 2.4, Accuracy Train')


    x = list(range(len(accuracy)))
    total_width, n = 0.8, 3
    width = total_width / n
    # plt.bar(x, accuracy, width=width, label='homogeity')
    for i in range(len(x)):
        x[i] = x[i] + width*(pc_y+pc_x)


    plt.bar(x, accuracy, width=width, label='PC'+str(pc_x)+' PC'+str(pc_y), tick_label=tick_label)
    plt.legend()
    plt.title('Task 2.4, Accuracy Train')
    plt.savefig('Task24accuracytrain.png')

    fig = plt.figure('Task 2.4, Accuracy Test')

    x = list(range(len(accuracy_test)))
    total_width, n = 0.8, 3
    width = total_width / n
    # plt.bar(x, accuracy, width=width, label='homogeity')
    for i in range(len(x)):
        x[i] = x[i] + width * (pc_y + pc_x)

    plt.bar(x, accuracy_test, width=width, label='PC' + str(pc_x) + ' PC' + str(pc_y), tick_label=tick_label)
    plt.legend()
    plt.title('Task 2.4, Accuracy Test')
    plt.savefig('Task24accuracytest.png')







def task2_3(principalComponents,targets):


    svm_task2_3(principalComponents, targets, 0)
    svm_task2_3(principalComponents, targets, 1)
    svm_task2_3(principalComponents, targets, 2)

def task2_4(principalComponents,targets):


    svm_task2_4(principalComponents, targets, 0, 1)
    svm_task2_4(principalComponents, targets, 0, 2)
    svm_task2_4(principalComponents, targets, 1, 2)

fignum =1


warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)

[principalComponents,targets,t]=task2_2()
#


print()
print('Task 2.3 .......')

task2_3(principalComponents,targets)


print()
print('Task 2.4 .......')
task2_4(principalComponents,targets)


plt.show()
