import pandas as pd
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from sklearn.externals.joblib.numpy_pickle_utils import xrange
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC,SVC
from sklearn import svm, metrics,datasets
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
import warnings
import Task2_234
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


def retreiveComponent(x, pc1,pc2):
    result =[]

    for list in x:
        r = []
        # print(list)
        r.append(list[pc1])
        r.append(list[pc2])
        result.append(r)
    return np.array(result)

def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label)

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



def plot(principalComponents,targets,pc_x,pc_y):
    # targets=np.array(targets)
    X = retreiveComponent(principalComponents, pc_x, pc_y)
    C = 1.0  # SVM regularization parameter
    models = (svm.SVC(kernel='linear', C=C),
              svm.LinearSVC(C=C),
              svm.SVC(kernel='rbf', gamma=0.7, C=C),
              svm.SVC(kernel='poly', degree=3, C=C))
    models = (clf.fit(X, targets) for clf in models)



    titles = ('SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel')


    fig, sub = plt.subplots(2, 2)



    plt.subplots_adjust(wspace=0.4, hspace=0.4)



    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = meshgrid(X0, X1)

    print('==============================')
    print('SVM, PC',str(pc_x),' PC'+str(pc_y))

    # for clf, title in zip(models, titles):
    #     y_pred = clf.predict(X)
    #     # print(y_pred)
    #     print('Accuracy (', title, '): ', accuracy_score(targets, y_pred))
    #     # print('MSE (', title, '): ', mean_squared_error(targets, y_pred))
    #     #


    for clf, title, ax in zip(models, titles, sub.flatten()):
        contours(ax, clf, xx, yy,
                 cmap=plt.cm.coolwarm, alpha=0.8)

        y_pred = clf.predict(X)
        # print(y_pred)
        print('Accuracy (',title, '): ', accuracy_score(targets, y_pred))

        ax.scatter(X0, X1, c=targets, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        x_label='PC'+str(pc_x)
        y_label='PC'+str(pc_y)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)



data=pd.read_csv('iris.data.txt',header=None)
x = data.drop(4,axis=1)
y = data[4]
# y= np.array(y)

targets=iris_type(y)
targets = np.array(targets)


# X = StandardScaler().fit_transform(x)
#print(X)
X=np.array(x)


pca = PCA(n_components=3)
principalComponents = pca.fit_transform(X)

print(pca.explained_variance_ratio_.sum())

# [principalComponents,targets,t]=Task2_2.get_PCA_result()
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)

plot(principalComponents,targets,0,1)
plot(principalComponents,targets,0,2)
plot(principalComponents,targets,1,2)


# plt.show()

