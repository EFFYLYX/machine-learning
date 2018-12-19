import pandas as pd
import numpy as np
import Task2_234
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import svm, metrics
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import warnings
from sklearn.metrics import mean_squared_error
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




    # for i in range(0,len(y)):
    #     if y[i] == 'Iris-setosa':
    #             y[i] = 0
    #     if y[i] == 'Iris-versicolor':
    #             y[i] = 1
    #     if y[i]== 'Iris-virginica':
    #            y[i] =2
    # for  i in range(0,len(y)):
    #     if y[i][0] == 'Iris-setosa':
    #         y[i][0] = 0
    #     if y[i][0] == 'Iris-versicolor':
    #         y[i][0] = 1
    #     if y[i][0] == 'Iris-virginica':
    #         y[i][0] =2

    # it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    # return it[s]


# data = np.loadtxt('iris.data.txt', dtype=float, delimiter=',', converters={4: iris_type()})
# print(data)

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

def retreiveComponent(x, pc1):
    result =[]

    for list in x:
        r = []
        # print(list)
        r.append(list[pc1])
        # r.append(list[pc2])
        result.append(r)
    return np.array(result)


def plot(principalComponents,targets,pc_x):
    X = retreiveComponent(principalComponents, pc_x)
    C = 1.0  # SVM regularization parameter
    models = (svm.SVC(kernel='linear', C=C),
              svm.LinearSVC(C=C),
              svm.SVC(kernel='rbf', gamma=0.7, C=C),
              svm.SVC(kernel='poly', degree=3, C=C))
    models = (clf.fit(X, np.ravel(targets)) for clf in models)



    titles = ('SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel')


    # fig, sub = plt.subplots(2, 2)


    # print(X)


    # plt.subplots_adjust(wspace=0.4, hspace=0.4)

    print('=============================')
    print('SVM, PC', str(pc_x))

    for clf, title in zip(models,titles):

        y_pred = clf.predict(X)
        # print(y_pred)
        print('Accuracy (', title, '): ', accuracy_score(targets, y_pred))
        print('MSE (', title, '): ',mean_squared_error(targets,y_pred))
        # plt.clf()
        # plt.scatter(X[:, 0], c=targets, zorder=10, cmap=plt.cm.Paired,
        #             edgecolor='k', s=20)






    # X0, X1 = X[:, 0], X[:, 1]
    # xx, yy = meshgrid(X0, X1)
    #
    # print('==============================')
    # print('SVM, PC',str(pc_x),' PC'+str(pc_y))
    #
    #
    # for clf, title, ax in zip(models, titles, sub.flatten()):
    #     contours(ax, clf, xx, yy,
    #              cmap=plt.cm.coolwarm, alpha=0.8)
    #
    #     y_pred = clf.predict(X)
    #     # print(y_pred)
    #     print('Accuracy (',title, '): ', accuracy_score(targets, y_pred))
    #
    #     ax.scatter(X0, X1, c=targets, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    #     ax.set_xlim(xx.min(), xx.max())
    #     ax.set_ylim(yy.min(), yy.max())
    #     x_label='PC'+str(pc_x)
    #     y_label='PC'+str(pc_y)
    #
    #     ax.set_xlabel(x_label)
    #     ax.set_ylabel(y_label)
    #
    #     ax.set_xticks(())
    #     ax.set_yticks(())
    #     ax.set_title(title)



# data=pd.read_csv('iris.data.txt',header=None)
# x = data.drop(4,axis=1)
# y = data[4]
# #print(y)
#
# # plt.figure(figsize=(10,10))
# targets=iris_type(y)
#
# t = iris_type(y)
# targets = pd.DataFrame(targets)
# # print(targets)
# # print(y)
#
#
#
#
#
# # X = StandardScaler().fit_transform(x)
# #print(X)
# X = np.array(x)
#
# pca = PCA(n_components=3)
# principalComponents = pca.fit_transform(X)
[principalComponents,targets,t]=Task2_234.task2_2()

# print(pca.explained_variance_ratio_.sum())


warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)

plot(principalComponents,targets,0)
plot(principalComponents,targets,1)
plot(principalComponents,targets,2)
# plt.show()

#print(principalComponents)

# princialDf = pd.DataFrame(data=principalComponents,columns=['PC1','PC2','PC3'])
#
#
#
# princialDf['target']=targets
#
#
#
# #print(princialDf)
# finalDf = princialDf
# #print(finalDf)
# # colors = ['r', 'g', 'b']
# # plt.figure(figsize = (8, 8))
# # plt.xlabel('PC1', )
# # plt.ylabel('PC2')
# # plt.title("2 components's PCA", size=20)
# # targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
# flower_datas = [finalDf[finalDf['target']==0],finalDf[finalDf['target']==1],finalDf[finalDf['target']==2]]
# # for flower_data, color in zip(flower_datas, colors):
# #     plt.scatter(flower_data.PC1, flower_data.PC2 ,c=color, s=50)
# #     plt.legend(targets)
# #     plt.grid()
# # plt.show()
# #
# #print(flower_datas)
# fig = plt.figure()
#
#
# ax=Axes3D(fig, elev=-150, azim=110)
#
# # for flower_data in flower_datas:
# #     ax.scatter(flower_data[0],flower_data[1],flower_data[2],c=colors[0])
# #
# #     # ax.scatter(flower_data[0], flower_data[1], flower_data[2],c=color, s=50)
# #
# # ax.scatter(principalComponents[:,0],principalComponents[:,1],principalComponents[:,2],c=t[:])
# ax.scatter(principalComponents[:,0],principalComponents[:,1],principalComponents[:,2],c=t,cmap=plt.cm.Set1, edgecolor='k',s=40)
#
# ax.set_xlabel('1st eigenvector')
# ax.w_xaxis.set_ticklabels([])
# ax.set_ylabel('2nd eigenvector')
# ax.w_yaxis.set_ticklabels([])
# ax.set_zlabel('3rd eigenvector')
# ax.w_zaxis.set_ticklabels([])
# plt.show()
#

