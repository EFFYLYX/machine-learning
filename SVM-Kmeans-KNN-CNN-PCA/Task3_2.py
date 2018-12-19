import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import svm, metrics
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.cluster import v_measure_score, homogeneity_score


def iris_type(t):
    targets = []
    for i in range(0, len(t)):
        # print(row)
        # print(t.iloc[i])

        if t.iloc[i] == 'Iris-setosa':
            # print(t.iloc[i][4])
            # t.iloc[i] = 0
            targets.append(0)
        if t.iloc[i] == 'Iris-versicolor':
            # t.iloc[i] = 1
            targets.append(1)
        if t.iloc[i] == 'Iris-virginica':
            # t.iloc[i] = 2
            targets.append(2)
    return targets


data = pd.read_csv('iris.data.txt', header=None)
x = data.drop(4, axis=1)
y = data[4]

targets = iris_type(y)

t = iris_type(y)

targets = np.array(iris_type(y))


X = np.array(x)

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(X)
print('PCA variance ratio', pca.explained_variance_ratio_.sum())

estimators = []
for i in range(1, 7):
    e1 = 'k-means ' + str(i) + ' clusters'
    estimators.append((e1, KMeans(n_clusters=i)))

titles = []
fignum = 1

v_measure = []
homogeneity = []

y_pred_3 = []

for i in range(1, 7):
    title = str(i) + ' clusters'
    titles.append(title)

for name, est in estimators:

    fig = plt.figure('Task 3.2, Figure ' + str(fignum))

    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(principalComponents)

    y_pred = est.predict(principalComponents)

    v_m = v_measure_score(targets, y_pred)
    v_measure.append(v_m)
    h = homogeneity_score(targets, y_pred)
    homogeneity.append(h)

    print()
    print('v measure (', name, '): ', v_m)

    print()

    if name == 'k-means ' + str(3) + ' clusters':
        y_pred_3 = y_pred

    labels = est.labels_

    ax.scatter(principalComponents[:, 0], principalComponents[:, 1], principalComponents[:, 2],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('1st eigenvector')
    ax.set_ylabel('2nd eigenvector')
    ax.set_zlabel('3rd eigenvector')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12

    plt.savefig('Task32Figure' + str(fignum) + '.png')

    fignum = fignum + 1


score_3 = accuracy_score(y_pred_3, targets)
print('The accuracy of 3 clusters: ', score_3)

tick_label = ['1 cluster', '2 cluster', '3 cluster', '4 cluster', '5 cluster', '6 cluster']
fig = plt.figure('Task 3.2, Figure ' + str(fignum))
x = list(range(len(homogeneity)))
total_width, n = 0.8, 2
width = total_width / n


plt.bar(x, v_measure, width=width, label='v-measure', tick_label=tick_label)
plt.legend()
plt.savefig('Task32accuracy.png')
plt.show()
