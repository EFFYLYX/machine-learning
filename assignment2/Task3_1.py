import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import svm, metrics
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, fbeta_score
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

targets = np.array(iris_type(y))
X = np.array(x)
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
# print(titles)
for name, est in estimators:
    fig = plt.figure('Task 3.1, Figure ' + str(fignum))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(X)

    y_pred = est.predict(X)

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

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal weigh')
    ax.set_zlabel('Petal length')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12

    plt.savefig('Task31Figure' + str(fignum) + '.png')

    fignum = fignum + 1

score_3 = accuracy_score(y_pred_3, targets)
print('The accuracy of 3 clusters: ', score_3)

tick_label = ['1 cluster', '2 cluster', '3 cluster', '4 cluster', '5 cluster', '6 cluster']
fig = plt.figure('Task 3.1, Figure ' + str(fignum))
x = list(range(len(homogeneity)))
total_width, n = 0.8, 2
width = total_width / n

plt.bar(x, v_measure, width=width, label='v-measure', tick_label=tick_label)
plt.legend()
plt.savefig('Task31accuracy.png')
plt.show()
