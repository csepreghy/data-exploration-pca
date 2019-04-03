import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from plotify import Plotify
from pca import pca
from mds import mds
plotify = Plotify()

diatoms = np.loadtxt('diatoms.txt')

# Plot one cell

x = diatoms[0][0::2]
y = diatoms[0][1::2]

fig, ax = plotify.get_figax()

plotify.scatter_plot(
    x_list=[x],
    y_list=[y],
    tickfrequencyone=False,
    xlabel='x',
    ylabel='y',
    title='First cell of the dataset',
    ax=ax,
    show_plot=False,
    equal_axis=True,
    legend_labels=['Landmark points of the cell']
)

ax.plot(x, y, c='#4FB99F')
plt.show()

fig2, ax2 = plotify.get_figax()

# for diatom in diatoms:
#   plotify.scatter_plot(
#       x_list=[diatom[0::2]],
#       y_list=[diatom[1::2]],
#       tickfrequencyone=False,
#       xlabel='x',
#       ylabel='y',
#       title='All cells on top of each other',
#       show_plot=False,
#       ax=ax2,
#       alpha=0.04,
#       equal_axis=True,
#       legend_labels=['Landmark points of the cell']
#   )

# plt.show()

oranges = plt.get_cmap('OrRd')

# eigenvalues, eigenvectors, mean = pca(diatoms)

scikit_pca = PCA()
scikit_pca.fit(diatoms)

cells = {}

for i in range(3):
  cells['pc' + str(i)] = [
    scikit_pca.mean_ - 2 * np.sqrt(scikit_pca.explained_variance_[i]) * scikit_pca.components_[i],
    scikit_pca.mean_ - np.sqrt(scikit_pca.explained_variance_[i]) * scikit_pca.components_[i],
    scikit_pca.mean_,
    scikit_pca.mean_ + np.sqrt(scikit_pca.explained_variance_[i]) * scikit_pca.components_[i],
    scikit_pca.mean_ + 2 * np.sqrt(scikit_pca.explained_variance_[i]) * scikit_pca.components_[i]
  ]

for key, cell_list in cells.items():
  fig, ax = plotify.get_figax()

  for i, cell in enumerate(cell_list):
    plt.fill(cell[0::2], cell[1::2], fill=0, c=oranges(i/len(cell_list)))

  plt.show()

# Exercise 3

toydata = np.loadtxt('pca_toydata.txt')

datamatrix = mds(toydata, 2, show_pc_plots=False)

plotify.scatter_plot(
    x_list=[datamatrix[0, :]],
    y_list=[datamatrix[1, :]],
    tickfrequencyone=False,
    xlabel='PC 1',
    ylabel='PC 2',
    title='First 2 PCs of the Toy dataset'
)

# toydata_shorter = toydata[:,-2]
print('toydata.shape', toydata.shape)

toydata_shorter = toydata[:-2]

print('toydata_shorter.shape', toydata.shape)

datamatrix = mds(toydata_shorter, 2, show_pc_plots=False)

plotify.scatter_plot(
    x_list=[datamatrix[0, :]],
    y_list=[datamatrix[1, :]],
    tickfrequencyone=False,
    xlabel='PC 1',
    ylabel='PC 2',
    title='First 2 PCs of the Toy dataset'
)


# Exercise 4

weed_crop_train = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
weed_crop_train_X = weed_crop_train[:, :-1] #

weed_crop_test = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')

eigenvalues, eigenvectors, mean = pca(weed_crop_train_X)

datamatrix = mds(weed_crop_train_X, 2, show_pc_plots=True)



X_train = weed_crop_train_X
y_train = weed_crop_train[:, -1]

X_test = weed_crop_test[:, :-1]
y_test = weed_crop_test[:, -1]

# Not optimal, but was required of us for grading purposes
starting_point = np.vstack((X_train[0, ], X_train[1, ]))

kmeans = KMeans(n_clusters=2, n_init=1, init=starting_point, algorithm='full').fit(X_train)

labels = kmeans.labels_

print('kmeans.cluster_centers_', kmeans.cluster_centers_)

datamatrix3d = mds(weed_crop_train_X, 3, show_pc_plots=False)

X = datamatrix3d
y = y_train

colors = []

for l in labels:
  if l == 0:
    colors.append('#4FB99F')
  elif l == 1:
    colors.append('#F2B134')

fig2, ax2d = plt.subplots(figsize=(8, 6))

fig2.patch.set_facecolor('#1C2024')
ax2d.set_facecolor('#1C2024')

ax2d.set_title('KMeans clusters in 2D (Pesticide Dataset)')
ax2d.set_xlabel('PC 1')
ax2d.set_ylabel('PC 2')
ax2d.scatter(X[0, :], X[1, :], c=colors, edgecolor='#333333', alpha=0.8)


cluster_centers = np.dot(np.array(eigenvectors).T, kmeans.cluster_centers_.T)
ax2d.scatter(cluster_centers[0], cluster_centers[1], c='red')

plt.show()

fig = plt.figure(figsize=(8, 6))
fig.patch.set_facecolor('#1C2024')
ax3d = Axes3D(fig, rect=[0, 0, 1, 1])
ax3d.set_facecolor('#1C2024')

cluster_centers3d = np.dot(np.array(eigenvectors).T, kmeans.cluster_centers_.T)
ax3d.scatter(X[0, :], X[1, :], X[2, :], c=colors, edgecolor='#333333', alpha=0.2)
ax3d.scatter(
  cluster_centers3d[0],
  cluster_centers3d[1],
  cluster_centers3d[2],
  c='red'
)

ax3d.w_xaxis.set_ticklabels([])
ax3d.w_yaxis.set_ticklabels([])
ax3d.w_zaxis.set_ticklabels([])
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False
ax3d.set_xlabel('PC 1')
ax3d.set_ylabel('PC 2')
ax3d.set_zlabel('PC 3')
ax3d.set_title('KMeans clusters in 3D (Pesticide Dataset)')
ax3d.dist = 12

plt.show()


