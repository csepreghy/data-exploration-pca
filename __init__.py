import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from plotify import Plotify
from pca import pca

diatoms = np.loadtxt('diatoms.txt')

plotify = Plotify()

# Plot one cell

x = diatoms[0][0::2]
y = diatoms[0][1::2]

# plotify.scatter_plot(
#     x_list=[x],
#     y_list=[y],
#     tickfrequencyone=False,
#     xlabel='x',
#     ylabel='y',
#     title='First cell of the dataset',
#     show_plot=True,
#     equal_axis=True,
#     legend_labels=['Landmark points of the cell']
# )

# fig, ax = plotify.get_figax()

# for diatom in diatoms:
#   plotify.scatter_plot(
#       x_list=[diatom[0::2]],
#       y_list=[diatom[1::2]],
#       tickfrequencyone=False,
#       xlabel='x',
#       ylabel='y',
#       title='All cells on top of each other',
#       show_plot=False,
#       ax=ax,
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