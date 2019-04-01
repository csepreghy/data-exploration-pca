from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from plotify import Plotify

plotify = Plotify()

plt.style.use('ggplot')

# input: datamatrix as loaded by numpy.loadtxt('dataset.txt')
# output:  1) the eigenvalues in a vector (numpy array) in descending order
#          2) the unit eigenvectors in a matrix (numpy array) with each column being an
#             eigenvector (in the same order as its associated eigenvalue)
#
# note: make sure the order of the eigenvalues (the projected variance) is decreasing, and
# the eigenvectors have the same order as their associated eigenvalues


def pca(X, show_pc_plots=False, with_std=False, with_mean=False):
  eigenvalues = []
  eigenvectors = []

  scaler = StandardScaler(with_std=with_std, with_mean=with_mean)

  X = scaler.fit_transform(X)

  mean = np.mean(X, axis=0)

  covariance_matrix = np.cov(X.T)

  eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

  key = np.argsort(eigenvalues)[::-1][:len(eigenvalues)]
  eigenvalues, eigenvectors = eigenvalues[key], eigenvectors[:, key]

  unit_eigenvectors = []

  # create the unit vectors out of eigenvectors
  for v in eigenvectors:
    unit_eigenvectors.append(v / np.linalg.norm(v))

  pca = PCA(n_components=len(eigenvalues))
  pca.fit(X)

  total = sum(eigenvalues)

  var_exp = [(i / total)*100 for i in sorted(eigenvalues, reverse=True)]

  total_cumulative_explain_varience = 0
  cumulative_explain_variences = []

  for i, eigen_value in enumerate(sorted(eigenvalues, reverse=True)):
    percentage = eigen_value / total * 100
    total_cumulative_explain_varience += percentage
    cumulative_explain_variences.append(total_cumulative_explain_varience)

  if show_pc_plots == True:
    xticks = []
    for _ in range(len(eigenvalues)):
      xticks.append('PC ' + str(_))

    plotify.bar(
        x_list=range(len(eigenvalues)),
        y_list=var_exp,
        title='Explained Variance by PC',
        ylabel='% Variance Explained',
        xlabel='PCs in order of descending variance',
        xticks=xticks,
        rotation=30
    )

    plotify.plot(
        y_list=cumulative_explain_variences,
        title='Cumulative Explained Variance',
        ylabel='% Variance Explained',
        xlabel='Number of Features',
    )

  return eigenvalues, unit_eigenvectors, mean
