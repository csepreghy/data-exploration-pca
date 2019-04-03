import matplotlib.pyplot as plt
import numpy as np

from pca import pca


def mds(data, d, show_pc_plots):
  eigen_values, eigen_vectors, mean = pca(data, show_pc_plots=show_pc_plots)

  datamatrix = np.dot(np.array(eigen_vectors).T, data.T)

  return datamatrix[:d]
