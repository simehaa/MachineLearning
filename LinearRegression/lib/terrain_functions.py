import numpy as np
from imageio import imread
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

# Load the terrain
DATA_DIR = "../data/"

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)  # create the path if it does not exist

# loadables: new_york.tif, oslo_fjorden.tif, key_west.tif, dead_sea.tif
terrain1 = np.asarray(imread(DATA_DIR + "dead_sea.tif"))
ny, nx = terrain1.shape
x = np.linspace(0, nx - 1, nx, dtype=np.int32)
y = np.linspace(0, ny - 1, ny, dtype=np.int32)
X, Y = np.meshgrid(x, y)

# Show the terrain
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# plot the original data
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel("height")
surface = ax.plot_surface(X, Y, terrain1, cmap=mpl.cm.coolwarm)
fig.colorbar(surface, shrink=0.5)
plt.show()
