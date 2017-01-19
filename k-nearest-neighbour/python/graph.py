import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def __init__():
    return plt

def draw_2d_plot (figure, width, height, x_label, y_label, clm_color, X,Y):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    plt.figure(figure, figsize=(width, height))
    plt.clf()

    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=clm_color, cmap=plt.cm.Paired)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

def draw_3d_plot(figure, title, width, height, x_label, y_label, z_label, clm_color, X,Y):

    fig = plt.figure(figure, figsize=(width, height))
    ax = Axes3D(fig, elev=-150, azim=110)

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=plt.cm.Paired)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel(y_label)
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel(z_label)
    ax.w_zaxis.set_ticklabels([])

def draw_pyplot():
    plt.show()