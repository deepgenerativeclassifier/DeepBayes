from __future__ import print_function

import numpy as np

def reshape_and_tile_images(array, shape=(28, 28), n_cols=None, margin=0, fill_val=None):
    if n_cols is None:
        n_cols = int(math.sqrt(array.shape[0]))
    n_rows = int(np.ceil(float(array.shape[0])/n_cols))
    if len(shape) == 2:
        order = 'C'
    else:
        order = 'F'

    def cell(i, j):
        ind = i*n_cols+j
        if i*n_cols+j < array.shape[0]:
            image = array[ind].reshape(*shape, order='C')
        else:
            image = np.zeros(shape)
        if margin > 0:
            tmp = np.ones([shape[0], 1]) * fill_val[ind]
            image = np.concatenate([tmp, image, tmp], axis=1)
            tmp = np.ones([1, shape[1] + 2]) * fill_val[ind]
            image = np.concatenate([tmp, image, tmp], axis=0)
        return image

    def row(i):
        return np.concatenate([cell(i, j) for j in range(n_cols)], axis=1)

    return np.concatenate([row(i) for i in range(n_rows)], axis=0)

def plot_images(images, shape, path, filename, n_rows = 10, margin=0, fill_val=None, color = True):
     # finally save to file
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    images = reshape_and_tile_images(images, shape, n_rows, margin, fill_val)
    if color:
        from matplotlib import cm
        plt.imsave(fname=path+filename+".png", arr=images, cmap=cm.Greys_r)
    else:
        plt.imsave(fname=path+filename+".png", arr=images, cmap='Greys')
    print("saving image to " + path + filename + ".png")
    plt.close()

