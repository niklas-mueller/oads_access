import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import mat73
from scipy import io


def joint_plot(xs, ys, cs=[None], axtitles:list=[''], figtitle='', xlabels:list=[''], ylabels:list=[''], figsize=(15,10)):
    fig = plt.figure(figsize=figsize)

    if type(xs[0]) == list or type(xs[0]) == np.ndarray:
        n = len(xs)
    else:
        n = 1
        xs = [xs]
        ys = [ys]

    gs = GridSpec(4,4*n)

    for i in range(n):
        ax_scatter = fig.add_subplot(gs[1:4, i*4:i*4+3])
        ax_hist_x = fig.add_subplot(gs[0, i*4:i*4+3])
        ax_hist_y = fig.add_subplot(gs[1:4, i*4+3])
        
        ax_scatter.scatter(xs[i],ys[i], c=cs[i])
        ax_scatter.set_xlabel(xlabels[i])
        ax_scatter.set_ylabel(ylabels[i])
        ax_scatter.set_title(axtitles[i])

        ax_hist_x.hist(xs[i])
        ax_hist_y.hist(ys[i], orientation = 'horizontal')

    fig.suptitle(figtitle)
    fig.tight_layout()

    return fig

def imscatter_all(xs, ys, images, ax=None, zoom=1, title='', xlabel='', ylabel=''):
    if ax is None:
        ax = plt.gca()

    for x, y, img in zip(xs, ys, images):
        imscatter(x, y, img, zoom=zoom, ax=ax)
        ax.scatter(x, y)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except (TypeError, AttributeError):
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def get_random_crops(img: Image.Image, n_crops: int, crop_size: tuple):
    crops = []
    # this is in (height, width) while np.array(img).shape[:2] is in (width, height) !!!!!!!!!!!!
    size = img.size
    for index in range(n_crops):
        left = np.random.randint(low=0, high=size[0] - crop_size[0])
        upper = np.random.randint(low=0, high=size[1] - crop_size[1])
        right = left+crop_size[0]
        lower = upper+crop_size[1]
        box = (left, upper, right, lower)
        crop = img.crop(box)
        crops.append((crop, box, index))

    return crops


def get_bin_values(data, bins, min_count):
    # Get index of bin for each entry in data
    indices = np.digitize(x=data, bins=bins)

    index_array = np.array([i for i in range(len(data))])

    # Check whether each index would actually occur the number of times (min_count) otherwise update
    min_count = min(min_count, min(
        [len(index_array[indices == i]) for i in range(len(bins))]))

    # For each bin, gather the indices of the data array, shuffle them and only take min_count many to balance the size of the bins
    # This index array can then be used to get the images corresponding to the CE/SC values that were chosen, now having a uniform distribution
    bin_indices = np.array([np.random.permutation(index_array[indices == i])[
                           :min_count] for i in range(len(bins))])

    # Get the actual data points for each bin
    bin_values = np.array([data[bin_indices[i]] for i in range(len(bins))])
    return bin_indices, bin_values


def plot_images(images, fig=None, max_images: int = None, figsize=(15, 10), titles:list=None, cmap=None, axis_off:bool=True, orientation='landscape', custom_func_per_image=None):
    if max_images is None:
        max_images = len(images)

    if fig is None:
        fig = plt.figure(figsize=figsize)

    # if max_images % 2 == 1:
    #     n_rows = int(np.floor(np.sqrt(max_images)))
    #     n_cols = int(np.floor(np.sqrt(max_images)))
    # else:
    if orientation == 'landscape':
        n_rows = int(np.floor(np.sqrt(max_images)))
        n_cols = int(np.ceil(max_images / n_rows))
    else:
        n_cols = int(np.floor(np.sqrt(max_images)))
        n_rows = int(np.ceil(max_images / n_cols))

    index = 0
    for _ in range(n_rows):
        for _ in range(n_cols):
            if index >= max_images:
                break
            ax = fig.add_subplot(n_rows, n_cols, index+1)
            ax.imshow(images[index], cmap=cmap)
            if axis_off:
                ax.axis('off')
            if titles is not None and len(titles) > index:
                ax.set_title(titles[index])

            if custom_func_per_image is not None:
                custom_func_per_image(ax, index)
            index += 1

    fig.tight_layout()

    return fig


def loadmat(filepath: str, use_attrdict=True):
    """Combined functionality of scipy.io.loadmat and mat73.loadmat in order to load any .mat version into a python dictionary.



        Parameters
        ----------
        filepath: str
            Path to file.

        Returns
        ----------
        any | dict
            Loaded datastructure.

        Example
        ----------
        >>> data = config.loadmat(filepath)
    """
    try:
        data = mat73.loadmat(filepath, use_attrdict=use_attrdict)
        return data
        # data = {}
        # with h5py.File(filepath, 'r') as f:
        #     for k in f.keys():
        #         data[k] = dict(f.get(k))

        #     return data

    except (NotImplementedError, OSError, TypeError) as e:
        print(
            "Could not load mat file with mat73 - trying to load with scipy.io.loadmat!")
        # if version is <7.2
        data = io.loadmat(filepath)
        return data
