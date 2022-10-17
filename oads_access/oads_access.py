import os, sys, json
from PIL import Image
import rawpy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.sparse import data
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch import nn as nn
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms

class TestModel(nn.Module):
    def __init__(self, input_channels, output_channels, input_shape) -> None:
        super(TestModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(3,3), padding='valid'),       # ((100,100) - (3,3)) / 1 + 1   = (98,98)
            nn.Conv2d(32, 2, kernel_size=(3,3), padding='valid'),                    # ((98,98) - (3,3)) / 1 + 1     = (96,96)
            nn.Flatten(),
        )
        self.fc = nn.Linear(in_features=96*96*2, out_features=output_channels)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        z = self.layers(x)
        # print(f"After conv shape: {z.shape}")
        return self.fc(z)

class OADSImageDataset(Dataset):
    def __init__(self, data: list, class_index_mapping, transform=None, target_transform=None, device='cuda:0') -> None:
        super().__init__()

        # self.oads = oads
        # self.train_data, self.val_data, self.test_data = oads.get_train_val_test_split(use_crops=use_crops, min_size=min_size, max_size=max_size)
        self.data = data
        self.transform = transform
        self.target_transform = target_transform
        self.class_index_mapping = class_index_mapping
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = np.transpose(np.array(img), (0, 1, 2))
        label = label['classId']
        label = self.class_index_mapping[label]
        
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        img = img.to(self.device)
        # print(type(img), img.device)
        # label.to(self.device)
        return img, label

class OADS_Access():
    """OADS_Access

    OADS Access - API to interact with downloaded Open Amsterdam Dataset.

        Provides functionality to load and visualize images, annotations and other meta info.

        All preprocessing and formatting for using this dataset to train a DNN is contained.

    Example
    ----------
    >>> oads = OADS_Access(basedir='../data/oads')
    """
    def __init__(self, basedir: str):
        self.basedir = basedir

        self.datasets = [x for x in os.listdir(self.basedir) if os.path.isdir(os.path.join(self.basedir, x))]

        
        self.image_names = {
            name: [x for x in os.listdir(os.path.join(self.basedir, name, 'img'))] for name in self.datasets
        }

    def get_meta_info(self):
        """get_meta_info

        Load the meta information form the meta.json file contained in the supervisely format.

        Returns
        ----------
        str
            Meta information

        Example
        ----------
        >>> 
        """
        if not hasattr(self, 'meta'):
            with open(os.path.join(self.basedir, 'meta.json'), 'r') as f:
                meta = json.load(f)
            self.meta = meta

        return self.meta

    def get_class_mapping(self):
        """get_class_mapping

        Get a mapping from class id to class name for all classes in this dataset.

        Returns
        ----------
        dict
            Mapping from classId to classTitle

        Example
        ----------
        >>> 
        """
        return {x['id']: x['title'] for x in self.get_meta_info()['classes']}

    def get_annotation(self, dataset_name, image_name):
        """get_annotation

        Get the annotations for a specific dataset+image pair.

        Parameters
        ----------
        dataset_name: str
            Dataset name
        image_name: str
            Image file name

        Returns
        ----------
        str
            Annotations in json string format

        Example
        ----------
        >>> 
        """
        path = os.path.join(self.basedir, dataset_name, 'ann', f"{image_name}.json")

        if os.path.exists(path):
            with open(path, 'r') as f:
                content = json.load(f)
            return content
        else:
            return None

    def get_data_iterator(self, dataset_names=None, file_formats:list=None):
        """get_data_iterator

        Get a list of pairs of images and labels. 
        Mostly, images will have multiple annotations, therefore the second part of the tuple will be a list of dictionaries holding all the annotation information.

        Parameters
        ----------
        dataset_names: str, optional
            Dataset name, if None all datasets will be included, by default None

        Returns
        ----------
        list
            List of tuple of image and label where label is a list of dictionaries containing annotation information.

        Example
        ----------
        >>> data = oads.get_data_iterator() 
        """
        if dataset_names is None:
            dataset_names = self.datasets

        data = []
        for dataset_name in dataset_names:
            for image_name in self.image_names[dataset_name]:
                filename = os.path.join(self.basedir, dataset_name, 'img', image_name)

                fileformat = os.path.splitext(filename)[-1]
                if file_formats is not None and fileformat not in file_formats:
                    continue

                if fileformat == '.arw' or fileformat == '.ARW':
                    with rawpy.imread(filename) as raw:
                        img = raw.postprocess()

                else:
                    img = Image.open(filename)
                    

                label = self.get_annotation(dataset_name=dataset_name, image_name=image_name)
                tup = (img, label)
                data.append(tup)
        return data

    def get_maximum_annotation_size(self):
        """get_maximum_annotation_size

        Compute the smallest annotation box size that is equal or bigger than all annotation boxes in the dataset.

        Returns
        ----------
        Tuple
            Minimum height and width that still includes all other annotation boxes.

        Example
        ----------
        >>> 
        """
        data = self.get_data_iterator()

        _max_height = 0
        _max_width = 0

        for (img, label) in data:
            for obj in label['objects']:
                ((left, top), (right, bottom)) = obj['points']['exterior']
                height = bottom - top
                width = right - left
                _max_height = max(_max_height, height)
                _max_width = max(_max_width, width)

        return _max_height, _max_width

    def get_train_val_test_split(self, data_iterator:"list|np.ndarray" = None, val_size:float=0.1, test_size:float=0.1, 
                                    use_crops:bool=False, min_size:tuple=(0,0), max_size:tuple=None, file_formats:list=None):
        """get_train_val_test_split

        Split the data_iterator into train, validation and test sets.

        Parameters
        ----------
        data_iterator: list | np.ndarray, optional
            Data to split, by default None
        val_size: float, optional
            Proportion of total data to be used in validation set, by default 0.1
        test_size: float, optional
            Proportion of total data to be used in test set, by default 0.1
        use_crops: bool, optional
            If data_iterator is None, whether to use crop_iterator. If False, data_iterator will be loaded, by default False
        min_size: tuple, optional
            If data_iterator is None and use_crops is True, the minimum size (width, height) to crop the image to.

        Returns
        ----------
        tuple
            Sets of training data, validation data, testing data with given proportions

        Example
        ----------
        >>> 
        """
        if data_iterator is None:
            if use_crops:
                data_iterator = self.get_crop_iterator(min_size=min_size, max_size=max_size, file_formats=file_formats)
            else:
                data_iterator = self.get_data_iterator(file_formats=file_formats)

        train_data, test_data = train_test_split(data_iterator, test_size=val_size+test_size)
        test_data, val_data = train_test_split(test_data, test_size=test_size / (val_size+test_size))

        return train_data, val_data, test_data

    def get_crop_iterator(self, data_iterator:"list|np.ndarray" = None, min_size=(0,0), max_size:tuple=None, file_formats:list=None):
        if data_iterator is None:
            data_iterator = self.get_data_iterator(file_formats=file_formats)
        
        crop_iterator = []
        for (img, label) in data_iterator:
            for obj in label['objects']:
                crop = get_image_crop(img=img, object=obj, min_size=min_size, max_size=max_size)
                crop_iterator.append((crop, obj))

        return crop_iterator

# create crops from image
def get_image_crop(img:"np.ndarray|list|Image.Image", object:dict, min_size:tuple, max_size:tuple=None):
    """get_image_crop

    Using the annotation box object, crop the original image to the given size.

    Parameters
    ----------
    img: ndarray, list, PIL Image
        Image to be cropped
    object: dict
        Annotation Box object containing labelling information.
    min_size: Tuple(int, int)
        Minimum size (width, height) to crop the image to.
    max_size: tuple(int,int), optional
        Maximum size (width, height) to crop the image to, by default None

    Returns
    ----------
    ndarray, list, PIL Image
        Cropped image

    Example
    ----------
    >>> crop = get_image_crop(img=image, object=obj, min_size=(50, 50)) 
    """
    ((left, top), (right, bottom)) = object['points']['exterior']

    # Check if crop would be too small
    if right-left < min_size[0]:
        mid_point = left +(right - left) / 2
        left = mid_point - min_size[0] / 2
        right = mid_point + min_size[0] / 2
    if bottom-top < min_size[1]:
        mid_point = top + (bottom - top) / 2
        top = mid_point - min_size[1] / 2
        bottom = mid_point + min_size[1] / 2

    # Check if crop would be too big
    if not max_size is None:
        if right - left > max_size[0]:
            mid_point = left + (right - left) / 2
            left = mid_point - max_size[0] / 2
            right = mid_point + max_size[0] / 2
        if bottom - top > max_size[1]:
            mid_point = top + (bottom - top) / 2
            top = mid_point - max_size[1] / 2
            bottom = mid_point + max_size[1] / 2

    if type(img) == np.ndarray:
        img = Image.fromarray(img)
    crop = img.crop((left, top, right, bottom))
    return crop



def plot_crops_from_data_tuple(data_tuple, min_size=(0, 0), figsize=(18,30), max_size=None):
    """plot_crops_from_data_tuple

    For a given tuple of (image, label) (e.g., from get_data_iterator) plot all the crops corresponding to the annotated labels.

    Parameters
    ----------
    data_tuple: Tuple(PIL Image, dict)
        Data tuple with image and label information
    min_size: tuple, optional
        Minimum size to crop each annotation box to, by default (0, 0)
    figsize: tuple, optional
        Figsize for plt.subplots, by default (18,30)

    Returns
    ----------
    plt.figure
        Figure with all crops of the original image in subplots.

    Example
    ----------
    >>> fig = plot_crops_from_data_tuple(data[0]) 
    """
    img = data_tuple[0]
    label = data_tuple[1]
    
    _n = len(label['objects'])
    if _n == 0:
        print("No labels present for image!")
        return None
    _n = int(np.ceil(np.sqrt(_n)))
    fig, ax = plt.subplots(_n,_n, figsize=figsize)

    for axis, obj in zip(ax.flatten(), label['objects']):
        crop = get_image_crop(img, obj, min_size=min_size, max_size=max_size)
        axis.imshow(crop)
        axis.set_title(obj['classTitle'])
        axis.axis('off')

    return fig


def add_label_box_to_axis(label:dict, ax, color:str='r', add_title:bool=False):
    """add_label_box_to_axis

    Given a label dict and a axis add a rectangular box with the annotation dimensions to the axis.

    Parameters
    ----------
    label: dict
        Dictionary containing information about the annotation label + box
    ax: Axis
        Axis to plot on.
    color: str, optional
        Color for annotation box, by default 'r'
    add_title: bool, optional
        Whether to add the annotation label name as title, by default False

    Example
    ----------
    >>> 
    """
    if len(label['objects']) > 0:
        img_height, img_width = label['size']['height'], label['size']['width']
        for obj in label['objects']:
            if obj['geometryType'] == 'rectangle':
                ((left, top), (right, bottom)) = obj['points']['exterior']
                rec = Rectangle(xy=(left, top), height=bottom-top, width=right-left, fill=False, color=color)
                ax.add_patch(rec)

                if add_title:
                    ax.annotate(text=obj['classTitle'], xy=(left, top-10), fontsize='x-small')


def rgb_to_opponent_space(img, normalize=False):
    """rgb_to_opponent_space

    Convert a image in RBG space to color opponent space, i.e., Intensity, Blue-Yellow (BY) opponent, Red-Green (RG) opponent.

    Parameters
    ----------
    img: ndarray, list, PIL Image
        Image to be converted
    normalize: bool, optional
        Whether to normalize pixel values by the maximum, by default False

    Returns
    ----------
    ndarray
        Array of length 3, with image converted to Intensity, BY, RG opponent, respectively.

    Example
    ----------
    >>> 
    """
    o1 = 0.3 * img[:,:,0] + 0.58 * img[:,:,1] + 0.11 * img[:,:,2]   # Intensity/Luminance
    o2 = 0.25 * img[:,:,0] + 0.25 * img[:,:,1] - 0.5 * img[:,:,2]   # BY opponent
    o3 = 0.5 * img[:,:,0] - 0.5 * img[:,:,1]                        # RG opponent

    if normalize:
        ret = []
        for _x in [o1, o2, o3]:
            _max = _x.max()
            ret.append(_x / _max)
        return np.array(ret)

    return np.array([o1, o2, o3])


def plot_image_in_color_spaces(image:np.ndarray, figsize=(10,5), cmap_rgb:str=None, cmap_opponent:str=None, cmap_original:str=None):
    """plot_image_in_color_spaces

    Given a image in RBG space, plot the indiviual channels as well as the opponents in opponent color space.

    Parameters
    ----------
    image: np.ndarray
        Image to plot.
    figsize: tuple, optional
        Figsize for plt.subplots, by default (10,5)

    Returns
    ----------
    fig
        Figure containing image in different color space as well as original image.

    Example
    ----------
    >>> 
    """
    fig, ax = plt.subplots(2,4, figsize=figsize)

    for i, title in enumerate(['Red', 'Green', 'Blue']):
        temp = np.zeros(image.shape, dtype='uint8')
        temp[:,:,i] = image[:,:,i]
        ax[0][i].imshow(temp, cmap=cmap_rgb)
        ax[0][i].axis('off')
        ax[0][i].set_title(title)

    ax[0][3].imshow(image)
    ax[0][3].axis('off')
    ax[0][3].set_title('Original')

    opponents = rgb_to_opponent_space(image)

    for i, (img, title) in enumerate(zip(opponents, ['Intensity', 'BY', 'RG'])):
        ax[1][i].imshow(img, cmap=cmap_opponent)
        ax[1][i].set_title(title)
        ax[1][i].axis('off')

    ax[1][3].imshow(image, cmap=cmap_original)
    ax[1][3].axis('off')
    ax[1][3].set_title('Original')

    return fig

