from fileinput import filename
import multiprocessing
import os
import json
from PIL import Image
from matplotlib import test
import rawpy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch import nn as nn
import tqdm




class OADS_Access():
    """OADS_Access

    OADS Access - API to interact with the downloaded Open Amsterdam Dataset.

        Provides functionality to load and visualize images, annotations and other meta info.

        All preprocessing and formatting for using this dataset to train a DNN is contained.

    Example
    ----------
    >>> oads = OADS_Access(basedir='../data/oads')
    """

    def __init__(self, basedir: str, file_formats:list=None, min_size_crops:tuple=(0,0), max_size_crops:tuple=None, exclude_oversized_crops:bool=False):
        self.basedir = basedir
        self.file_formats = file_formats
        self.min_size_crops = min_size_crops        
        self.max_size_crops = max_size_crops
        self.exclude_oversized_crops = exclude_oversized_crops

        if not os.path.exists(os.path.join(basedir, 'meta.json')):
            self.has_raw_images = True
            self.img_dir = os.path.join(self.basedir, 'oads_arw')
            self.basedir = os.path.join(self.basedir, 'oads_jpeg')
        else:
            self.has_raw_images = False

        self.check_has_crop_files()

        self.datasets = [x for x in os.listdir(
            self.basedir) if os.path.isdir(os.path.join(self.basedir, x)) and x != 'crops'] 

        self.image_names = {
            name: [x for x in os.listdir(os.path.join(self.basedir, name, 'img'))] for name in self.datasets
        }

    def check_has_crop_files(self):
        if os.path.exists(os.path.join(self.basedir, 'crops')):
            self.has_crops_files = True
            self.crop_files_names = {
                class_name: [
                    x for x in os.listdir(os.path.join(self.basedir, 'crops', class_name)) if not x.endswith('.json')
                ] for class_name in os.listdir(os.path.join(self.basedir, 'crops')) if os.path.isdir(os.path.join(self.basedir, 'crops'))
            } 
        else:
            self.has_crops_files = False

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

    def get_annotation(self, dataset_name, image_name, is_raw=False):
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
        path = os.path.join(self.basedir, dataset_name,
                            'ann', f"{image_name}.json")

        if os.path.exists(path):
            with open(path, 'r') as f:
                content = json.load(f)
                content['is_raw'] = is_raw
            return content
        else:
            return None

    def load_images_from_dataset(self, args: list):
        dataset_name = args[0]
        max_number_images = args[1]
        use_crops = args[2]

        if max_number_images is not None:
            mask = np.where(np.arange(len(self.image_names[dataset_name])) >= max_number_images, False, True)
            np.random.shuffle(mask)
            image_name_iter = np.array(self.image_names[dataset_name])[mask]
        else:
            image_name_iter = self.image_names[dataset_name]

        data = []
        image_name: str
        
        for image_name in image_name_iter:
            tup = self.load_image(dataset_name=dataset_name, image_name=image_name)
            if tup is not None:
                if use_crops:
                    crops = self.make_and_save_crops_from_image(img=tup[0], label=tup[1])
                    data.extend(crops)
                else:
                    data.append(tup)

        return data

    def load_image(self, dataset_name:str, image_name:str):
        if self.has_raw_images:
            filename = os.path.join(
                self.img_dir, 'ARW', f"{image_name.split('.')[0]}.ARW")
            if not os.path.exists(filename):
                print(f"File doesn't exists! Skipping: {filename}")
                # continue
                return None
        else:
            filename = os.path.join(
                self.basedir, dataset_name, 'img', image_name)

        fileformat = os.path.splitext(filename)[-1]
        if self.file_formats is not None and fileformat not in self.file_formats:
            # continue
            return None

        is_raw = False
        if fileformat == '.arw' or fileformat == '.ARW':
            is_raw = True
            image_name = f"{image_name.split('.')[0]}.jpg"
            with rawpy.imread(filename) as raw:
                img = raw.postprocess()

        else:
            img = Image.open(filename)

        label = self.get_annotation(
            dataset_name=dataset_name, image_name=image_name, is_raw=is_raw)
        tup = (img, label)

        return tup

    def load_crop(self, class_name:str, image_name:str):
        fileformat = os.path.splitext(image_name)[-1]
        if self.file_formats is not None and fileformat not in self.file_formats:
            return None

        path = os.path.join(self.basedir, 'crops', class_name, image_name)
        if image_name.endswith('.npy'):
            crop = np.load(path)
        else:
            crop = Image.open(path)

        with open(os.path.join(self.basedir, 'crops', class_name, f"{image_name}.json"), 'r') as f:
            label = json.load(f)

        return (crop, label)

    def get_data_iterator(self, dataset_names=None, use_crops:bool=False, max_number_images: int = None):
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

        if max_number_images is not None:
            n_per_dataset = int(max_number_images / len(dataset_names))
        else:
            n_per_dataset = None

        max_number_images = [n_per_dataset for _ in range(len(dataset_names))]
        use_crops = [use_crops for _ in range(len(dataset_names))]

        # self.file_formats = file_formats

        # data = []
        with multiprocessing.Pool() as pool:
            data = list(tqdm.tqdm(pool.imap(self.load_images_from_dataset, zip(dataset_names, max_number_images, use_crops)),
                                  total=len(dataset_names)))
        data = [x for dataset in data for x in dataset]

        return data

    def get_annotation_size(self, obj: dict, is_raw):
        ((left, top), (right, bottom)), _, _ = get_annotation_dimensions(
            obj, is_raw=is_raw)
        height = bottom - top
        width = right - left

        return height, width

    def get_maximum_annotation_size(self, data_iterator: list = None):
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
        if data_iterator is None:
            data_iterator = self.get_data_iterator()

        _max_height = 0
        _max_width = 0

        for (_, label) in data_iterator:
            for obj in label['objects']:
                height, width = self.get_annotation_size(
                    obj, is_raw=label['is_raw'])
                _max_height = max(_max_height, height)
                _max_width = max(_max_width, width)

        return _max_height, _max_width

    def get_train_val_test_split(self, data_iterator: "list|np.ndarray" = None, val_size: float = 0.1, test_size: float = 0.1,
                                 use_crops: bool = False, file_formats: list = None, max_number_images: int = None):
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
                data_iterator = self.get_crop_iterator(file_formats=file_formats, max_number_images=max_number_images)
            else:
                data_iterator = self.get_data_iterator(
                    file_formats=file_formats, max_number_images=max_number_images)

        train_data, test_data = train_test_split(
            data_iterator, test_size=val_size+test_size)
        test_data, val_data = train_test_split(
            test_data, test_size=test_size / (val_size+test_size))

        return train_data, val_data, test_data

    def get_train_val_test_split_indices(self, use_crops:bool, val_size:float=0.1, test_size:float=0.1, remove_duplicates:bool=True):
        if use_crops:
            image_ids = [(class_name, image_name) for class_name, image_names in self.crop_files_names.items() for image_name in image_names]
        else:
            image_ids = [(dataset_name, image_name) for dataset_name, image_names in self.image_names.items() for image_name in image_names]

        if remove_duplicates:
            image_ids = list(set(image_ids))
        train_ids, test_ids = train_test_split(image_ids, test_size=val_size+test_size)
        test_ids, val_ids = train_test_split(test_ids, test_size=test_size / (val_size+test_size))

        return train_ids, val_ids, test_ids


    def load_crop_class(self, args:tuple):
        class_name, file_names = args
        crops = []
        for file_name in file_names:
            tup = self.load_crop(class_name=class_name, image_name=file_name)
            if tup is not None:
                crops.append(tup)

        return crops

    def _prepare_crops_dataset(self, args):
        dataset_name, convert_to_opponent_space = args
        for image_name in self.image_names[dataset_name]:
            tup = self.load_image(dataset_name=dataset_name, image_name=image_name)
            if tup is not None:
                (img, label) = tup
                if convert_to_opponent_space:
                    img = rgb_to_opponent_space(np.array(img))
                _ = self.make_and_save_crops_from_image(img=img, label=label, is_opponent_space=convert_to_opponent_space, save_files=True)

    def prepare_crops(self, dataset_names:list=None, convert_to_opponent_space:bool=False):
        if dataset_names is None:
            dataset_names = self.datasets

        args = zip(dataset_names, [convert_to_opponent_space for _ in range(len(dataset_names))])

        with multiprocessing.Pool() as pool:
            print(f"Number of processes: {pool._processes}")
            _ = list(tqdm.tqdm(pool.imap(self._prepare_crops_dataset, args), total=len(dataset_names)))

        self.check_has_crop_files()
        

    def get_crop_iterator(self, data_iterator: "list|np.ndarray" = None, convert_to_opponent_space:bool=False,
                          max_number_images: int = None, save_files:bool=False, recompute_crops:bool=False):

        crop_iterator = []
        if not recompute_crops and self.has_crops_files:
            with multiprocessing.Pool() as pool:
                results = list(tqdm.tqdm(pool.imap(self.load_crop_class, self.crop_files_names.items()), total=len(self.crop_files_names.items())))
            
            crop_iterator = [x for dataset in results for x in dataset]

        else:
        
            if data_iterator is None:
                crop_iterator = self.get_data_iterator( max_number_images=max_number_images, use_crops=True)
            else:
                for (img, label) in data_iterator:
                    if convert_to_opponent_space:
                        img = rgb_to_opponent_space(np.array(img))
                    crops = self.make_and_save_crops_from_image(img=img, label=label, is_opponent_space=convert_to_opponent_space, save_files=save_files)
                    crop_iterator.extend(crops)

        return crop_iterator

    def make_and_save_crops_from_image(self, img, label, is_opponent_space:bool=False, save_files:bool=False):
        crops = []
        for obj in label['objects']:
            if self.exclude_oversized_crops:
                height, width = self.get_annotation_size(
                            obj, is_raw=label['is_raw'])
                if height > self.max_size_crops[0] or width > self.max_size_crops[1]:
                    continue
            crop = get_image_crop(
                        img=img, object=obj, min_size=self.min_size_crops, max_size=self.max_size_crops, is_raw=label['is_raw'], is_opponent_space=is_opponent_space)
            crops.append((crop, obj))

            if save_files:
                if is_opponent_space:
                    fileending = '.npy'
                else:
                    fileending = '.jpg'
                filedir = os.path.join(self.basedir, 'crops', str(obj['classId']))
                filename = os.path.join(filedir, f"{str(obj['id'])}{fileending}")
                os.makedirs(filedir, exist_ok=True)
                if is_opponent_space:
                    np.save(arr=crop, file=filename, allow_pickle=False)
                else:
                    crop.save(fp=filename)
                with open(f"{filename}.json", 'w') as f:
                    json.dump(obj, fp=f)

        return crops

    def get_dataset_stats(self):
        means, stds = [], []

        results = self.apply_per_image(lambda x: (np.mean(np.array(x[0]), axis=(0, 1)), np.std(np.array(x[0]), axis=(0, 1))))

        for _, images in results.items():
            for _, (m, s) in images.items():
                means.append(m)
                stds.append(s)
        
        # for img, _ in data_iterator:
        #     img_np = np.array(img)
        #     mean = np.mean(img_np, axis=(0, 1))
        #     std = np.std(img_np, axis=(0, 1))
        #     means.append(mean)
        #     stds.append(std)

        return np.array(means), np.array(stds)

    def plot_image_size_distribution(self, use_crops: bool, data_iterator:list=None, file_formats: list = None, figsize: tuple = (10, 5)):

        # Make scatter plot of x and y sizes and each images as dot
        # train_data, val_data, test_data = self.get_train_val_test_split(
        #     use_crops=use_crops, file_formats=file_formats, exclude_oversized_crops=False)
        results = self.apply_per_image(get_image_size, max_number_images=10000)
        
        height_sizes = []
        width_sizes = []
        for _, images in results.items():
            for _, (height, width) in images.items():
                height_sizes.append(height)
                width_sizes.append(width)
        # for img, _ in np.concatenate((train_data, val_data, test_data)):
        #     (height, width, c) = np.array(img).shape

        #     height_sizes.append(height)
        #     width_sizes.append(width)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.scatter(height_sizes, width_sizes)
        print(len(height_sizes), len(width_sizes))
        ax.set_xlabel('Image height')
        ax.set_ylabel('Image width')

        return fig

    def apply_per_crop(self, custom_function, max_number_crops:int=None):
        results = {}

        crop_counter = 0
        for dataset_name in self.crop_files_names.keys():
            results[dataset_name] = {}
            for crop_name in self.crop_files_names[dataset_name]:
                tup = self.load_crop(class_name=dataset_name, image_name=crop_name)
                if tup is not None:
                    if max_number_crops is not None and crop_counter >= max_number_crops:
                        return results
                    results[dataset_name][crop_name] = custom_function(tup)
                    crop_counter += 1

        return results

    def apply_per_image(self, custom_function, max_number_images:int=None):
        results = {}
        image_counter = 0
        for dataset_name in self.datasets:
            results[dataset_name] = {}
            for image_name in self.image_names[dataset_name]:
                tup = self.load_image(dataset_name=dataset_name, image_name=image_name)
                if tup is not None:
                    if max_number_images is not None and image_counter >= max_number_images:
                        return results
                    results[dataset_name][image_name] = custom_function(tup)
                    image_counter += 1

        return results

    def apply_custom_data_augmentation(self, data_iterator: list, augmentation_function):
        return list(map(augmentation_function, data_iterator))

    def rotate_image_90cc(self, data_tuple):
        img = Image.fromarray(np.rot90(np.array(data_tuple[0])))
        label = data_tuple[1]
        label['points']['exterior'] = list(
            np.rot90(data_tuple[1]['points']['exterior']))
        return (img, label)

    def rotate_image_90c(self, data_tuple):
        img = Image.fromarray(np.rot90(np.array(data_tuple[0]), k=3))
        label = data_tuple[1]
        label['points']['exterior'] = list(
            np.rot90(data_tuple[1]['points']['exterior']))
        return (img, label)

    def rotate_image_180(self, data_tuple):
        img = Image.fromarray(np.rot90(np.array(data_tuple[0]), k=2))
        label = data_tuple[1]
        label['points']['exterior'] = list(
            np.rot90(data_tuple[1]['points']['exterior']))
        return (img, label)

    def flip_image_horizontally(self, data_tuple):
        img = Image.fromarray(np.array(data_tuple[0])[::-1, :, :])
        label = data_tuple[1]
        label['points']['exterior'] = list(
            np.array(data_tuple[1]['points']['exterior'])[::-1, :, :])
        return (img, label)

    def flip_image_vertically(self, data_tuple):
        img = Image.fromarray(np.array(data_tuple[0])[::, ::-1, :])
        label = data_tuple[1]
        label['points']['exterior'] = list(
            np.array(data_tuple[1]['points']['exterior'])[::, ::-1, :])
        return (img, label)


def get_image_size(tup):
    img, _ = tup
    (height, width, c) = np.array(img).shape
    return height, width

def get_annotation_dimensions(obj: dict, is_raw, min_size: tuple = None, max_size: tuple = None):
    ((left, top), (right, bottom)) = obj['points']['exterior']
    if is_raw:
        left = left * 4
        top = top * 4
        right = right * 4
        bottom = bottom * 4
        if min_size is not None:
            min_size = tuple(np.multiply(min_size, 4))
        if max_size is not None:
            max_size = tuple(np.multiply(max_size, 4))

    return ((left, top), (right, bottom)), min_size, max_size

# create crops from image


def get_image_crop(img: "np.ndarray|list|Image.Image", object: dict, min_size: tuple, max_size: tuple = None, is_raw: bool = False, is_opponent_space:bool=False):
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
    ((left, top), (right, bottom)), min_size, max_size = get_annotation_dimensions(
        object, is_raw=is_raw, min_size=min_size, max_size=max_size)

    # Check if crop would be too small
    if right-left < min_size[0]:
        mid_point = left + (right - left) / 2
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

    if is_opponent_space:
        crop = []
        for _x in img:
            crop.append(np.array(Image.fromarray(_x).crop((left, top, right, bottom)), dtype=np.float64))

        crop = np.array(crop, dtype=np.float64).transpose((1,2,0)) # Make sure channels are last
    else:
        if type(img) == np.ndarray:
            img = Image.fromarray(img)
        crop = img.crop((left, top, right, bottom))
    return crop


def plot_crops_from_data_tuple(data_tuple, min_size=(0, 0), figsize=(18, 30), max_size=None):
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
    fig, ax = plt.subplots(_n, _n, figsize=figsize)

    for axis, obj in zip(ax.flatten(), label['objects']):
        crop = get_image_crop(img, obj, min_size=min_size,
                              max_size=max_size, is_raw=label['is_raw'])
        axis.imshow(crop)
        axis.set_title(obj['classTitle'])
        axis.axis('off')

    fig.tight_layout()

    return fig


def add_label_box_to_axis(label: dict, ax, color: str = 'r', add_title: bool = False):
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
                ((left, top), (right, bottom)), _, _ = get_annotation_dimensions(
                    obj, is_raw=label['is_raw'])
                rec = Rectangle(xy=(left, top), height=bottom -
                                top, width=right-left, fill=False, color=color)
                ax.add_patch(rec)

                if add_title:
                    ax.annotate(text=obj['classTitle'], xy=(
                        left, top-10), fontsize='x-small')


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
    o1 = 0.3 * img[:, :, 0] + 0.58 * img[:, :, 1] + \
        0.11 * img[:, :, 2]   # Intensity/Luminance
    o2 = 0.25 * img[:, :, 0] + 0.25 * img[:, :, 1] - \
        0.5 * img[:, :, 2]   # BY opponent
    o3 = 0.5 * img[:, :, 0] - 0.5 * \
        img[:, :, 1]                        # RG opponent

    if normalize:
        ret = []
        for _x in [o1, o2, o3]:
            _max = _x.max()
            ret.append(_x / _max)
        return np.array(ret)

    return np.array([o1, o2, o3])


def plot_image_in_color_spaces(image: np.ndarray, figsize=(10, 5), cmap_rgb: str = None, cmap_opponent: str = None, cmap_original: str = None):
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
    fig, ax = plt.subplots(2, 4, figsize=figsize)

    for i, title in enumerate(['Red', 'Green', 'Blue']):
        temp = np.zeros(image.shape, dtype='uint8')
        temp[:, :, i] = image[:, :, i]
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


class OADSImageDataset(Dataset):
    def __init__(self, oads_access:OADS_Access, use_crops:bool, item_ids:list,
                class_index_mapping:dict=None, transform=None, target_transform=None, 
                device='cuda:0', target:str='label') -> None:
        super().__init__()

        self.oads_access = oads_access
        self.use_crops = use_crops
        self.item_ids = item_ids
        
        self.transform = transform
        self.target_transform = target_transform
        self.class_index_mapping = class_index_mapping
        self.device = device

        self.target = target

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, idx):
        dataset_name, image_name = self.item_ids[idx]
        tup = None
        while tup is None:
            if self.use_crops:
                tup = self.oads_access.load_crop(class_name=dataset_name, image_name=image_name)
            else:
                tup = self.oads_access.load_image(dataset_name=dataset_name, image_name=image_name)
        img, label = tup

        # img, label = self.data[idx]
        # img = np.transpose(np.array(img), (0, 1, 2))
        
        if self.target == 'label':
            label = label['classId']
            if self.class_index_mapping is not None:
                label = self.class_index_mapping[label]

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        if self.target == 'image':
            label = img

        return (img, label)
