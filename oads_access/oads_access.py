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
import torch
import tqdm
import random


class OADS_Access():
    """OADS_Access

    OADS Access - API to interact with the downloaded Open Amsterdam Dataset.

        Provides functionality to load and visualize images, annotations and other meta info.

        All preprocessing and formatting for using this dataset to train a DNN is contained.

    Example
    ----------
    >>> oads = OADS_Access(basedir='../data/oads')
    """

    def __init__(self, basedir: str, file_formats: list = None, use_jpg: bool = False, use_avg_crop_size: bool = True,
                 min_size_crops: tuple = (0, 0), max_size_crops: tuple = None, exclude_oversized_crops: bool = False,
                 n_processes: int = 8, exclude_classes: list = []):
        self.basedir = basedir
        self.file_formats = file_formats
        self.min_size_crops = min_size_crops
        self.max_size_crops = max_size_crops
        self.size_folder_name = self.min_size_crops if self.min_size_crops == self.max_size_crops else f"{self.min_size_crops}_{self.max_size_crops}"
        self.size_folder_name = str(self.size_folder_name).replace(' ', '')

        self.exclude_oversized_crops = exclude_oversized_crops
        self.n_processes = n_processes

        self.use_jpg = use_jpg

        self.img_dir = os.path.join(self.basedir, 'oads_arw', 'ARW')
        self.crops_dir = os.path.join(self.basedir, 'oads_arw', 'crops', self.size_folder_name)
        os.makedirs(self.crops_dir, exist_ok=True)
        self.ann_dir = os.path.join(self.basedir, 'oads_annotations')

        # self.check_has_crop_files()

        self.datasets = [x for x in os.listdir(
            self.ann_dir) if os.path.isdir(os.path.join(self.ann_dir, x))]

        self.images_per_dataset = {dataset_name: []
                                   for dataset_name in self.datasets}

        self.images_per_class = {}
        self.classes = []
        self.image_names = {}

        for dataset_name in self.datasets:

            # Get all annotations
            for annotation_file_name in os.listdir(os.path.join(self.ann_dir, dataset_name, 'ann')):

                tup = {'dataset_name': dataset_name}
                tup['annotation_file_name'] = annotation_file_name
                tup['annotation_file_path'] = os.path.join(
                    self.ann_dir, dataset_name, 'ann', annotation_file_name)
                if use_jpg:
                    # Get JPG information
                    jpg_path = os.path.join(
                        self.ann_dir, dataset_name, 'img', annotation_file_name.replace('.json', ''))
                    if os.path.exists(jpg_path):
                        tup['jpg_file_path'] = jpg_path
                        tup['jpg_file_name'] = annotation_file_name.replace(
                            '.json', '')

                raw_path = os.path.join(self.img_dir, annotation_file_name.replace(
                    'jpg', 'ARW').replace('.json', ''))
                if os.path.exists(raw_path):
                    tup['raw_file_name'] = annotation_file_name.replace(
                        'jpg', 'ARW').replace('.json', '')
                    tup['raw_file_path'] = raw_path

                file_id = annotation_file_name.split('.')[0]
                i = 0
                tup['crop_file_type'] = '.tiff'
                tup['crop_file_paths'] = []
                while True:
                    crop_path = os.path.join(self.crops_dir, f"{file_id}_{i}.tiff")
                    if os.path.exists(crop_path):
                        tup['crop_file_paths'].append(crop_path)
                        i += 1
                    else:
                        break
                
                ################
                # Get annotations
                if os.path.exists(tup['annotation_file_path']):
                    tup['object_labels'] = []
                    with open(tup['annotation_file_path'], 'r') as f:
                        content = json.load(f)
                        for obj in content['objects']:
                            if obj['classTitle'] not in exclude_classes:
                                tup['object_labels'].append(obj['classTitle'])
                                self.classes.append(obj['classTitle'])
                                if obj['classTitle'] in self.images_per_class:
                                    self.images_per_class[obj['classTitle']].append(f"{file_id}_{i}")
                                else:
                                    self.images_per_class[obj['classTitle']] = [f"{file_id}_{i}"]
                ################
    
                self.image_names[file_id] = tup
                self.images_per_dataset[dataset_name].append(
                    file_id)

        self.classes = set(self.classes)
        return


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
            with open(os.path.join(self.ann_dir, 'meta.json'), 'r') as f:
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

    def get_class(self, image_name:str, index:int, dataset_name:str = None):
        if 'object_labels' in self.image_names[image_name]:
            return self.image_names[image_name][index]
        else:
            return self.get_annotation(image_name=image_name)['objects'][index]['classTitle']

    def get_annotation(self, image_name: str,  dataset_name: str = None,  is_raw=False):
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
        # if is_raw:
        #     image_name = image_name.replace('ARW', 'jpg')
        # path = os.path.join(self.ann_dir, dataset_name,
        #                     'ann', f"{image_name}.json")
        info = self.image_names[image_name]
        path = info['annotation_file_path']
        if dataset_name is None:
            dataset_name = info['dataset_name']
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
            mask = np.where(np.arange(
                len(self.images_per_dataset[dataset_name])) >= max_number_images, False, True)
            np.random.shuffle(mask)
            image_name_iter = np.array(
                self.images_per_dataset[dataset_name])[mask]
        else:
            image_name_iter = self.images_per_dataset[dataset_name]

        data = []
        image_name: str

        for image_name in image_name_iter:
            tup = self.load_image(
                dataset_name=dataset_name, image_name=image_name)
            if tup is not None:
                if use_crops:
                    crops = self.make_and_save_crops_from_image(
                        img=tup[0], label=tup[1])
                    data.extend(crops)
                else:
                    data.append(tup)

        return data

    def load_image(self, image_name: str, dataset_name: str = None):
        info = self.image_names[image_name]
        if 'raw_file_path' in info.keys():
            filename = info['raw_file_path']
        elif 'jpg_file_path' in info.keys():
            filename = info['jpg_file_path']
        else:
            return None

        fileformat = os.path.splitext(filename)[-1]
        if self.file_formats is not None and fileformat not in self.file_formats:
            # print(f"File {fileformat} not in file formats {self.file_formats}. Skipping")
            return None

        is_raw = False
        if fileformat == '.arw' or fileformat == '.ARW':
            is_raw = True
            with rawpy.imread(filename) as raw:
                img = raw.postprocess()

        else:
            img = Image.open(filename)

        label = self.get_annotation(
            dataset_name=dataset_name, image_name=image_name, is_raw=is_raw)
        tup = (img, label)

        return tup

    def load_crop(self, crop_path: str, image_name: str, index: int):
        fileformat = os.path.splitext(crop_path)[-1]
        if self.file_formats is not None and fileformat not in self.file_formats:
            return None

        crop = Image.open(crop_path)
        label = self.get_annotation(image_name=image_name)['objects'][index]

        return (crop, label)

    def load_crop_from_image(self, image_name: str, index: int,  dataset_name: str = None,  is_opponent_space: bool = False):
        crop_path = self.image_names[image_name]['crop_file_paths'][index]
        if os.path.exists(crop_path):
            crop, obj = self.load_crop(
                crop_path, image_name=image_name, index=index)
        else:
            tup = self.load_image(
                dataset_name=dataset_name, image_name=image_name)

            if tup is None:
                return None

            img, label = tup

            # This index needs to be normalized/adjust somehow
            obj = label['objects'][index]
            if self.exclude_oversized_crops:
                width, height = self.get_annotation_size(
                    obj, is_raw=label['is_raw'])
                if height > self.max_size_crops[0] or width > self.max_size_crops[1]:
                    return None
            crop = self.get_image_crop(
                img=img, object=obj, is_raw=label['is_raw'], is_opponent_space=is_opponent_space)

        return crop, obj

    def get_data_list(self, dataset_names=None, use_crops: bool = False, max_number_images: int = None):
        """get_data_list

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
        >>> data = oads.get_data_list() 
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
        with multiprocessing.Pool(self.n_processes) as pool:
            data = list(tqdm.tqdm(pool.imap(self.load_images_from_dataset, zip(dataset_names, max_number_images, use_crops)),
                                  total=len(dataset_names)))
        data = [x for dataset in data for x in dataset]

        return data

    def get_annotation_size(self, obj: dict, is_raw:bool=True):
        ((left, top), (right, bottom)), _, _ = get_annotation_dimensions(
            obj, is_raw=is_raw)
        height = bottom - top
        width = right - left

        return width, height

    def get_maximum_annotation_size(self, data_list: list = None):
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
        if data_list is None:
            data_list = self.get_data_list()

        _max_height = 0
        _max_width = 0

        for (_, label) in data_list:
            for obj in label['objects']:
                width, height = self.get_annotation_size(
                    obj, is_raw=label['is_raw'])
                _max_height = max(_max_height, height)
                _max_width = max(_max_width, width)

        return _max_width, _max_height

    # Deprecated
    def get_train_val_test_split(self, data_list: "list|np.ndarray" = None, val_size: float = 0.1, test_size: float = 0.1,
                                 use_crops: bool = False, max_number_images: int = None):
        """get_train_val_test_split

        Split the data_list into train, validation and test sets.

        Parameters
        ----------
        data_list: list | np.ndarray, optional
            Data to split, by default None
        val_size: float, optional
            Proportion of total data to be used in validation set, by default 0.1
        test_size: float, optional
            Proportion of total data to be used in test set, by default 0.1
        use_crops: bool, optional
            If data_list is None, whether to use crop_list. If False, data_list will be loaded, by default False
        min_size: tuple, optional
            If data_list is None and use_crops is True, the minimum size (width, height) to crop the image to.

        Returns
        ----------
        tuple
            Sets of training data, validation data, testing data with given proportions

        Example
        ----------
        >>> 
        """
        if data_list is None:
            if use_crops:
                data_list = self.get_crop_list(
                    max_number_images=max_number_images)
            else:
                data_list = self.get_data_list(
                    max_number_images=max_number_images)

        train_data, test_data = train_test_split(
            data_list, test_size=val_size+test_size)
        test_data, val_data = train_test_split(
            test_data, test_size=test_size / (val_size+test_size))

        return train_data, val_data, test_data

    def get_number_of_annotations(self, image_name: str):
        annotation = self.get_annotation(image_name=image_name)
        self.image_names[image_name]['number_of_annotations'] = len(
            annotation['objects'])

        return len(annotation['objects'])

    def get_train_val_test_split_indices(self, use_crops: bool, val_size: float = 0.1, test_size: float = 0.1, remove_duplicates: bool = True):
        if use_crops:
            # Tuple of image_name+index for index counts the number of crops for this image
            image_ids = []
            for image_name, info in self.image_names.items():
                if 'number_of_annotations' in info:
                    number_of_annotations = info['number_of_annotations']
                else:
                    number_of_annotations = self.get_number_of_annotations(
                        image_name=image_name)

                for i in range(number_of_annotations):
                    fileformat = info['crop_file_type']
                    if self.file_formats is not None and fileformat not in self.file_formats:
                        continue
                    if len(self.image_names[image_name]['crop_file_paths']) == 0:
                        continue
                    image_ids.append((image_name, i))

        else:
            # image_ids = get_list(self.image_names.items())
            image_ids = list(self.image_names.keys())

        if remove_duplicates:
            image_ids = list(set(image_ids))
        train_ids, test_ids = train_test_split(
            image_ids, test_size=val_size+test_size)
        test_ids, val_ids = train_test_split(
            test_ids, test_size=test_size / (val_size+test_size))

        return train_ids, val_ids, test_ids

    def _prepare_crops_dataset(self, args):
        dataset_name, (convert_to_opponent_space, overwrite) = args
        for image_name in self.images_per_dataset[dataset_name]:
            tup = self.load_image(
                dataset_name=dataset_name, image_name=image_name)
            if tup is not None:
                (img, label) = tup
                if convert_to_opponent_space:
                    img = rgb_to_opponent_space(np.array(img))
                _ = self.make_and_save_crops_from_image(
                    img=img, label=label, image_name=image_name, is_opponent_space=convert_to_opponent_space, save_files=True, overwrite=overwrite, return_crops=False)

    def prepare_crops(self, dataset_names: list = None, convert_to_opponent_space: bool = False, overwrite: bool = False):
        if dataset_names is None:
            dataset_names = self.datasets

        args = zip(dataset_names, [
                   (convert_to_opponent_space, overwrite) for _ in range(len(dataset_names))])

        with multiprocessing.Pool(self.n_processes) as pool:
            print(f"Number of processes: {pool._processes}")
            _ = list(tqdm.tqdm(pool.imap(self._prepare_crops_dataset,
                     args), total=len(dataset_names)))

        # self.check_has_crop_files()

    # TODO this needs to be adjusted to the new layout
    def get_crop_list(self, data_list: "list|np.ndarray" = None, convert_to_opponent_space: bool = False,
                      max_number_images: int = None, save_files: bool = False, recompute_crops: bool = False):

        crop_list = []
        if not recompute_crops and self.has_crops_files:
            # with multiprocessing.Pool(self.n_processes) as pool:
            #     results = list(tqdm.tqdm(pool.imap(self.load_crop_class, self.crop_files_names.items()), total=len(self.crop_files_names.items())))

            # crop_list = [x for dataset in results for x in dataset]
            pass

        else:

            if data_list is None:
                crop_list = self.get_data_list(
                    max_number_images=max_number_images, use_crops=True)
            else:
                for (img, label) in data_list:
                    if convert_to_opponent_space:
                        img = rgb_to_opponent_space(np.array(img))
                    crops = self.make_and_save_crops_from_image(
                        img=img, label=label, is_opponent_space=convert_to_opponent_space, save_files=save_files)
                    crop_list.extend(crops)

        return crop_list

    def make_and_save_crops_from_image(self, img, label, image_name: str, is_opponent_space: bool = False, save_files: bool = False, overwrite: bool = False, return_crops: bool = True):
        if return_crops:
            crops = []
        
        fileending = '.tiff'
        for index, obj in enumerate(label['objects']):
            filename = os.path.join(
                self.crops_dir, f"{image_name}_{index}{fileending}")

            if os.path.exists(filename) and not overwrite:
                continue
            if self.exclude_oversized_crops:
                width, height = self.get_annotation_size(
                    obj, is_raw=label['is_raw'])
                if height > self.max_size_crops[0] or width > self.max_size_crops[1]:
                    continue
            crop = self.get_image_crop(
                img=img, object=obj, is_raw=label['is_raw'], is_opponent_space=is_opponent_space)
            if return_crops:
                crops.append((crop, obj))

            if save_files:
                try:
                    crop.save(fp=filename)
                    with open(f"{filename}.json", 'w') as f:
                        json.dump(obj, fp=f)
                except SystemError as e:
                    print(f"Cannot save image {filename} due to:\n\n{e}")

        if return_crops:
            return crops

    def get_dataset_stats(self):
        means, stds = [], []

        results = self.apply_per_image(lambda x: (
            np.mean(np.array(x[0]), axis=(0, 1)), np.std(np.array(x[0]), axis=(0, 1))))

        for image_name, (m, s) in results.items():
            means.append(m)
            stds.append(s)

        # for img, _ in data_list:
        #     img_np = np.array(img)
        #     mean = np.mean(img_np, axis=(0, 1))
        #     std = np.std(img_np, axis=(0, 1))
        #     means.append(mean)
        #     stds.append(std)

        return np.array(means), np.array(stds)

    def plot_image_size_distribution(self, use_crops: bool, data_list: list = None, file_formats: list = None, figsize: tuple = (10, 5)):

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

    def apply_per_crop_annotation(self, custom_function):
        results = {}

        for dataset_name, image_names in self.images_per_dataset.items():
            if dataset_name not in results.keys():
                results[dataset_name] = {}
            for image_name in image_names:
                results[dataset_name][image_name] = []
                info = self.image_names[image_name]
                if 'number_of_annotations' in info:
                    number_of_annotations = info['number_of_annotations']
                else:
                    number_of_annotations = self.get_number_of_annotations(
                        image_name=image_name)

                for index in range(number_of_annotations):
                    label = self.get_annotation(dataset_name=dataset_name, image_name=image_name, is_raw=True)
                    for obj in label['objects']:
                        results[dataset_name][image_name].append(custom_function(obj))

        return results
    
    def apply_per_crop(self, custom_function, max_number_crops: int = None):
        results = {}

        crop_counter = 0
        for dataset_name, image_names in self.images_per_dataset.items():
            if dataset_name not in results.keys():
                results[dataset_name] = {}
            for image_name in image_names:
                info = self.image_names[image_name]
                if 'number_of_annotations' in info:
                    number_of_annotations = info['number_of_annotations']
                else:
                    number_of_annotations = self.get_number_of_annotations(
                        image_name=image_name)

                for index in range(number_of_annotations):
                    tup = self.load_crop_from_image(
                        dataset_name=dataset_name, image_name=image_name, index=index)
                    if tup is not None:
                        if max_number_crops is not None and crop_counter >= max_number_crops:
                            return results
                        results[dataset_name][f"{image_name}_{index}"] = custom_function(
                            tup)
                        crop_counter += 1
            

        # for dataset_name, image_name, index in self.item_to_image:
        #     if dataset_name not in results.keys():
        #         results[dataset_name] = {}
        #     tup = self.load_crop_from_image(
        #         dataset_name=dataset_name, image_name=image_name, index=index)
        #     if tup is not None:
        #         if max_number_crops is not None and crop_counter >= max_number_crops:
        #             return results
        #         results[dataset_name][f"{image_name}_{index}"] = custom_function(
        #             tup)
        #         crop_counter += 1

        return results

    # TODO test this with new layout

    def apply_per_dataset(self, args):
        results = {}
        image_counter = 0
        for image_name in self.images_per_dataset[args]:
            tup = self.load_image(image_name=image_name)
            if tup is not None:
                if self.max_number_images is not None and image_counter >= self.max_number_images:
                    return results
                results[image_name] = self.custom_function(tup)
                image_counter += 1

    def apply_per_image(self, custom_function, max_number_images: int = None):
        results = {}
        
        self.custom_function = custom_function
        self.max_number_images = max_number_images
        with multiprocessing.Pool(self.n_processes) as pool:
            results = list(tqdm.tqdm(pool.imap(self.apply_per_dataset, self.images_per_dataset.keys(
            )), total=len(self.images_per_dataset.keys())))

        results = [x for dataset in results for x in dataset]

        return results

    def apply_custom_data_augmentation(self, data_list: list, augmentation_function):
        return list(map(augmentation_function, data_list))

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

    def get_image_crop(self, img: "np.ndarray|list|Image.Image", object: dict, is_raw: bool = False, is_opponent_space: bool = False):
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
            object, is_raw=is_raw, min_size=self.min_size_crops, max_size=self.max_size_crops)

        height, width = img.shape[:2]

        old = (left, top, right, bottom, height, width, min_size, max_size)

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

        # make sure nothing is cropped outside the actual image
        redo = True
        while redo:
            if left < 0:
                redo = True
                right -= left
                left = 0
            elif right > width:
                redo = True
                left -= (right - width)
                right = width
            elif top < 0:
                redo = True
                bottom -= top
                top = 0
            elif bottom > height:
                redo = True
                top -= (bottom - height)
                bottom = height
            else:
                redo = False

        ########

        if is_opponent_space:
            crop = []
            for _x in img:
                crop.append(np.array(Image.fromarray(_x).crop(
                    (left, top, right, bottom)), dtype=np.float64))

            crop = np.array(crop, dtype=np.float64).transpose(
                (1, 2, 0))  # Make sure channels are last
        else:
            if type(img) == np.ndarray:
                img = Image.fromarray(img)
            if left == right or top == bottom:
                print(f"Label: {object}")
            crop = img.crop((left, top, right, bottom))
        return crop

    def plot_crops_from_data_tuple(self, data_tuple, min_size=(0, 0), figsize=(18, 30), max_size=None):
        """plot_crops_from_data_tuple

        For a given tuple of (image, label) (e.g., from get_data_list) plot all the crops corresponding to the annotated labels.

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

        for index, axis in enumerate(ax.flatten()):
            if index >= len(label['objects']):
                plt.delaxes(axis)
            else:
                obj = label['objects'][index]
                crop = self.get_image_crop(img, obj, min_size=min_size,
                                           max_size=max_size, is_raw=label['is_raw'])
                axis.imshow(crop)
                axis.set_title(obj['classTitle'])
                axis.axis('off')

        fig.tight_layout()

        return fig


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


def add_label_box_to_axis(obj: dict, ax, is_raw:bool, color:str='r', add_title:bool=False):
    ((left, top), (right, bottom)), _, _ = get_annotation_dimensions(
        obj, is_raw=is_raw)
    rec = Rectangle(xy=(left, top), height=bottom -
                    top, width=right-left, fill=False, color=color)
    ax.add_patch(rec)

    if add_title:
        ax.annotate(text=obj['classTitle'], xy=(
            left, top-10), fontsize='x-small')

def add_label_boxes_to_axis(label: dict, ax, color: str = 'r', add_title: bool = False):
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
        # img_height, img_width = label['size']['height'], label['size']['width']
        for obj in label['objects']:
            if obj['geometryType'] == 'rectangle':
                add_label_box_to_axis(obj=obj, ax=ax, color=color, add_title=add_title, is_raw=label['is_raw'])


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
    def __init__(self, oads_access: OADS_Access, use_crops: bool, item_ids: list,
                 class_index_mapping: dict = None, transform=None, target_transform=None,
                 device='cuda:0', target: str = 'label') -> None:
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
        if self.use_crops:
            image_name, index = self.item_ids[idx]
            tup = self.oads_access.load_crop_from_image(
                image_name=image_name, index=index)
        else:
            image_name = self.item_ids[idx]
            tup = self.oads_access.load_image(image_name=image_name)

        if tup is None:
            return None

        img, label = tup

        if img is None or label is None:
            return None

        if self.target == 'label':
            label = label['classId']
            if self.class_index_mapping is not None:
                label = self.class_index_mapping[label]

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        img = img.float()

        if self.target == 'image':
            label = img

        return (img, label)
