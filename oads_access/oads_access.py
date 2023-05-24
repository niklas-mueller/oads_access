import multiprocessing
import os
import json
from PIL import Image
from pytorch_utils.pytorch_utils import ToJpeg
import rawpy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch import nn as nn
import torch
import ctypes
import tqdm
from p_tqdm import p_map


class OADS_Access():
    """OADS_Access

    OADS Access - API to interact with the downloaded Open Amsterdam Dataset.

        Provides functionality to load and visualize images, annotations and other meta info.

        All preprocessing and formatting for using this dataset to train a DNN is contained.

    Example
    ----------
    >>> oads = OADS_Access(basedir='../data/oads')
    """
    FULL_SIZE = (5496, 3672)

    def __init__(self, basedir: str, file_formats: list = None, use_jpeg: bool = False,
                 n_processes: int = 8, exclude_classes: list = [], exclude_datasets:list=[], 
                 jpeg_p=0.5, jpeg_quality=90, min_size_crops=None, crops_dir:str=None):
        self.basedir = basedir
        self.file_formats = file_formats
        self.n_processes = n_processes
        self.exclude_classes = exclude_classes
        self.exclude_datasets = exclude_datasets
        self.use_jpeg = use_jpeg
        self.min_size_crops = min_size_crops


        self.jpeg_p = jpeg_p
        self.jpeg_quality = jpeg_quality

        self.img_dir = os.path.join(self.basedir, 'oads_arw', 'ARW')
        self.crops_dir = crops_dir
        if self.crops_dir is None:
            self.crops_dir = os.path.join(self.basedir, 'oads_arw', 'crops', 'jpeg' if use_jpeg else 'tiff')
        os.makedirs(self.crops_dir, exist_ok=True)
        self.ann_dir = os.path.join(self.basedir, 'oads_annotations')

        self.datasets = [x for x in os.listdir(
            self.ann_dir) if os.path.isdir(os.path.join(self.ann_dir, x)) and x not in self.exclude_datasets]

        self.images_per_dataset = {dataset_name: []
                                   for dataset_name in self.datasets}

        self.images_per_class = {}
        self.classes = []
        self.image_names = {}
        self.label_id_to_image = {}
        
        self.missing_raw_images = {}

        for dataset_name in self.datasets:

            # Get all annotations
            for annotation_file_name in os.listdir(os.path.join(self.ann_dir, dataset_name, 'ann')):

                tup = {'dataset_name': dataset_name}
                tup['annotation_file_name'] = annotation_file_name
                tup['annotation_file_path'] = os.path.join(
                    self.ann_dir, dataset_name, 'ann', annotation_file_name)
                if use_jpeg:
                    # Get JPG information
                    jpg_path = os.path.join(
                        self.ann_dir, dataset_name, 'img', annotation_file_name.replace('.json', ''))
                    if os.path.exists(jpg_path):
                        tup['jpg_file_path'] = jpg_path
                        tup['jpg_file_name'] = annotation_file_name.replace(
                            '.json', '')

                file_id = annotation_file_name.split('.')[0]

                raw_path = os.path.join(self.img_dir, annotation_file_name.replace(
                    'jpg', 'ARW').replace('.json', ''))
                if os.path.exists(raw_path):
                    tup['raw_file_name'] = annotation_file_name.replace(
                        'jpg', 'ARW').replace('.json', '')
                    tup['raw_file_path'] = raw_path
                else:
                    # Get data of missing RAW image
                    if os.path.exists(tup['annotation_file_path']):
                        with open(tup['annotation_file_path'], 'r') as f:
                            content = json.load(f)
                    else:
                        content = None
                    self.missing_raw_images[file_id] = {
                        'dataset': dataset_name,
                        'person': []
                    }
                    if content is not None:
                        for obj in content['objects']:
                            self.missing_raw_images[file_id]['person'].append(obj['labelerLogin'])
                    continue

                for type_folder_name in os.listdir(os.path.join(basedir, 'oads_arw')):
                    if os.path.isdir(os.path.join(basedir, 'oads_arw', type_folder_name)) and type_folder_name not in ['ARW', 'crops']:
                        type_filename = annotation_file_name.replace('jpg', type_folder_name).replace('.json', '')
                        tup[f'{type_folder_name}_file_name'] = type_filename
                        tup[f'{type_folder_name}_file_path'] = os.path.join(basedir, 'oads_arw', type_folder_name, type_filename)
                
                i = 0
                tup['crop_file_type'] = '.jpeg' if self.use_jpeg else '.tiff'
                tup['crop_file_paths'] = {}
                while True:
                    crop_path = os.path.join(self.crops_dir, f"{file_id}_{i}{tup['crop_file_type']}")
                    if os.path.exists(crop_path):
                        tup['crop_file_paths'][i] = crop_path
                        i += 1
                    else:
                        break
                
                ################
                # Get annotations
                tup['object_labels'] = {}
                if os.path.exists(tup['annotation_file_path']):
                    with open(tup['annotation_file_path'], 'r') as f:
                        content = json.load(f)
                        tup['number_of_annotations'] = len(content['objects'])
                        for index, obj in enumerate(content['objects']):
                            if obj['classTitle'] not in exclude_classes:
                                tup['object_labels'][index] = obj['classTitle']

                                self.classes.append(obj['classTitle'])
                                if obj['classTitle'] in self.images_per_class:
                                    self.images_per_class[obj['classTitle']].append(f"{file_id}_{index}")
                                else:
                                    self.images_per_class[obj['classTitle']] = [f"{file_id}_{index}"]

                                self.label_id_to_image[obj['id']] = file_id

                # if len(tup['crop_file_paths']) == 0:
                #     tup['number_of_annotations'] = len(tup['object_labels'])
                ################
    
                self.image_names[file_id] = tup
                self.images_per_dataset[dataset_name].append(
                    file_id)

        self.classes = set(self.classes)
        return


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
            return self.get_annotation(image_name=image_name, dataset_name=dataset_name)['objects'][index]['classTitle']

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
                        image_name=image_name)
                    data.extend(crops)
                else:
                    data.append(tup)

        return data

    def load_image(self, image_name: str, dataset_name: str = None):
        info = self.image_names[image_name]
        if self.file_formats is not None:
            fileformat: str
            for fileformat in self.file_formats:
                if fileformat in ['.ARW']:
                    filename = info['raw_file_path']
                    continue
                fileformat = fileformat.replace('.', '')
                if f'{fileformat}_file_path' in info.keys():
                    filename = info[f'{fileformat}_file_path']
                    break
        
        else:
            if 'raw_file_path' in info.keys():
                filename = info['raw_file_path']
            elif 'jpg_file_path' in info.keys():
                filename = info['jpg_file_path']
            else:
                return None

        fileformat = os.path.splitext(filename)[-1]
        if self.file_formats is not None and fileformat not in self.file_formats:
            print(f"File {fileformat} not in file formats {self.file_formats}. Skipping")
            return None

        is_raw = False
        if fileformat == '.arw' or fileformat == '.ARW':
            is_raw = True
            with rawpy.imread(filename) as raw:
                img = raw.postprocess()
                img = Image.fromarray(img)

        else:
            img = Image.open(filename)
            if 'tiff' in fileformat and img.size == self.FULL_SIZE:
                is_raw = True

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

    def make_crop_from_image(self, image_name: str, index: int, filename:str, dataset_name:str = None):
        tup = self.load_image(
            dataset_name=dataset_name, image_name=image_name)

        if tup is None:
            print(f'tup is none: {image_name}')
            return None

        img, label = tup

        is_raw = label['is_raw']
        if self.use_jpeg:
            img = ToJpeg(resize=False, p=self.jpeg_p, quality=self.jpeg_quality)(img)
            # is_raw = False

        # This index needs to be normalized/adjust somehow
        obj = label['objects'][index]
        # if self.exclude_oversized_crops:
        #     width, height = self.get_annotation_size(
        #         obj, is_raw=is_raw)
        #     if height > self.max_size_crops[0] or width > self.max_size_crops[1]:
        #         return None
        crop = self.get_image_crop(
            img=img, object=obj, is_raw=is_raw) # , is_opponent_space=is_opponent_space
        
        # Update self
        self.image_names[image_name]['crop_file_paths'][index] = filename
        
        return crop, obj

    def load_crop_from_image(self, image_name: str, index: int,  dataset_name: str = None, force_recompute:bool=False):
        paths = self.image_names[image_name]['crop_file_paths']
        if index in paths and os.path.exists(paths[index]) and not force_recompute:
            # print(f'Using Preloaded Crop:\n{paths[index]}')
            crop_path = paths[index]
            tup = self.load_crop(
                crop_path, image_name=image_name, index=index)
            
        else:
            filename = self.make_image_crop_name(image_name=image_name, index=index)
            tup = self.make_crop_from_image(image_name=image_name, index=index, dataset_name=dataset_name, filename=filename)
            # tup = self.load_image(
            #     dataset_name=dataset_name, image_name=image_name)

            # if tup is None:
            #     print(image_name)
            #     return None

            # img, label = tup

            # is_raw = label['is_raw']
            # if use_jpeg:
            #     img = ToJpeg(resize=False, p=self.jpeg_p, quality=self.jpeg_quality)(img)
            #     # is_raw = False

            # # This index needs to be normalized/adjust somehow
            # obj = label['objects'][index]
            # if self.exclude_oversized_crops:
            #     width, height = self.get_annotation_size(
            #         obj, is_raw=is_raw)
            #     if height > self.max_size_crops[0] or width > self.max_size_crops[1]:
            #         return None
            # crop = self.get_image_crop(
            #     img=img, object=obj, is_raw=is_raw, is_opponent_space=is_opponent_space)

        return tup

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
        (left, top), (right, bottom) = get_annotation_dimensions(
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

    def get_number_of_annotations(self, image_name: str, annotation=None):
        if annotation is None:
            annotation = self.get_annotation(image_name=image_name)
        self.image_names[image_name]['number_of_annotations'] = len(
            annotation['objects'])

        return len(annotation['objects'])

    def get_train_val_test_split_indices(self, use_crops: bool, val_size: float = 0.1, test_size: float = 0.1, remove_duplicates: bool = True, dataset_names:list = [], shuffle:bool=False, random_state:int=42):
        if use_crops:
            # Tuple of image_name+index for index counts the number of crops for this image
            image_ids = []
            for image_name, info in self.image_names.items():
                if len(dataset_names) > 0 and info['dataset_name'] not in dataset_names:
                    continue
                # if 'number_of_annotations' in info:
                #     number_of_annotations = info['number_of_annotations']
                # else:
                #     number_of_annotations = self.get_number_of_annotations(
                #         image_name=image_name)

                # for i in range(number_of_annotations):
                #     fileformat = info['crop_file_type']
                #     if len(self.image_names[image_name]['crop_file_paths']) > 0:
                #         if self.file_formats is not None and fileformat not in self.file_formats:
                #             continue

                #     if self.image_names[image_name]['object_labels'][i] not in self.exclude_classes:
                #         image_ids.append((image_name, i))
                for index, _ in self.image_names[image_name]['object_labels'].items():
                        image_ids.append((image_name, index))

        else:
            # image_ids = get_list(self.image_names.items())
            image_ids = list(self.image_names.keys())

        if remove_duplicates:
            image_ids = list(set(image_ids))
        train_ids, test_ids = train_test_split(
            image_ids, test_size=val_size+test_size, shuffle=shuffle, random_state=random_state)
        test_ids, val_ids = train_test_split(
            test_ids, test_size=test_size / (val_size+test_size), shuffle=shuffle, random_state=random_state)

        return train_ids, val_ids, test_ids
    
    def prepare_crops(self, dataset_names: list=None, overwrite:bool=False):
        if dataset_names is None:
            dataset_names = self.datasets

        args = []
        for dataset_name in dataset_names:
            for img in self.images_per_dataset[dataset_name]:
                args.append((img, True, overwrite, False))

        with multiprocessing.Pool(self.n_processes) as pool:
            _ = list(tqdm.tqdm(pool.starmap(self.make_and_save_crops_from_image, args), total=len(args)))
        

    # def _prepare_crops_dataset(self, args):
    #     dataset_name, (overwrite) = args
    #     for image_name in self.images_per_dataset[dataset_name]:
    #         # tup = self.load_image(
    #         #     dataset_name=dataset_name, image_name=image_name)
    #         # if tup is not None:
    #         #     (img, label) = tup
    #         #     if convert_to_opponent_space:
    #         #         img = rgb_to_opponent_space(np.array(img))
    #         #     _ = self.make_and_save_crops_from_image(
    #         #         img=img, label=label, image_name=image_name, is_opponent_space=convert_to_opponent_space, save_files=True, overwrite=overwrite, return_crops=False)
    #         _ = self.make_and_save_crops_from_image(image_name=image_name, overwrite=overwrite, save_files=True)

    # def prepare_crops(self, dataset_names: list = None, overwrite: bool = False):
    #     if dataset_names is None:
    #         dataset_names = self.datasets

    #     args = zip(dataset_names, [
    #                (overwrite) for _ in range(len(dataset_names))])

    #     with multiprocessing.Pool(self.n_processes) as pool:
    #         print(f"Number of processes: {pool._processes}")
    #         _ = list(tqdm.tqdm(pool.imap(self._prepare_crops_dataset,
    #                  args), total=len(dataset_names)))

    #     # self.check_has_crop_files()

    def make_image_crop_name(self, image_name, index:int):
        file_ending = f'.{self.crops_dir.split("/")[-1]}'

        filename = os.path.join(
            self.crops_dir, f"{image_name}_{index}{file_ending}")
        
        return filename
    
    def make_and_save_crops_from_image(self, image_name: str, save_files: bool = False, overwrite: bool = False, return_crops: bool = True):
        if return_crops:
            crops = []
        
        # Get label
        label = self.get_annotation(image_name=image_name, is_raw=True)
        
        # Iterate over all annotations
        for index, obj in enumerate(label['objects']):

            filename = self.make_image_crop_name(image_name=image_name, index=index)
            
            if os.path.exists(filename) and not overwrite:
                continue
            
            # Make crop for current annotation/crop
            crop, obj = self.make_crop_from_image(image_name=image_name, index=index, filename=filename)

            # self.image_names[image_name]['crop_file_paths'].append(filename)

            # if self.exclude_oversized_crops:
            #     width, height = self.get_annotation_size(
            #         obj, is_raw=label['is_raw'])
            #     if height > self.max_size_crops[0] or width > self.max_size_crops[1]:
            #         continue
            # crop = self.get_image_crop(
            #     img=img, object=obj, is_raw=label['is_raw'], is_opponent_space=is_opponent_space)

            # Return if wanted
            if return_crops:
                crops.append((crop, obj))

            # Save if wanted
            if save_files:
                try:
                    # Save image
                    crop.save(fp=filename)

                    # Save annotation
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

    def plot_image_size_distribution(self, figsize: tuple = (10, 5)):

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

    def get_image_crop(self, img: "np.ndarray|list|Image.Image", object: dict, is_raw: bool = False):
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
        ((left, top), (right, bottom)) = get_annotation_dimensions(
            object, is_raw=is_raw)

        if type(img) is np.ndarray:
            height, width = img.shape[:2]
        else:
            height, width = np.array(img).shape[:2]

        # old = (left, top, right, bottom, height, width)

        if self.min_size_crops is not None:
            min_size = self.min_size_crops # (800, 800)

            # Check if crop would be too small
            if right-left < min_size[0]:
                mid_point = left + (right - left) / 2
                left = mid_point - min_size[0] / 2
                right = mid_point + min_size[0] / 2
            if bottom-top < min_size[1]:
                mid_point = top + (bottom - top) / 2
                top = mid_point - min_size[1] / 2
                bottom = mid_point + min_size[1] / 2

        # # Check if crop would be too big
        # if not max_size is None:
        #     if right - left > max_size[0]:
        #         mid_point = left + (right - left) / 2
        #         left = mid_point - max_size[0] / 2
        #         right = mid_point + max_size[0] / 2
        #     if bottom - top > max_size[1]:
        #         mid_point = top + (bottom - top) / 2
        #         top = mid_point - max_size[1] / 2
        #         bottom = mid_point + max_size[1] / 2

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

        # if is_opponent_space:
        #     coc_crop = []
        #     if is_raw:
        #         max_size = tuple(np.multiply(max_size, 1/4).astype(int))
        #     for _x in img.transpose((2,0,1)):
        #         # crop.append(np.array(Image.fromarray(_x).crop(
        #         #     (left, top, right, bottom)), dtype=np.float64))
        #         crop = Image.fromarray(_x).crop(
        #             (old[0], old[1], old[2], old[3]))
        #         re_crop = crop.resize(max_size)
        #         coc_crop.append(np.array(re_crop, dtype=np.float64))

        #     crop = np.array(coc_crop, dtype=np.float64).transpose(
        #         (1, 2, 0))  # Make sure channels are last
        # else:

        if type(img) == np.ndarray:
            img = Image.fromarray(img)
        if left == right or top == bottom:
            print(f"Label: {object}")
        crop = img.crop((left, top, right, bottom))
        # crop = img.crop((old[0], old[1], old[2], old[3]))
        # if is_raw:
        #     max_size = tuple(np.multiply(max_size, 1/4).astype(int))
        # crop = crop.resize(max_size)

        return crop

    def plot_crops_from_data_tuple(self, data_tuple, figsize=(18, 30), axis_off:bool=True):
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
                crop = self.get_image_crop(img, obj, is_raw=label['is_raw'])
                axis.imshow(crop)
                axis.set_title(obj['classTitle'])

                if axis_off:
                    axis.axis('off')

        fig.tight_layout()

        return fig


def get_image_size(tup):
    img, _ = tup
    (height, width, channels) = np.array(img).shape
    return height, width


def get_annotation_dimensions(obj: dict, is_raw):
    ((left, top), (right, bottom)) = obj['points']['exterior']
    if is_raw:
        left = left * 4
        top = top * 4
        right = right * 4
        bottom = bottom * 4

    return ((left, top), (right, bottom))


def add_label_box_to_axis(obj: dict, ax, is_raw:bool, color:str='r', add_title:bool=False, fontsize='x-small'):
    ((left, top), (right, bottom)) = get_annotation_dimensions(
        obj, is_raw=is_raw)
    rec = Rectangle(xy=(left, top), height=bottom -
                    top, width=right-left, fill=False, color=color)
    ax.add_patch(rec)

    if add_title:
        ax.annotate(text=obj['classTitle'], xy=(
            left, top-10), fontsize=fontsize)

def add_label_boxes_to_axis(label: dict, ax, color: str = 'r', add_title: bool = False, fontsize='x-small'):
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
                add_label_box_to_axis(obj=obj, ax=ax, color=color, add_title=add_title, is_raw=label['is_raw'], fontsize=fontsize)


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

class OADSImageDatasetSharedMem(Dataset):
    def __init__(self, oads_access: OADS_Access, use_crops: bool, item_ids: list, size: tuple,
                 class_index_mapping: dict = None, transform=None, target_transform=None,
                 device='cuda:0', target: str = 'label', force_recompute:bool=False,
                 return_index:bool=False) -> None:
        super().__init__()

        c, h, w = size

        self.oads_access = oads_access
        self.use_crops = use_crops
        self.item_ids = item_ids

        self.transform = transform
        self.target_transform = target_transform
        self.class_index_mapping = class_index_mapping
        self.device = device

        self.target = target
        self.force_recompute = force_recompute

        self.return_index = return_index

        self.nb_samples = len(self.item_ids)

        shared_array_base_x = multiprocessing.Array(ctypes.c_float, self.nb_samples*c*h*w)
        shared_array_x = np.ctypeslib.as_array(shared_array_base_x.get_obj())
        shared_array_x = shared_array_x.reshape(self.nb_samples, c, h, w)
        self.shared_array_x = torch.from_numpy(shared_array_x)

        if target == 'label':
            shared_array_base_y = multiprocessing.Array(ctypes.c_float, self.nb_samples)
            shared_array_y = np.ctypeslib.as_array(shared_array_base_y.get_obj())
            shared_array_y = shared_array_y.reshape(self.nb_samples,)
        elif target == 'image':
            shared_array_base_y = multiprocessing.Array(ctypes.c_float, self.nb_samples*c*h*w)
            shared_array_y = np.ctypeslib.as_array(shared_array_base_y.get_obj())
            shared_array_y = shared_array_y.reshape(self.nb_samples, c, h, w)
        self.shared_array_y = torch.from_numpy(shared_array_y)

        shared_array_base_use_cache = multiprocessing.Array(ctypes.c_bool, self.nb_samples)
        shared_array_use_cache = np.ctypeslib.as_array(shared_array_base_use_cache.get_obj())
        shared_array_use_cache = shared_array_use_cache.reshape(self.nb_samples,)
        self.shared_array_use_cache = torch.from_numpy(shared_array_use_cache)

    
    def __len__(self):
        return self.nb_samples

    def __getitem__(self, idx):
        if not self.shared_array_use_cache[idx]:
            # print(f'Caching')
            if self.use_crops:
                image_name, index = self.item_ids[idx]
                tup = self.oads_access.load_crop_from_image(
                    image_name=image_name, index=index, force_recompute=self.force_recompute)
                
            else:
                image_name = self.item_ids[idx]
                tup = self.oads_access.load_image(image_name=image_name)
        
            if tup is None:
                return None

            img, label = tup
            del tup

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

            self.shared_array_x[idx] = img
            self.shared_array_y[idx] = label

            self.shared_array_use_cache[idx] = True
        else:
            # print(f'Using cached')
            img = self.shared_array_x[idx]
            label = self.shared_array_y[idx]

        if self.return_index:
            return (img, label, idx)
        
        return (img, label)


class OADSImageDataset(Dataset):
    def __init__(self, oads_access: OADS_Access, use_crops: bool, item_ids: list,
                 class_index_mapping: dict = None, transform=None, target_transform=None,
                 device='cuda:0', target: str = 'label', force_recompute:bool=False, preload_all:bool=False,
                 return_index:bool=False) -> None:
        super().__init__()

        self.oads_access = oads_access
        self.use_crops = use_crops
        self.item_ids = item_ids

        self.transform = transform
        self.target_transform = target_transform
        self.class_index_mapping = class_index_mapping
        self.device = device

        self.target = target
        self.force_recompute = force_recompute
        self.preload_all = preload_all

        self.return_index = return_index

        self.tupels = {}

        if self.preload_all:
            print(f'Preloading {len(item_ids)} items.')
            with multiprocessing.Pool(oads_access.n_processes) as pool:
                # results = list(tqdm.tqdm(pool.map(self.iterate, [idx for idx in range(len(item_ids))]), total=len(item_ids)))
                results = list(tqdm.tqdm(pool.imap(self.iterate, [idx for idx in range(len(item_ids))]), total=len(item_ids)))

            # results = p_map(self.iterate, [idx for idx in range(len(item_ids))])
            for idx, tup in results:
                self.tupels[idx] = tup
            

    def iterate(self, idx):
        if self.use_crops:
            image_name, index = self.item_ids[idx]
            tup = self.oads_access.load_crop_from_image(
                image_name=image_name, index=index, force_recompute=self.force_recompute)
        else:
            image_name = self.item_ids[idx]
            tup = self.oads_access.load_image(image_name=image_name)

        if tup is None:
            return None, None
        
        # tup[0].load()
        return idx, tup

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, idx):
        if self.preload_all:
            tup = self.tupels[idx]
        else:
            if self.use_crops:
                image_name, index = self.item_ids[idx]
                tup = self.oads_access.load_crop_from_image(
                    image_name=image_name, index=index, force_recompute=self.force_recompute)
            else:
                image_name = self.item_ids[idx]
                tup = self.oads_access.load_image(image_name=image_name)

        if tup is None:
            return None

        img, label = tup
        del tup

        if img is None or label is None:
            return None

        if self.transform:
            img = self.transform(img)

        img = img.float()

        if self.target == 'label':
            label = label['classId']
            if self.class_index_mapping is not None:
                label = self.class_index_mapping[label]

        elif self.target == 'image':
            label = img

        else:
            label = np.array([])

        if self.target_transform:
            label = self.target_transform(label)

        if self.return_index:
            return (img, label, idx)
        
        return (img, label)
