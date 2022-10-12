import os, sys, json
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

class OADS_Access():
    def __init__(self, basedir: str):
        self.basedir = basedir

        self.datasets = [x for x in os.listdir(self.basedir) if os.path.isdir(os.path.join(self.basedir, x))]

        
        self.image_names = {
            name: [x for x in os.listdir(os.path.join(self.basedir, name, 'img'))] for name in self.datasets
        }

    def get_meta_info(self):
        if not hasattr(self, 'meta'):
            with open(os.path.join(self.basedir, 'meta.json'), 'r') as f:
                meta = json.load(f)
            self.meta = meta

        return self.meta

    def get_class_mapping(self):
        return {x['id']: x['title'] for x in self.get_meta_info()['classes']}

    def get_annotation(self, dataset_name, image_name):
        path = os.path.join(self.basedir, dataset_name, 'ann', f"{image_name}.json")

        if os.path.exists(path):
            with open(path, 'r') as f:
                content = json.load(f)
            return content
        else:
            return None

    def get_data_iterator(self, dataset_names=None):
        if dataset_names is None:
            dataset_names = self.datasets

        data = []
        for dataset_name in dataset_names:
            for image_name in self.image_names[dataset_name]:
                
                img = Image.open(os.path.join(self.basedir, dataset_name, 'img', image_name))
                label = self.get_annotation(dataset_name=dataset_name, image_name=image_name)
                tup = (img, label)
                data.append(tup)
        return data

    def get_maximum_annotation_size(self):
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


# create crops from image
def get_image_crop(img, object, min_size):
    ((left, top), (right, bottom)) = object['points']['exterior']
    if right-left < min_size[0]:
        left = left - min_size[0] / 2
        right = right + min_size[0] / 2
    if bottom-top < min_size[1]:
        top = top - min_size[1] / 2
        bottom = bottom + min_size[1] / 2
    crop = img.crop((left, top, right, bottom))
    return crop

def plot_crops_from_data_tuple(data_tuple, min_size=(0, 0), figsize=(18,30)):
    img = data_tuple[0]
    label = data_tuple[1]
    
    _n = len(label['objects'])
    if _n == 0:
        print("No labels present for image!")
        return None
    _n = int(np.ceil(np.sqrt(_n)))
    fig, ax = plt.subplots(_n,_n, figsize=figsize)

    for axis, obj in zip(ax.flatten(), label['objects']):
        crop = get_image_crop(img, obj, min_size=min_size)
        axis.imshow(crop)
        axis.set_title(obj['classTitle'])
        axis.axis('off')

    return fig


def add_label_box_to_axis(label, ax, color='r', add_title=False):
    if len(label['objects']) > 0:
            img_height, img_width = label['size']['height'], label['size']['width']
            for obj in label['objects']:
                if obj['geometryType'] == 'rectangle':
                    ((left, top), (right, bottom)) = obj['points']['exterior']
                    rec = Rectangle(xy=(left, top), height=bottom-top, width=right-left, fill=False, color=color)
                    ax.add_patch(rec)

                    if add_title:
                        ax.annotate(text=obj['classTitle'], xy=(left, top-10), fontsize='x-small')


