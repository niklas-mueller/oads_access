from oads_access.oads_access import OADS_Access, OADSImageDataset, plot_image_in_color_spaces
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorch_utils.pytorch_utils import train, evaluate, visualize_layers, collate_fn
import torch
import time, multiprocessing
import argparse
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from result_manager.result_manager import ResultManager
import numpy as np

#home = '../../data/oads'
parser = argparse.ArgumentParser()
parser.add_argument('--home', help='Path to input directory.', default='/home/niklas/projects/data/oads')

args = parser.parse_args()

home = args.home
result_manager = ResultManager(root='./analysis')
size = (400,400)
oads = OADS_Access(home, exclude_classes=['MASK', 'Xtra Class 2', 'Xtra Class 1'])#, max_size_crops=size, min_size_crops=size)

# print(len(oads.image_names))

image_name = list(oads.image_names.keys())[0]

fig, ax = plt.subplots(1,1, figsize=(10,8))
classes = []
height = []
for _class, images in oads.images_per_class.items():
    classes.append(_class)
    height.append(len(images))

inds = np.argsort(height)
ax.bar(x=np.array(classes)[inds], height=np.array(height)[inds])
ax.set_xticklabels(classes, rotation=45)
result_manager.save_pdf(figs=[fig], filename='oads_image_statistics.pdf')


# print(oads.image_names[image_name])

# print(oads.get_annotation(image_name=image_name))


# print(oads.load_image(image_name=image_name))

# print(oads.load_crop_from_image(image_name=image_name, index=0))

# oads.min_size_crops = size
# oads.max_size_crops = size
# oads.prepare_crops()

use_crops = True

train_ids, val_ids, test_ids = oads.get_train_val_test_split_indices(use_crops=use_crops)

print(len(train_ids))

input_channels = size[0] #np.array(train_data[0][0]).shape[-1]
output_channels = len(oads.get_class_mapping())
class_index_mapping = {key: index for index, key in enumerate(list(oads.get_class_mapping().keys()))}

batch_size = 8


# Get the custom dataset and dataloader
print(f"Getting data loaders")
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(means.mean(axis=0), stds.mean(axis=0))   # TODO fix this
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device {device}")

traindataset = OADSImageDataset(oads_access=oads, item_ids=train_ids, use_crops=use_crops, class_index_mapping=class_index_mapping, transform=transform, device=device)
valdataset = OADSImageDataset(oads_access=oads, item_ids=val_ids, use_crops=use_crops, class_index_mapping=class_index_mapping, transform=transform, device=device)
testdataset = OADSImageDataset(oads_access=oads, item_ids=test_ids, use_crops=use_crops, class_index_mapping=class_index_mapping, transform=transform, device=device)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

# trainloader = MultiEpochsDataLoader(traindataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count())
# valloader = MultiEpochsDataLoader(valdataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count())
# testloader = MultiEpochsDataLoader(testdataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count())
trainloader = DataLoader(traindataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count())
valloader = DataLoader(valdataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count())
testloader = DataLoader(testdataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count())


print("Done getting data loader")


being = time.time()
for i, data in enumerate(trainloader):
    print(i)
    pass
end = time.time()

print(end - being)



######### To be tested with more memory
# print(len(oads.get_data_list()))


# print(f"Getting dataset stats")
# means, stds = oads.get_dataset_stats()
# if not means.shape == (3,):
#     print(means.shape, stds.shape)