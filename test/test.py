from oads_access.oads_access import OADS_Access, OADSImageDataset, plot_image_in_color_spaces
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorch_utils.pytorch_utils import train, evaluate, visualize_layers, collate_fn
import torch
import time, multiprocessing
import argparse

#home = '../../data/oads'
parser = argparse.ArgumentParser()
parser.add_argument('--home', help='Path to input directory.')

args = parser.parse_args()

home = args.home

size = (400,400)
oads = OADS_Access(home, max_size_crops=size, min_size_crops=size)

# print(len(oads.image_names))

image_name = list(oads.image_names.keys())[0]

# print(oads.image_names[image_name])

# print(oads.get_annotation(image_name=image_name))


# print(oads.load_image(image_name=image_name))

# print(oads.load_crop_from_image(image_name=image_name, index=0))

# oads.min_size_crops = size
# oads.max_size_crops = size
# oads.prepare_crops()


train_ids, val_ids, test_ids = oads.get_train_val_test_split_indices(use_crops=True)


input_channels = size[0] #np.array(train_data[0][0]).shape[-1]
output_channels = len(oads.get_class_mapping())
class_index_mapping = {key: index for index, key in enumerate(list(oads.get_class_mapping().keys()))}

batch_size = 32


# Get the custom dataset and dataloader
print(f"Getting data loaders")
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(means.mean(axis=0), stds.mean(axis=0))   # TODO fix this
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device {device}")

traindataset = OADSImageDataset(oads_access=oads, item_ids=train_ids, use_crops=True, class_index_mapping=class_index_mapping, transform=transform, device=device)
valdataset = OADSImageDataset(oads_access=oads, item_ids=val_ids, use_crops=True, class_index_mapping=class_index_mapping, transform=transform, device=device)
testdataset = OADSImageDataset(oads_access=oads, item_ids=test_ids, use_crops=True, class_index_mapping=class_index_mapping, transform=transform, device=device)


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
trainloader = DataLoader(traindataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=0)#multiprocessing.cpu_count())
valloader = DataLoader(valdataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=0)#multiprocessing.cpu_count())
testloader = DataLoader(testdataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=0)#multiprocessing.cpu_count())


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