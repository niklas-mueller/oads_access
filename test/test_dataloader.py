from torchvision import transforms
import torch,sys
from torch.utils.data import DataLoader
from oads_access.oads_access import OADSImageDataset, OADS_Access
sys.path.append('../..')
from pytorch_utils.pytorch_utils.pytorch_utils import collate_fn
import time
import numpy as np

home = "/home/niklas/projects/data/oads/mini_oads_2/datasets"

size = (200, 200)
oads = OADS_Access(home, file_formats=['.ARW'], min_size_crops=size, max_size_crops=size)


# oads.prepare_crops()

train_ids, val_ids, test_ids = oads.get_train_val_test_split_indices(use_crops=True)


transform = transforms.Compose([
    transforms.ToTensor(),
])

class_index_mapping = {key: index for index, key in enumerate(list(oads.get_class_mapping().keys()))}
device = torch.device('cuda')
batch_size = 2

traindataset = OADSImageDataset(oads_access=oads, item_ids=train_ids, use_crops=True, class_index_mapping=class_index_mapping, transform=transform, device=device)
valdataset = OADSImageDataset(oads_access=oads, item_ids=val_ids, use_crops=True, class_index_mapping=class_index_mapping, transform=transform, device=device)
testdataset = OADSImageDataset(oads_access=oads, item_ids=test_ids, use_crops=True, class_index_mapping=class_index_mapping, transform=transform, device=device)

trainloader = DataLoader(traindataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=1)
valloader = DataLoader(valdataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=1)
testloader = DataLoader(testdataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=1)


sizes = []

print("Starting iterating")
start = time.time()
for index, (img, label) in enumerate(trainloader):
    sizes.append(img.shape)

end = time.time()
print("Done iterating")

print(f"Time: {end-start}")
print(f"Len: {len(sizes)}")