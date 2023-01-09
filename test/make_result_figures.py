from result_manager.result_manager import ResultManager
from pytorch_utils.pytorch_utils import get_result_figures, get_confusion_matrix, collate_fn
from oads_access.oads_access import OADS_Access, OADSImageDataset
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision import transforms

res_dir = '/home/nmuller/projects/oads_results/coc/jpeg'

# identifier = "Tue_Nov_22_14:04:03_2022"
# identifier = "Fri_Dec__2_15:27:02_2022"
# identifier = "Mon_Dec__5_16:14:01_2022"
identifier = "Fri_Dec__9_13:07:29_2022"

result_manager = ResultManager(res_dir)

result = result_manager.load_result(os.path.join(res_dir, f'training_results_{identifier}.yml'))

# if not torch.cuda.is_available():
#     print("GPU not available. Exiting ....")
#     device = torch.device('cpu')
#     # exit(1)
# else:
#     device = torch.device("cuda")
#     print("Using GPU!")
device = torch.device('cpu')
#################
model = resnet50()
model.fc = torch.nn.Linear(
    in_features=2048, out_features=19, bias=True)



model.load_state_dict(torch.load(os.path.join(res_dir, f'best_model_{identifier}.pth')))
model = model.to(device)
##################

home = '/home/nmuller/projects/data/oads'

oads = OADS_Access(home, file_formats=['.tiff'], use_avg_crop_size=True, n_processes=4, min_size_crops=(400,400), max_size_crops=(400,400))

train_ids, val_ids, test_ids = oads.get_train_val_test_split_indices(
        use_crops=True)

classes = oads.get_class_mapping()
class_index_mapping = {key: index for index, key in enumerate(
        list(classes.keys()))}
index_class_mapping = {index: classes[key] for index, key in enumerate(
        list(classes.keys()))}


batch_size = 2

transform = transforms.Compose([
        # ToJpeg() if args.use_jpeg else None,
        # ToOpponentChannel() if convert_to_opponent_space else None,
        transforms.ToTensor(),
        # transforms.Normalize(means.mean(axis=0), stds.mean(axis=0))   # TODO fix this
    ])

traindataset = OADSImageDataset(oads_access=oads, item_ids=train_ids, use_crops=True,
                                    class_index_mapping=class_index_mapping, transform=transform, device=device)
valdataset = OADSImageDataset(oads_access=oads, item_ids=val_ids, use_crops=True,
                                class_index_mapping=class_index_mapping, transform=transform, device=device)
testdataset = OADSImageDataset(oads_access=oads, item_ids=test_ids, use_crops=True,
                                class_index_mapping=class_index_mapping, transform=transform, device=device)

trainloader = DataLoader(traindataset, collate_fn=collate_fn,
                            batch_size=batch_size, shuffle=True, num_workers=oads.n_processes)
valloader = DataLoader(valdataset, collate_fn=collate_fn,
                        batch_size=batch_size, shuffle=True, num_workers=oads.n_processes)
testloader = DataLoader(testdataset, collate_fn=collate_fn,
                        batch_size=batch_size, shuffle=True, num_workers=oads.n_processes)



##############
if os.path.exists(os.path.join(res_dir, f"confusion-matrix_{identifier}.npy")):
        confusion_matrix = np.load(os.path.join(res_dir, f"confusion-matrix_{identifier}.npy"))
else:
        confusion_matrix = get_confusion_matrix(model=model, loader=testloader, n_classes=19, device=device)
        np.save(arr=confusion_matrix, file=os.path.join(res_dir, f"confusion-matrix_{identifier}.npy"))

get_result_figures(result, result_manager=result_manager, confusion_matrix=confusion_matrix, classes=index_class_mapping)