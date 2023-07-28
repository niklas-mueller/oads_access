import sys
import numpy as np
# sys.path.append('/home/niklas/projects/style_transfer')
from oads_access.oads_access import OADS_Access, OADSImageDataset
from pytorch_utils.pytorch_utils import collate_fn
from tqdm import tqdm
# from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader

# from oads_access.oads_style_transfer import run_style_transfer, image_loader, imshow_tensor, UnNormalize
from oads_access.oads_style_transfer import PytorchNeuralStyleTransfer
from torch.autograd import Variable
import copy
import os
from result_manager.result_manager import ResultManager
from torchvision.models import resnet50

def test_on_model(result_dir, model, oads_transform):
    result_manager = ResultManager(root=result_dir)

    image_names = os.listdir(result_dir)

    n_total = 0

    any_correct = 0
    texture_correct = 0
    shape_correct = 0

    for image_name in image_names:
        img = result_manager.load_result(filename=image_name)
        shape_desc, texture_desc = image_name.split('-')
        shape_class, shape_image_id, shape_index = shape_desc.split('_')
        texture_class, texture_image_id, texture_index = texture_desc.split('_')
        texture_index = texture_index.split('.')[0]


        scores = model(oads_transform(img).unsqueeze(0))

        _, predictions = scores.max(1)
        # # predictions
        print(f'Shape: {shape_class}, texture: {texture_class}, pred: {index_label_mapping[predictions.cpu().numpy()[0]]}')
        pred_class = index_label_mapping[predictions.cpu().numpy()[0]]
        if pred_class == shape_class:
            any_correct += 1
            shape_correct += 1
        elif pred_class == texture_class:
            any_correct += 1
            texture_correct += 1
        else:
            pass

        n_total += 1

    return any_correct, shape_correct, texture_correct, n_total



def get_oads_model(device):
    n_input_channels = 3
    output_channels = 19

    model_path = f'{os.path.expanduser("~")}/projects/oads_results_snellius/normalized/oads_results/resnet50/rgb/2023-03-24-17:09:46/best_model_24-03-23-17:09:53.pth'
    model = resnet50()
    model.conv1 = nn.Conv2d(in_channels=n_input_channels, out_channels=model.conv1.out_channels, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
    model.fc = nn.Linear(
        in_features=2048, out_features=output_channels, bias=True)

    model = nn.DataParallel(model,)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device=device)

    model.eval()

    return model

def get_oads_dataloader(oads, classes, class_index_mapping, oads_transform):
    all_filenames = {
        c: [x.split('_') for x in oads.images_per_class[c]][:200] for c in classes
    }

    ids = [(x[0], int(x[1])) for c,l in all_filenames.items() for x in l]
    np.random.shuffle(ids)

    dataset = OADSImageDataset(oads_access=oads, item_ids=ids, use_crops=True, preload_all=False, target='label', return_index=True,
                                class_index_mapping=class_index_mapping, transform=oads_transform, device=device)

    oads_loader = DataLoader(dataset, collate_fn=collate_fn,
                                    batch_size=1, shuffle=False, num_workers=oads.n_processes)
    
    return oads_loader, ids


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    if not torch.cuda.is_available():
        exit(1)

    home_path = os.path.expanduser('~')

    style_weight_factor = float(sys.argv[1])
    # style_weight_factor = 1.0
    # style_weight_factor = 0.01
    # style_weight_factor = 100.0

    # result_dir = f'{home_path}/projects/oads_access/results/pytorch_neural_style_transfer_oads'
    # result_dir = f'{home_path}/projects/fmgstorage/oads_texture_shape_images/larger/{str(style_weight_factor)}'
    result_dir = f'/mnt/z/Projects/2023_Scholte_FMG1441/oads_texture_shape_images/larger/{str(style_weight_factor)}'


    basedir = f'{home_path}/projects/data/oads/'

    oads = OADS_Access(basedir=basedir)

    unloader = transforms.ToPILImage()  # reconvert into PIL image

    criterion = nn.CrossEntropyLoss()

    mean = [0.3410, 0.3123, 0.2787]
    std = [0.2362, 0.2252, 0.2162]

    imsize = (400, 400)  # use small size if no gpu

    transform = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])
    
    oads_transform = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])  # transform it into a torch tensor
    
    class_index_mapping = {}
    index_label_mapping = {}
    for index, (key, item) in enumerate(list(oads.get_class_mapping().items())):
        class_index_mapping[key] = index
        index_label_mapping[index] = item

    classes = ['Bollard', 'Van', 'Front door', 'Oma fiets', 'Truck', 'Bench', 'Lamppost', 'Tree']
    # print(classes[-2:])
    # exit(1)

    model = get_oads_model(device=device)

    oads_loader, ids = get_oads_dataloader(oads=oads, classes=classes, class_index_mapping=class_index_mapping, oads_transform=oads_transform)

    filenames = {}
    # image_names = os.listdir(f'{home_path}/projects/oads_access/texture_shape_dataset/larger/1.0')
    # image_names = os.listdir(f'{home_path}/projects/oads_access/texture_shape_dataset/small')

    # for x in image_names:
    #     c, id, index = x.split('.')[0].split('-')[0].split('_')
    #     if c in filenames:
    #         filenames[c].append((id, int(index)))
    #     else:
    #         filenames[c] = [(id, int(index))]

    # for key, value in filenames.items():
    #     filenames[key] = set(value)

    # filenames = {}
    # n_per_class = 4
    # counter = {
    #     c: 0 for c in classes
    # }

    # with tqdm(total=n_per_class*len(classes)) as pbar:
    #     for item in oads_loader:
    #         x = item[0]
    #         y = item[1]
    #         z = item[2]

    #         c = y.cpu().numpy()[0]

    #         if counter[index_label_mapping[c]] >= n_per_class:
    #             continue

    #         scores = model(x)
    #         _, predictions = scores.max(1)

    #         if c == predictions.cpu().numpy()[0]:
    #             name, index = ids[z]
    #             if index_label_mapping[c] in filenames:
    #                 filenames[index_label_mapping[c]].append((name, index))
    #             else:
    #                 filenames[index_label_mapping[c]] = [(name, index)]
    #             counter[index_label_mapping[c]] += 1
    #             pbar.update(1)

    #         if pbar.n == pbar.total:
    #             break

    filename_result_manager = ResultManager(root=f'/mnt/z/Projects/2023_Scholte_FMG1441/oads_texture_shape_images/larger')
    # filename_result_manager.save_result(result=filenames, filename='filenames.yaml')
    filenames = filename_result_manager.load_result(filename='filenames.yaml')


    
    result_manager = ResultManager(root=result_dir, verbose=False)

    style_transfer = PytorchNeuralStyleTransfer(img_size=imsize, device=device, mean=mean)

    style_transfer.weights = [x * style_weight_factor if i < len(style_transfer.weights)-1 else x for i, x in enumerate(style_transfer.weights)]

    verbose = False
    save_to_file = True
    results = {}

    max_n_images = 10

    counter = 0
    for c, tuples in tqdm(filenames.items(), desc='Classes'):
        # if c in classes[-2:]:
        #     continue
        # print(c)

        results[c] = {}
        for image, index in tqdm(tuples, desc='Images', leave=False):
            # print(c, image, index)

            results[c][f'{image}_{index}'] = {}

            for other_c, other_tuples in tqdm(filenames.items(), leave=False, desc='Other classes'):
                counter = 0
                results[c][f'{image}_{index}'][other_c] = {}

                if other_c == c:
                    continue
                for other_image, other_index in tqdm(other_tuples, leave=False, desc='Other images'):
                    if os.path.exists(os.path.join(result_dir, f'{c}_{image}_{index}-{other_c}_{other_image}_{other_index}.png')):
                        continue

                    if counter > max_n_images:
                        break
                    
                    results[c][f'{image}_{index}'][other_c][f'{other_image}_{other_index}'] = None
                    
                    img, lbl = oads.load_crop_from_image(image_name=image, index=index)
                    texture, _ = oads.load_crop_from_image(image_name=other_image, index=other_index)

                    style_image = Variable(style_transfer.prep(texture).unsqueeze(0)).to(device)
                    content_image = Variable(style_transfer.prep(img).unsqueeze(0)).to(device)

                    output = style_transfer.run(style_image=style_image, content_image=content_image, max_iter=500, verbose=verbose)
                    
                    
                    results[c][f'{image}_{index}'][other_c][f'{other_image}_{other_index}'] = output
                    
                    counter += 1
                    
                    if save_to_file:
                        result_manager.save_result(result=output, filename=f'{c}_{image}_{index}-{other_c}_{other_image}_{other_index}.png', overwrite=True)
                # result_manager.save_result(result=results, filename='stylized_images_more.pkl', overwrite=True)


    print("Done creating images.")



    # print("Testing model")
    # any_correct, shape_correct, texture_correct, n_total = test_on_model(result_dir=result_dir, model=model, oads_transform=oads_transform)
    # print(f'Responses:\n{any_correct}/{n_total} correct\n{shape_correct}/{n_total} shape correct\n{texture_correct}/{n_total} texture correct\n{shape_correct/any_correct} shape bias')