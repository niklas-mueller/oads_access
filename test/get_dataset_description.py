import argparse
import os
import sys
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import numpy as np
# from model import visTensor, TestModel, evaluate
from result_manager.result_manager import ResultManager
from oads_access.oads_access import OADS_Access, OADSImageDataset, plot_image_in_color_spaces
# from retina_model import RetinaCortexModel
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50, alexnet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from pytorch_utils.pytorch_utils import train, evaluate, visualize_layers, collate_fn
import multiprocessing

if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', help='Path to input directory.')
    parser.add_argument('--output_dir', help='Path to output directory.')
    parser.add_argument(
        '--force_recrop', help='Whether to recompute the crops from the images.', default=False)
    parser.add_argument(
        '--get_visuals', help='Whether to run some visual instead of training.', default=False)
    parser.add_argument('--image_representation',
                        help='Way images are represented. Can be `RGB`, `COC` (color opponent channels)', default='RGB')
    parser.add_argument('--n_processes', help='Number of processes to use.',
                        default=multiprocessing.cpu_count())

    args = parser.parse_args()

    # Check if GPU is available
    if not torch.cuda.is_available():
        print("GPU not available. Exiting ....")
        device = torch.device('cpu')
        # exit(1)
    else:
        device = torch.device("cuda")
        print("Using GPU!")

    # Setting weird stuff
    torch.multiprocessing.set_start_method('spawn')
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # initialize data access
    # home = '../../data/oads/mini_oads/'
    size = (400, 400)

    if args.image_representation == 'RGB':
        print(f"Image representation: RGB. File format: .tiff")
        file_formats = ['.tiff']
        convert_to_opponent_space = False
    elif args.image_representation == 'COC':
        print(f"Image representation: color opponent space. File format: .tiff")
        #file_formats = ['.npy']
        # file_formats = ['.ARW']
        file_formats = ['.tiff']
        convert_to_opponent_space = True
    else:
        print(f"Image representation is not know. Exiting.")
        exit(1)

    home = args.input_dir
    oads = OADS_Access(home, file_formats=file_formats, use_avg_crop_size=True, n_processes=int(
        args.n_processes), image_representation=args.image_representation, min_size_crops=size, max_size_crops=size)

    # Compute crops if necessary
    if args.force_recrop:
        print(f"Recomputing crops.")
        oads.file_formats = ['.ARW']
        oads.min_size_crops = size
        oads.max_size_crops = size
        oads.prepare_crops(
            convert_to_opponent_space=convert_to_opponent_space, overwrite=True)
        oads.file_formats = file_formats

    result_manager = ResultManager(root=args.output_dir)

    # if args.get_visuals:
    #     fig = oads.plot_image_size_distribution(use_crops=True, figsize=(30, 30))
    #     result_manager.save_pdf(figs=[fig], filename='image_size_distribution.pdf')

    # get train, val, test split, using crops if specific size
    train_ids, val_ids, test_ids = oads.get_train_val_test_split_indices(
        use_crops=True)
    print(f"Loaded data with train_ids.shape: {len(train_ids)}")
    print(f"Loaded data with val_ids.shape: {len(val_ids)}")
    print(f"Loaded data with test_ids.shape: {len(test_ids)}")

    # fig = oads.plot_image_size_distribution(use_crops=True, figsize=(20, 20))
    # result_manager.save_pdf(figs=[fig], filename='image_size_distribution.pdf')
    # exit(1)

    if args.get_visuals:
        print(f"Getting visuals: plot_image_in_color_spaces")
        figs = []
        results = oads.apply_per_crop(lambda x: plot_image_in_color_spaces(
            np.array(x[0]), cmap_opponent='gray'), max_number_crops=50)
        for _, images in results.items():
            for _, fig in images.items():
                figs.append(fig)

        result_manager.save_pdf(
            figs=figs, filename=f'image_in_color_spaces_{size[0]}x{size[1]}.pdf')

    n_input_channels = 3
    output_channels = len(oads.get_class_mapping())
    class_index_mapping = {key: index for index, key in enumerate(
        list(oads.get_class_mapping().keys()))}

    batch_size = 2

    # print(f"Getting dataset stats")
    # means, stds = oads.get_dataset_stats()
    # if not means.shape == (3,):
    #     print(means.shape, stds.shape)

    # Get the custom dataset and dataloader
    print(f"Getting data loaders")
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(means.mean(axis=0), stds.mean(axis=0))   # TODO fix this
    ])

    ids = np.concatenate((train_ids, test_ids, val_ids))
    dataset = OADSImageDataset(oads_access=oads, item_ids=ids, use_crops=True,
                                    class_index_mapping=class_index_mapping, transform=transform, device=device)

    loader = DataLoader(dataset, collate_fn=collate_fn,
                             batch_size=batch_size, shuffle=True, num_workers=oads.n_processes)
    print(f"Loaded data loaders")

    # Get dataset description

    print("Getting annotation sizes")
    annotation_sizes = oads.apply_per_crop_annotation(oads.get_annotation_size)

    sizes_flat = np.array([size for _, image_names in annotation_sizes.items() for _, sizes in image_names.items() for size in sizes])

    print("Creating plot")
    # (300, 400) -> (300+, 400+)
    number_sizes_bigger = np.zeros((1000, 1000))
    for size in sizes_flat:
        number_sizes_bigger[size[0]:,size[1]:] += 1
        # for x,y in zip(range(999, 0, -1), range(999, 0, -1)):
        #     if x > size[0] and y > size[1]:
        #         number_sizes_bigger[x][y] += 1
        #     else:
        #         break

    # print(sizes_flat > (1000,1000))
    fig, ax = plt.subplots(1,1, figsize=(15,15))
    bar = ax.imshow(number_sizes_bigger)
    fig.colorbar(bar, ax=ax)
    result_manager.save_pdf(figs=[fig], filename='crop_size_plot.pdf')
    exit(1)

    description = {
        'number_images': len(oads.image_names),
        'number_crops': len(ids),
        # how many crops/annotations are left for a given crop size
        
    }

    print(description)