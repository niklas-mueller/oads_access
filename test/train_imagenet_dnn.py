import os
nproc = 12

os.environ["OMP_NUM_THREADS"] = str(nproc)
os.environ["OPENBLAS_NUM_THREADS"] = str(nproc)
os.environ["MKL_NUM_THREADS"] = str(nproc)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(nproc)
os.environ["NUMEXPR_NUM_THREADS"] = str(nproc)

import argparse
import os
import sys
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import tqdm
# from model import visTensor, TestModel, evaluate
from result_manager.result_manager import ResultManager
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50, alexnet, vgg16, vgg11_bn
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import time
from pytorch_utils.pytorch_utils import train, evaluate, visualize_layers, collate_fn, ToJpeg, EdgeResize, ToOpponentChannel, ToRGBEdges, ToRetinalGanglionCellSampling, get_confusion_matrix, get_result_figures, ImageNetKaggle
from pytorch_utils.resnet10 import ResNet10
import multiprocessing
from PIL import Image

####### FLEX CONV
import sys
home_path = os.path.expanduser('~')
sys.path.append(f'{home_path}/projects/oads_flexconv')
from models.resnet import ResNet_image
from omegaconf import OmegaConf
###########

if __name__ == '__main__':

    c_time = datetime.now().strftime("%d-%m-%y-%H:%M:%S")

    # Instantiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_dir', help='Path to input directory.', default='/projects/2/managed_datasets/Imagenet')
    parser.add_argument(
        '--output_dir', help='Path to output directory.', default=f'{home_path}/projects/imagenet_results/{c_time}')
    parser.add_argument(
        '--model_name', help='Model name to save the model under', default=f'model_{c_time}')
    parser.add_argument(
        '--n_epochs', help='Number of epochs for training.', default=30)
    parser.add_argument(
        '--optimizer', help='Optimizer to use for training', default='adam')
    parser.add_argument(
        '--model_path', help='Path to model to continue training on.', default=None)
    parser.add_argument(
        '--model_type', help='Model to use for training.', default='resnet50')
    # parser.add_argument(
    #     '--image_representation', help='Way images are represented. Can be `RGB`, `COC` (color opponent channels), or `RGBCOC` (stacked RGB and COC)', default='RGB')
    parser.add_argument(
        '--n_processes', help='Number of processes to use.', default=18)
    parser.add_argument(
        '--batch_size', help='Batch size for training.', default=256)
    parser.add_argument(
        '--image_size', help='Batch size for training.', default=224)
    parser.add_argument(
        '--random_state', help='Random state for train test split function. Control for crossvalidation. Can be integer or None. Default 42', default=42)
    # parser.add_argument(
    #     '-use_jpeg', help='Whether to use JPEG Compression or not', action='store_true')
    # parser.add_argument(
    #     '--jpeg_quality', help='Batch size for training.', default=90)
    # parser.add_argument(
    #     '-new_dataloader', help='Whether to use new dataloader or use the path in --dataloader_path to load existing ones. If new_dataloader is given, --dataloader_path will be use as target directory to store dataloaders', action='store_true')
    # parser.add_argument(
    #     '--dataloader_path', help='Path to a directory where the dataloaders can be stored from', default='/home/niklas/projects/oads_access/dataloader')
    # parser.add_argument(
    #     '-preload_all', help='Whether to preloader all images into memory before. Will require around 260GB of RAM but will boost performance a lot.', action='store_true')
    # parser.add_argument(
    #     '-no_normalization', help='Whether to test', action='store_true')
    parser.add_argument(
        '-test', help='Whether to test', action='store_true')

    args = parser.parse_args()

    # Check if GPU is available
    if not torch.cuda.is_available():
        print("GPU not available. Exiting ....")
        device = torch.device('cpu')
        exit(1)
    else:
        device = torch.device("cuda:1")
        torch.cuda.empty_cache()
        print(f"Using GPU: {device}!")

    # Setting weird stuff
    torch.multiprocessing.set_start_method('spawn')

    size = (int(args.image_size), int(args.image_size))

    n_input_channels = 3
    output_channels = 1000

    result_manager = ResultManager(root=args.output_dir)

    batch_size = int(args.batch_size)

    #Imagenet-1k
    mean_image = [0.485, 0.456, 0.406]
    std_image = [0.229, 0.224, 0.225]

    transform_list = []
    transform_list.append(transforms.Resize(256))
    transform_list.append(transforms.CenterCrop(int(args.image_size)))
    # if image_representation == 'coc' or image_representation == 'COC':
    #     transform_list.append(ToOpponentChannel())
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean_image, std_image))
    transform = transforms.Compose(transform_list)


    if 'ILSVRC' in os.listdir(args.input_dir):
        imagenet = ImageNetKaggle(args.input_dir, split='train', transform=transform)
        train_dataset, test_dataset  = random_split(imagenet, [1153051, 128116])
        val_dataset = ImageNetKaggle(args.input_dir, split='val', transform=transform)

    elif 'managed_datasets' in args.input_dir:
        imagenet = ImageNetKaggle(args.input_dir, split='train', transform=transform, root_extension="", return_index=True, val_label_filepath='/home/mullern/projects/data/imagenet-1k/')
        train_dataset, test_dataset  = random_split(imagenet, [1153051, 128116])
        val_dataset = ImageNetKaggle(args.input_dir, split='validation', transform=transform, root_extension="", return_index=True, val_label_filepath='/home/mullern/projects/data/imagenet-1k/')

    elif 'imagenet-object-localization-challenge.zip' in os.listdir(args.input_dir):
        print('ZIP is currently not functional.')
        exit(1)
        # train_dataset = ImageNetKaggleZip(args.input_dir, split='train', transform=transform)
        # # test_dataset = ImageNetKaggleZip(args.input_dir, split='test', transform=transform)
        # val_dataset = ImageNetKaggleZip(args.input_dir, split='val', transform=transform)
    else:
        imagenet = ImageFolder(root=os.path.join(args.input_dir, 'train'), transform=transform)
        train_dataset, test_dataset  = random_split(imagenet, [1153051, 128116])
        val_dataset = ImageFolder(root=os.path.join(args.input_dir, 'val'), transform=transform)

    
    trainloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=int(args.n_processes))
    testloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False, num_workers=int(args.n_processes))
    valloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False, num_workers=int(args.n_processes))

    # Initialize model
    if args.model_type == 'resnet10':
        model = ResNet10(n_output_channels=output_channels, n_input_channels=n_input_channels)
    elif args.model_type == 'resnet18':
        model = resnet18()
        model.conv1 = torch.nn.Conv2d(in_channels=n_input_channels, out_channels=model.conv1.out_channels, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        model.fc = torch.nn.Linear(
            in_features=512, out_features=output_channels, bias=True)
    elif args.model_type == 'resnet50':
        model = resnet50()
        model.conv1 = torch.nn.Conv2d(in_channels=n_input_channels, out_channels=model.conv1.out_channels, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        model.fc = torch.nn.Linear(
            in_features=2048, out_features=output_channels, bias=True)
        
    ####################################### FLEX CONV
    elif args.model_type == 'flex_resnet50':
        cfg = OmegaConf.load(f'{home_path}/projects/oads_flexconv/cfg/oads_config.yaml')

        cfg.net.data_dim = 2

        model = ResNet_image(in_channels= n_input_channels,
        out_channels= output_channels,
        net_cfg=cfg.net,
        kernel_cfg= cfg.kernel,
        conv_cfg= cfg.conv,
        mask_cfg= cfg.mask,)

        # print(model)
    #######################################
    elif args.model_type == 'alexnet':
        # print('AlesNet is not supported ATM.')
        # exit(1)
        model = alexnet()
        model.classifier[6] = torch.nn.Linear(4096, output_channels, bias=True)
    elif args.model_type == 'vgg16':
        model = vgg16()
        model.features[0] = torch.nn.Conv2d(in_channels=n_input_channels, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        model.classifier[-1] = torch.nn.Linear(4096,
                                               output_channels, bias=True)
    elif args.model_type == 'vgg11_bn':
        model = vgg11_bn()
        model.features[0] = torch.nn.Conv2d(in_channels=n_input_channels, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        model.classifier[-1] = torch.nn.Linear(4096,
                                               output_channels, bias=True)

    elif args.model_type == 'retina_cortex':
        print('RetinaCortexModel is not supported at the moment')
        exit(1)
        # model = RetinaCortexModel(n_retina_layers=2, n_retina_in_channels=n_input_channels, n_retina_out_channels=2, retina_width=32,
        #                         input_shape=size, kernel_size=(9,9), n_vvs_layers=2, out_features=output_channels, vvs_width=32)

    print(f"Created model {args.model_type}")

    results = {}
    n_epochs = int(args.n_epochs)
    if args.model_path is not None:
        print(f"Loading model state {args.model_path}")
        try:
            model.load_state_dict(torch.load(args.model_path))
        except:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(args.model_path))

        if 'best_model_' in args.model_path:
            res_path, ident = args.model_path.split('best_model_')
        elif 'final_model_' in args.model_path:
            res_path, ident = args.model_path.split('final_model_')
        else:
            res_path, ident = args.model_path.split('model_')

        ident = ident.split('.pth')[0]
        if os.path.exists(os.path.join(res_path, f'training_results_{ident}.yml')):
            results = result_manager.load_result(filename=f'training_results_{ident}.yml', path=res_path)
            
            pre_epochs = 0
            pre_epochs = max([int(x.split('epoch-')[-1]) for x in results.keys() if 'epoch-' in x])
            n_epochs = [x for x in range(pre_epochs+1, pre_epochs+1+n_epochs)]

    # Use DataParallel to make use of multiple GPUs if available
    # if type(model) is not torch.nn.DataParallel:
    #     # model = model.to('cuda:1')
    #     model = torch.nn.DataParallel(model, device_ids=[0,1])
    # else:
    model = model.to(device)  # , dtype=torch.float32

    criterion = nn.CrossEntropyLoss()
    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9)

    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5)
    
    # info = {
    #     'training_indices': train_ids,
    #     'testing_indices': test_ids,
    #     'validation_indices': val_ids,
    #     'optimizer': str(optimizer),
    #     'scheduler': str(plateau_scheduler),
    #     'model': str(model),
    #     'args': str(args),
    #     'device': str(device),
    #     'criterion': str(criterion),
    #     'size': size,
    #     'transform': transform,
    #     'file_formats': file_formats,
    #     'image_representation': args.image_representation,
    #     'input_dir': args.input_dir,
    #     'output_dir': args.output_dir
    # }
    # if len(results) == 0:
    #     result_manager.save_result(
    #         result=info, filename=f'fitting_description_{c_time}.yaml')


    train(model=model, trainloader=trainloader, valloader=valloader, device=device, results=results,
            loss_fn=criterion, optimizer=optimizer, n_epochs=n_epochs, result_manager=result_manager,
            testloader=testloader, plateau_lr_scheduler=plateau_scheduler, current_time=c_time)
