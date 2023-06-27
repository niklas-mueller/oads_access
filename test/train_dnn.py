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
from oads_access.oads_access import OADS_Access, OADSImageDataset, plot_image_in_color_spaces, OADSImageDatasetSharedMem
# from retina_model import RetinaCortexModel
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50, alexnet, vgg16, vgg11_bn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from pytorch_utils.pytorch_utils import train, evaluate, visualize_layers, collate_fn, ToJpeg, EdgeResize, ToOpponentChannel, ToRGBEdges, ToRetinalGanglionCellSampling, get_confusion_matrix, get_result_figures
from pytorch_utils.resnet10 import ResNet10
import multiprocessing
from PIL import Image
from oads_access.utils import plot_images, imscatter_all, loadmat

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
        '--input_dir', help='Path to input directory.', default='/home/niklas/projects/data/oads')
    parser.add_argument(
        '--output_dir', help='Path to output directory.', default=f'/home/niklas/projects/oads_access/results/dnn/{c_time}')
    parser.add_argument(
        '--model_name', help='Model name to save the model under', default=f'model_{c_time}')
    parser.add_argument(
        '--n_epochs', help='Number of epochs for training.', default=30)
    parser.add_argument(
        '-force_recrop', help='Whether to recompute the crops from the images.', action='store_true')
    parser.add_argument(
        '-get_visuals', help='Whether to run some visual instead of training.', action='store_true')
    parser.add_argument(
        '--optimizer', help='Optimizer to use for training', default='adam')
    parser.add_argument(
        '--model_path', help='Path to model to continue training on.', default=None)
    parser.add_argument(
        '--model_type', help='Model to use for training. Can be "test" or "retina_cortex"', default='resnet50')
    parser.add_argument(
        '--image_representation', help='Way images are represented. Can be `RGB`, `COC` (color opponent channels), or `RGBCOC` (stacked RGB and COC)', default='RGB')
    parser.add_argument(
        '--n_processes', help='Number of processes to use.', default=18)
    parser.add_argument(
        '--batch_size', help='Batch size for training.', default=256)
    parser.add_argument(
        '--image_size', help='Batch size for training.', default=400)
    parser.add_argument(
        '--random_state', help='Random state for train test split function. Control for crossvalidation. Can be integer or None. Default 42', default=42)
    parser.add_argument(
        '-use_jpeg', help='Whether to use JPEG Compression or not', action='store_true')
    parser.add_argument(
        '--jpeg_quality', help='Batch size for training.', default=90)
    parser.add_argument(
        '-new_dataloader', help='Whether to use new dataloader or use the path in --dataloader_path to load existing ones. If new_dataloader is given, --dataloader_path will be use as target directory to store dataloaders', action='store_true')
    parser.add_argument(
        '--dataloader_path', help='Path to a directory where the dataloaders can be stored from', default='/home/niklas/projects/oads_access/dataloader')
    parser.add_argument(
        '-preload_all', help='Whether to preloader all images into memory before. Will require around 260GB of RAM but will boost performance a lot.', action='store_true')
    parser.add_argument(
        '-no_normalization', help='Whether to test', action='store_true')
    parser.add_argument(
        '-test', help='Whether to test', action='store_true')

    args = parser.parse_args()

    # Check if GPU is available
    if not torch.cuda.is_available():
        print("GPU not available. Exiting ....")
        device = torch.device('cpu')
        exit(1)
    else:
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        print(f"Using GPU: {device}!")

    # Setting weird stuff
    torch.multiprocessing.set_start_method('spawn')
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # initialize data access
    # home = '../../data/oads/mini_oads/'
    size = (int(args.image_size), int(args.image_size))

    n_input_channels = 3

    use_rgbedges = False
    convert_to_opponent_space = False
    convert_to_gcs = False

    if args.image_representation == 'RGB':
        file_formats = ['.ARW', '.tiff']
    elif args.image_representation == 'COC':
        file_formats = ['.ARW', '.tiff']
        # file_formats = ['.tiff']
        convert_to_opponent_space = True
    elif args.image_representation == 'RGBEdges':
        file_formats = ['.ARW', '.npy']
        n_input_channels = 6
        use_rgbedges = True
    elif args.image_representation == 'RGB_GCS':
        file_formats = ['.ARW', '.tiff']
        n_input_channels =3
        convert_to_gcs = True
    elif args.image_representation == 'COC_GCS':
        file_formats = ['.ARW', '.tiff']
        n_input_channels =3
        convert_to_opponent_space = True
        convert_to_gcs = True

    else:
        print(f"Image representation is not know. Exiting.")
        exit(1)
    print(
        f"Image representation: {args.image_representation}. File format: {file_formats}")

    exclude_classes = ['MASK', "Xtra Class 1", 'Xtra Class 2']

    home = args.input_dir
    oads = OADS_Access(home, file_formats=file_formats, use_jpeg=bool(args.use_jpeg), n_processes=int(
        args.n_processes), exclude_classes=exclude_classes, jpeg_quality=int(args.jpeg_quality), use_rgbedges=use_rgbedges)

    # Compute crops if necessary
    if args.force_recrop:
        print(f"Recomputing crops.")
        oads.file_formats = ['.ARW']
        oads.prepare_crops(overwrite=True)
        oads.file_formats = file_formats

    result_manager = ResultManager(root=args.output_dir)


    output_channels = len(oads.get_class_mapping())
    class_index_mapping = {}
    index_label_mapping = {}
    for index, (key, item) in enumerate(list(oads.get_class_mapping().items())):
        class_index_mapping[key] = index
        index_label_mapping[index] = item

    batch_size = int(args.batch_size) # 256

    if convert_to_opponent_space:
        # OADS COC Crops (400,400) mean, std
        mean = [0.30080804, 0.02202087, 0.01321364]
        std =  [0.06359817, 0.01878176, 0.0180428]
    else:
        if not use_rgbedges:
            # OADS RGB Crops (400,400) mean, std
            mean = [0.3410, 0.3123, 0.2787]
            std = [0.2362, 0.2252, 0.2162]
        else:
            mean = None
            std = None

    # Get the custom dataset and dataloader
    print(f"Getting data loaders")
    transform_list = []

    if use_rgbedges:
        transform_list.append(EdgeResize(size))
    else:
        transform_list.append(transforms.Resize(size))

    if convert_to_gcs:
        transform_list.append(ToRetinalGanglionCellSampling())
        
    # Apply color opponnent channel representation
    if convert_to_opponent_space:
        transform_list.append(ToOpponentChannel())

    # Compute edge map and use as input instead
    # if use_rgbedges:
        # threshold_lgn_path = f'{os.path.expanduser("~")}/projects/lgnpy/ThresholdLGN.mat'
        # default_config_path = f'{os.path.expanduser("~")}/projects/lgnpy/lgnpy/CEandSC/default_config.yml'
        # threshold_lgn = loadmat(threshold_lgn_path)['ThresholdLGN']
        # transform_list.append(ToRGBEdges(threshold_lgn=threshold_lgn, default_config_path=default_config_path))


    transform_list.append(transforms.ToTensor())

    # if not bool(args.no_normalization) or not convert_to_rgbedges:
    if mean is not None and std is not None:
        transform_list.append(transforms.Normalize(mean, std))

    transform = transforms.Compose(transform_list)

    try:
        new_dataloader = False
        ids = np.load(os.path.join(args.dataloader_path, 'item_ids.npz'))
        train_ids, test_ids, val_ids = ids['train_ids'], ids['test_ids'], ids['val_ids']
        train_ids = [(x[0], int(x[1])) for x in train_ids]
        test_ids = [(x[0], int(x[1])) for x in test_ids]
        val_ids = [(x[0], int(x[1])) for x in val_ids]

        if args.test:
            train_ids = train_ids[:1000]
            test_ids = test_ids[:1000]
            val_ids = val_ids[:1000]

    except (Exception, FileNotFoundError) as e:
        print(e)
        new_dataloader = True        

    if args.new_dataloader or new_dataloader:
        # get train, val, test split, using crops if specific size
        random_state = int(args.random_state) if args.random_state != 'None' else None
        train_ids, val_ids, test_ids = oads.get_train_val_test_split_indices(use_crops=True, random_state=random_state, shuffle=True)

        # ids = {"train_ids": train_ids, "test_ids": test_ids, "val_ids": val_ids}
        # os.makedirs(args.dataloader_path, exist_ok=True)
        # np.savez_compressed(file=os.path.join(args.dataloader_path, 'item_ids.npz'), **ids)
        
    if use_rgbedges:
        train_ids = [('c0b2d8e1d3d39afe', 0)]
        val_ids = [('c0b2d8e1d3d39afe', 1)]
        test_ids = [('c0b2d8e1d3d39afe', 2)]

    print(f"Loaded data with train_ids.shape: {len(train_ids)}")
    print(f"Loaded data with val_ids.shape: {len(val_ids)}")
    print(f"Loaded data with test_ids.shape: {len(test_ids)}")
    print(f"Total of {len(train_ids) + len(val_ids) + len(test_ids)} datapoints.")

    # Created custom OADS datasets
    if bool(args.preload_all):
        traindataset = OADSImageDatasetSharedMem(oads_access=oads, item_ids=train_ids, use_crops=True, size=(n_input_channels, size[0], size[1]),
                                        class_index_mapping=class_index_mapping, transform=transform, device=device)
        valdataset = OADSImageDatasetSharedMem(oads_access=oads, item_ids=val_ids, use_crops=True, size=(n_input_channels, size[0], size[1]),
                                        class_index_mapping=class_index_mapping, transform=transform, device=device)
        testdataset = OADSImageDatasetSharedMem(oads_access=oads, item_ids=test_ids, use_crops=True, size=(n_input_channels, size[0], size[1]),
                                        class_index_mapping=class_index_mapping, transform=transform, device=device)
    else:
        traindataset = OADSImageDataset(oads_access=oads, item_ids=train_ids, use_crops=True, return_index=True,
                                        class_index_mapping=class_index_mapping, transform=transform, device=device)
        valdataset = OADSImageDataset(oads_access=oads, item_ids=val_ids, use_crops=True, return_index=True,
                                        class_index_mapping=class_index_mapping, transform=transform, device=device)
        testdataset = OADSImageDataset(oads_access=oads, item_ids=test_ids, use_crops=True, return_index=True,
                                        class_index_mapping=class_index_mapping, transform=transform, device=device)

    # Create loaders - shuffle training set, but not validation or test set
    trainloader = DataLoader(traindataset, collate_fn=collate_fn,
                                batch_size=batch_size, shuffle=True, num_workers=oads.n_processes)
    valloader = DataLoader(valdataset, collate_fn=collate_fn,
                            batch_size=batch_size, shuffle=False, num_workers=oads.n_processes)
    testloader = DataLoader(testdataset, collate_fn=collate_fn,
                            batch_size=batch_size, shuffle=False, num_workers=oads.n_processes)

    print(f"Loaded data loaders")

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
        cfg = OmegaConf.load('/home/nmuller/projects/oads_flexconv/cfg/oads_config.yaml')

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

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9)

    # Learning Rate Scheduler
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5)

    info = {
        'training_indices': train_ids,
        'testing_indices': test_ids,
        'validation_indices': val_ids,
        'optimizer': str(optimizer),
        'scheduler': str(plateau_scheduler),
        'model': str(model),
        'args': str(args),
        'device': str(device),
        'criterion': str(criterion),
        'size': size,
        'transform': transform,
        'file_formats': file_formats,
        'image_representation': args.image_representation,
        'input_dir': args.input_dir,
        'output_dir': args.output_dir
    }


    if args.test:
        print('Starting TEST')
        # print(f'Preloaded all crops:')

        # start = time.time()
        # for epoch in range(10):
        #     print(f'Epoch: {epoch}')
        #     for item in tqdm.tqdm(trainloader, total=len(trainloader)):
        #         pass
        # end = time.time()
        # print(end-start)
        fig, ax = plt.subplots(3, 20, figsize=(30, 5))
        index = 0
        for image, label in trainloader:
            for img, lbl in zip(image, label):
                if index < 20:
                    img = np.array(img)
                    ax[0][index].imshow(img[0], cmap='gray')
                    ax[0][index].set_title('R')
                    ax[1][index].imshow(img[1], cmap='gray')
                    ax[1][index].set_title('G')
                    ax[2][index].imshow(img[2], cmap='gray')
                    ax[2][index].set_title('B')
                    index += 1
                else:
                    break
        fig.tight_layout()
        result_manager.save_pdf(
                figs=[fig], filename=f'oads_example_train_stimuli_{args.image_representation}_jpeg_{args.use_jpeg}.pdf')
        

        # print('Getting mean and std')
        # means = []
        # stds = []
        # # Get dataset stats
        # for x, _ in tqdm.tqdm(trainloader):
        #     x = x.to(device)

        #     # print(x.shape, torch.mean(x, axis=(0,1)).shape)
        #     means.append(x.mean(axis=(0,2,3)))
        #     stds.append(x.std(axis=(0,2,3)))
        #     # print(x.shape, torch.std(x, axis=(0,1)).shape)
        #     # stds.append(torch.std(x, axis=(0,1)))
        #     # for img in x:
        #     #     means.append(torch.mean(img))
        #     #     stds.append(torch.std(img))

        # mean = torch.vstack(tuple((means[i]) for i in range(len(means)))).mean(axis=0, dtype=torch.float32)
        # std = torch.vstack(tuple((stds[i]) for i in range(len(stds)))).mean(axis=0, dtype=torch.float32)

        # print(mean.shape, std.shape)
        # print(mean, std)

        
        # if args.image_representation == 'COC':
        #     cmap_rg = LinearSegmentedColormap.from_list(
        #         'rg', ["r", "w", "g"], N=256)
        #     cmap_by = LinearSegmentedColormap.from_list(
        #         'by', ["b", "w", "y"], N=256)

        #     fig, ax = plt.subplots(3, 20, figsize=(30, 5))
        #     index = 0
        #     for image, label in trainloader:
        #         for img, lbl in zip(image, label):
        #             if index < 20:
        #                 coc = np.array(img)
        #                 ax[0][index].imshow(img[0], cmap='gray')
        #                 ax[0][index].set_title('I')
        #                 ax[1][index].imshow(img[1], cmap=cmap_rg)
        #                 ax[1][index].set_title('RG')
        #                 ax[2][index].imshow(img[2], cmap=cmap_by)
        #                 ax[2][index].set_title('BY')
        #                 index += 1
        #             else:
        #                 break
        #     fig.tight_layout()
        # elif args.image_representation == 'RGBEdges':
        #     cmap_rg = LinearSegmentedColormap.from_list(
        #         'rg', ["r", "w", "g"], N=256)
        #     cmap_by = LinearSegmentedColormap.from_list(
        #         'by', ["b", "w", "y"], N=256)

        #     fig, ax = plt.subplots(6, 20, figsize=(30, 10))
        #     index = 0
        #     for image, label in trainloader:
        #         for img, lbl in zip(image, label):
        #             if index < 20:
        #                 # coc = np.array(img)
        #                 # ax[0][index].imshow(img[0], cmap='Reds')
        #                 # ax[0][index].set_title('R')
        #                 # ax[1][index].imshow(img[1], cmap='Greens')
        #                 # ax[1][index].set_title('G')
        #                 # ax[2][index].imshow(img[2], cmap='Blues')
        #                 # ax[2][index].set_title('B')
        #                 # ax[3][index].imshow(img[3], cmap='gray')
        #                 # ax[3][index].set_title('I')
        #                 # ax[4][index].imshow(img[4], cmap=cmap_rg)
        #                 # ax[4][index].set_title('RG')
        #                 # ax[5][index].imshow(img[5], cmap=cmap_by)
        #                 # ax[5][index].set_title('BY')
        #                 for img_part_index, (img_part, img_part_label) in enumerate(zip(np.array(img), ['R', 'G', 'B', 'Par1', 'Par2', 'Par3', 'Mag1', 'Mag2', 'Mag3'])):
        #                     ax[img_part_index][index].imshow(img_part, cmap='gray')
        #                     ax[img_part_index][index].set_title(img_part_label)
        #                 index += 1
        #             else:
        #                 break

        #     fig.tight_layout()
        # else:
        #     images = []
        #     titles = []
        #     for index, (image, label) in enumerate(trainloader):
        #         print(index)
        #         if index < 2:
        #             for img, lbl in zip(image, label):
        #                 titles.append(index_label_mapping[int(lbl)])
        #                 images.append(transforms.ToPILImage()(img))
        #         else:
        #             break

        #     fig = plot_images(images=images, titles=titles, axis_off=False)
        ###############################
            # result_manager.save_pdf(
            #     figs=[fig], filename=f'oads_example_train_stimuli_{args.image_representation}_jpeg_{args.use_jpeg}.pdf')

            # current_time = 'Fri_Feb_17_14:13:44_2023'
            # # eval = evaluate(loader=testloader, model=model, criterion=criterion, verbose=True)
            # # result_manager.save_result(eval, filename=f'test_results_{current_time}.yml')
            # # print(oads.classes, len(oads.classes))
            # # print(class_index_mapping, len(class_index_mapping))
            # # print(index_label_mapping, len(index_label_mapping))
            # # exit(1)
            # confusion_matrix = result_manager.load_result(f'confusion_matrix_{current_time}.npz', allow_pickle=False)
            # if confusion_matrix is None:
            #     confusion_matrix = get_confusion_matrix(model=model, loader=testloader, n_classes=len(index_label_mapping), device=device)
            #     result_manager.save_result(result={'confusion_matrix': confusion_matrix}, filename=f'confusion_matrix_{current_time}.npz')

            # get_result_figures(results=None, model=model, classes=oads.classes, result_manager=result_manager, index_label_mapping=index_label_mapping, confusion_matrix=confusion_matrix, pdf_filename=f'test_visuals_{current_time}.pdf')
            # if type(confusion_matrix) == np.NpzFile:
            #     confusion_matrix.close()
    else:
        if len(results) == 0:
            result_manager.save_result(
                result=info, filename=f'fitting_description_{c_time}.yaml')

        train(model=model, trainloader=trainloader, valloader=valloader, device=device, results=results,
              loss_fn=criterion, optimizer=optimizer, n_epochs=n_epochs, result_manager=result_manager,
              testloader=testloader, plateau_lr_scheduler=plateau_scheduler, current_time=c_time)
