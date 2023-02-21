import argparse
import os
import sys
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
# from model import visTensor, TestModel, evaluate
from result_manager.result_manager import ResultManager
from oads_access.oads_access import OADS_Access, OADSImageDataset, plot_image_in_color_spaces
# from retina_model import RetinaCortexModel
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50, alexnet, vgg16
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from pytorch_utils.pytorch_utils import train, evaluate, visualize_layers, collate_fn, ToJpeg, ToOpponentChannel, get_confusion_matrix, get_result_figures
import multiprocessing
from PIL import Image
from oads_access.utils import plot_images, imscatter_all

if __name__ == '__main__':

    c_time = datetime.now().strftime("%d-%m-%y-%H:%M:%S")

    # Instantiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', help='Path to input directory.',
                        default='/home/niklas/projects/data/oads')
    parser.add_argument('--output_dir', help='Path to output directory.',
                        default=f'/home/niklas/projects/oads_access/results/dnn/{c_time}')
    parser.add_argument('--model_name', help='Model name to save the model under',
                        default=f'model_{c_time}')
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
    parser.add_argument('--image_representation',
                        help='Way images are represented. Can be `RGB`, `COC` (color opponent channels)', default='RGB')
    parser.add_argument('--n_processes', help='Number of processes to use.',
                        default=multiprocessing.cpu_count()-1)
    parser.add_argument(
        '-use_jpeg', help='Whether to use JPEG Compression or not', action='store_true')
    parser.add_argument('-new_dataloader', help='Whether to use new dataloader or use the path in --dataloader_path to load existing ones. If new_dataloader is given, --dataloader_path will be use as target directory to store dataloaders', action='store_true')
    parser.add_argument('--dataloader_path', help='Path to a directory where the dataloaders can be stored from',
                        default='/home/niklas/projects/oads_access/dataloader')
    parser.add_argument('-test', help='Whether to test', action='store_true')

    args = parser.parse_args()

    # Check if GPU is available
    if not torch.cuda.is_available():
        print("GPU not available. Exiting ....")
        device = torch.device('cpu')
        exit(1)
    else:
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        print("Using GPU!")

    # Setting weird stuff
    torch.multiprocessing.set_start_method('spawn')
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # initialize data access
    # home = '../../data/oads/mini_oads/'
    size = (400, 400)

    if args.image_representation == 'RGB':
        file_formats = ['.ARW']
        convert_to_opponent_space = False
    elif args.image_representation == 'COC':
        file_formats = ['.ARW']
        # file_formats = ['.tiff']
        convert_to_opponent_space = True

    else:
        print(f"Image representation is not know. Exiting.")
        exit(1)
    print(
        f"Image representation: {args.image_representation}. File format: {file_formats}")

    exclude_classes = ['MASK', "Xtra Class 1", 'Xtra Class 2']

    home = args.input_dir
    oads = OADS_Access(home, file_formats=file_formats, use_avg_crop_size=True, n_processes=int(
        args.n_processes), min_size_crops=size, max_size_crops=size, exclude_classes=exclude_classes)

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


    n_input_channels = 3
    output_channels = len(oads.get_class_mapping())
    class_index_mapping = {}
    index_label_mapping = {}
    for index, (key, item) in enumerate(list(oads.get_class_mapping().items())):
        class_index_mapping[key] = index
        index_label_mapping[index] = item

    batch_size = 32

    # print(f"Getting dataset stats")
    # means, stds = oads.get_dataset_stats()
    # if not means.shape == (3,):
    #     print(means.shape, stds.shape)

    # Get the custom dataset and dataloader
    print(f"Getting data loaders")
    transform_list = []
    # if args.use_jpeg:
    #     transform_list.append(ToJpeg())
    if convert_to_opponent_space:
        transform_list.append(ToOpponentChannel())

    transform_list.append(transforms.ToTensor())

    transform = transforms.Compose(transform_list)
    #     # transforms.Normalize(means.mean(axis=0), stds.mean(axis=0))   # TODO fix this

    try:
        new_dataloader = False
        ids = np.load(os.path.join(args.dataloader_path, 'item_ids.npz'))
        train_ids, test_ids, val_ids = ids['train_ids'], ids['test_ids'], ids['val_ids']
        train_ids = [(x[0], int(x[1])) for x in train_ids]
        test_ids = [(x[0], int(x[1])) for x in test_ids]
        val_ids = [(x[0], int(x[1])) for x in val_ids]

    except (Exception, FileNotFoundError) as e:
        print(e)
        new_dataloader = True        

    if args.new_dataloader or new_dataloader:
        # get train, val, test split, using crops if specific size
        train_ids, val_ids, test_ids = oads.get_train_val_test_split_indices(
            use_crops=True)

        ids = {"train_ids": train_ids, "test_ids": test_ids, "val_ids": val_ids}
        np.savez_compressed(file=os.path.join(args.dataloader_path, 'item_ids.npz'), **ids)
        

    print(f"Loaded data with train_ids.shape: {len(train_ids)}")
    print(f"Loaded data with val_ids.shape: {len(val_ids)}")
    print(f"Loaded data with test_ids.shape: {len(test_ids)}")
    traindataset = OADSImageDataset(oads_access=oads, item_ids=train_ids, use_crops=True, use_jpeg=args.use_jpeg,
                                    class_index_mapping=class_index_mapping, transform=transform, device=device)
    valdataset = OADSImageDataset(oads_access=oads, item_ids=val_ids, use_crops=True, use_jpeg=args.use_jpeg,
                                    class_index_mapping=class_index_mapping, transform=transform, device=device)
    testdataset = OADSImageDataset(oads_access=oads, item_ids=test_ids, use_crops=True, use_jpeg=args.use_jpeg,
                                    class_index_mapping=class_index_mapping, transform=transform, device=device)

    trainloader = DataLoader(traindataset, collate_fn=collate_fn,
                                batch_size=batch_size, shuffle=False, num_workers=oads.n_processes)
    valloader = DataLoader(valdataset, collate_fn=collate_fn,
                            batch_size=batch_size, shuffle=False, num_workers=oads.n_processes)
    testloader = DataLoader(testdataset, collate_fn=collate_fn,
                            batch_size=batch_size, shuffle=False, num_workers=oads.n_processes)

    print(f"Loaded data loaders")

    # Initialize model
    if args.model_type == 'resnet18':
        model = resnet18()
        model.fc = torch.nn.Linear(
            in_features=512, out_features=output_channels, bias=True)
    elif args.model_type == 'resnet50':
        model = resnet50()
        model.fc = torch.nn.Linear(
            in_features=2048, out_features=output_channels, bias=True)
    elif args.model_type == 'alexnet':
        model = alexnet()
        model.classifier[6] = torch.nn.Linear(4096, output_channels, bias=True)
    elif args.model_type == 'vgg16':
        model = vgg16()
        model.classifier[-1] = torch.nn.Linear(4096,
                                               output_channels, bias=True)

    # elif args.model_type == 'retina_cortex':
    #     model = RetinaCortexModel(n_retina_layers=2, n_retina_in_channels=n_input_channels, n_retina_out_channels=2, retina_width=32,
    #                             input_shape=size, kernel_size=(9,9), n_vvs_layers=2, out_features=output_channels, vvs_width=32)

    print(f"Create model {args.model_type}")

    if args.model_path is not None:
        print(f"Loading model state {args.model_path}")
        model.load_state_dict(torch.load(args.model_path))

    # model = torch.nn.DataParallel(model)
    model = model.to(device)  # , dtype=torch.float32

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9)

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
        
        if args.image_representation == 'COC':
            cmap_rg = LinearSegmentedColormap.from_list(
                'rg', ["r", "w", "g"], N=256)
            cmap_by = LinearSegmentedColormap.from_list(
                'by', ["b", "w", "y"], N=256)

            fig, ax = plt.subplots(3, 20, figsize=(30, 10))
            index = 0
            for image, label in trainloader:
                for img, lbl in zip(image, label):
                    if index < 20:
                        coc = np.array(img)
                        ax[0][index].imshow(img[0], cmap='gray')
                        ax[0][index].set_title('I')
                        ax[1][index].imshow(img[1], cmap=cmap_rg)
                        ax[1][index].set_title('RG')
                        ax[2][index].imshow(img[2], cmap=cmap_by)
                        ax[2][index].set_title('BY')
                        index += 1
                    else:
                        break

        else:
            images = []
            titles = []
            for index, (image, label) in enumerate(trainloader):
                print(index)
                if index < 2:
                    for img, lbl in zip(image, label):
                        titles.append(index_label_mapping[int(lbl)])
                        images.append(transforms.ToPILImage()(img))
                else:
                    break

            fig = plot_images(images=images, titles=titles, axis_off=False)
        result_manager.save_pdf(
            figs=[fig], filename='oads_example_train_stimuli.pdf')

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
        result_manager.save_result(
            result=info, filename=f'fitting_description_{c_time}.yaml')

        train(model=model, trainloader=trainloader, valloader=valloader, device=device,
              loss_fn=criterion, optimizer=optimizer, n_epochs=int(args.n_epochs), result_manager=result_manager,
              testloader=testloader, plateau_lr_scheduler=plateau_scheduler, current_time=c_time)
