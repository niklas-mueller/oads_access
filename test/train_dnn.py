import argparse, os, sys
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import numpy as np
# from model import visTensor, TestModel, evaluate
from result_manager.result_manager import ResultManager
from oads_access.oads_access import OADS_Access, OADSImageDataset, plot_image_in_color_spaces
from retina_model import RetinaCortexModel
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50, alexnet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from pytorch_utils.pytorch_utils import train, evaluate, visualize_layers

if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', help='Path to input directory.')
    parser.add_argument('--output_dir', help='Path to output directory.')
    parser.add_argument('--model_name', help='Model name to save the model under', default=f'model_{datetime.now().strftime("%d-%m-%y-%H:%M:%S")}')
    parser.add_argument('--n_epochs', help='Number of epochs for training.')
    parser.add_argument('--force_recrop', help='Whether to recompute the crops from the images.', default=False)
    parser.add_argument('--get_visuals', help='Whether to run some visual instead of training.', default=False)
    parser.add_argument('--optimizer', help='Optimizer to use for training', default='sgd')
    parser.add_argument('--model_path', help='Path to model to continue training on.', default=None)
    parser.add_argument('--model_type', help='Model to use for training. Can be "test" or "retina_cortex"', default='retina_cortex')
    parser.add_argument('--continue_model_path', help='Path to load model from to continue training.', default=None)

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
    home = args.input_dir
    size = (200, 200)
    oads = OADS_Access(home, min_size_crops=size, max_size_crops=size)

    # Compute crops if necessary
    if args.force_recrop:
        print(f"Recomputing crops.")
        oads.prepare_crops()


    result_manager = ResultManager(root=args.output_dir)

    if args.get_visuals:
        fig = oads.plot_image_size_distribution(use_crops=True, figsize=(30, 30))
        result_manager.save_pdf(figs=[fig], filename='image_size_distribution.pdf')

    # get train, val, test split, using crops if specific size
    train_ids, val_ids, test_ids = oads.get_train_val_test_split_indices(use_crops=True)
    print(f"Loaded data with train_ids.shape: {len(train_ids)}")
    print(f"Loaded data with val_ids.shape: {len(val_ids)}")
    print(f"Loaded data with test_ids.shape: {len(test_ids)}")

    # fig = oads.plot_image_size_distribution(use_crops=True, figsize=(20, 20))
    # result_manager.save_pdf(figs=[fig], filename='image_size_distribution.pdf')
    # exit(1)
    
    if args.get_visuals:
        print(f"Getting visuals: plot_image_in_color_spaces")
        figs = []
        results = oads.apply_per_crop(lambda x: plot_image_in_color_spaces(np.array(x[0]), cmap_opponent='gray'), max_number_crops=50)
        for _, images in results.items():
            for _, fig in images.items():
                figs.append(fig)

        # for img in train_data:
        #     fig = plot_image_in_color_spaces(np.array(img[0]), cmap_opponent='gray')
        #     figs.append(fig)
        result_manager.save_pdf(figs=figs, filename=f'image_in_color_spaces_{size[0]}x{size[1]}.pdf')


    input_channels = size[0] #np.array(train_data[0][0]).shape[-1]
    output_channels = len(oads.get_class_mapping())
    class_index_mapping = {key: index for index, key in enumerate(list(oads.get_class_mapping().keys()))}

    batch_size = 32

    print(f"Getting dataset stats")
    means, stds = oads.get_dataset_stats()
    if not means.shape == (3,):
        print(means.shape, stds.shape)

    # Get the custom dataset and dataloader
    print(f"Getting data loaders")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means.mean(axis=0), stds.mean(axis=0))
    ])

    traindataset = OADSImageDataset(oads_access=oads, item_ids=train_ids, use_crops=True, class_index_mapping=class_index_mapping, transform=transform, device=device)
    valdataset = OADSImageDataset(oads_access=oads, item_ids=val_ids, use_crops=True, class_index_mapping=class_index_mapping, transform=transform, device=device)
    testdataset = OADSImageDataset(oads_access=oads, item_ids=test_ids, use_crops=True, class_index_mapping=class_index_mapping, transform=transform, device=device)


    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=16)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True, num_workers=16)
    testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True, num_workers=16)

    print(f"Loaded data loaders")

    # Initialize model
    if args.model_type == 'resnet18':
        model = resnet18()
        model.fc = torch.nn.Linear(in_features=512, out_features=output_channels, bias=True)
    elif args.model_type == 'resnet50':
        model = resnet50()
        model.fc = torch.nn.Linear(in_features=2048, out_features=output_channels, bias=True)
    elif args.model_type == 'alexnet':
        model = alexnet()
        model.classifier[6] = torch.nn.Linear(4096, output_channels)
    elif args.model_type == 'retina_cortex':
        model = RetinaCortexModel(n_retina_layers=2, n_retina_in_channels=input_channels, n_retina_out_channels=2, retina_width=32,
                                input_shape=size, kernel_size=(9,9), n_vvs_layers=2, out_features=output_channels, vvs_width=32)

    print(f"Create model {args.model_type}")

    if args.model_path is not None:
        print(f"Loading model state {args.model_path}")
        model.load_state_dict(torch.load(args.model_path))

    model = torch.nn.DataParallel(model)
    model = model.to(device)



    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9)

    eval_valid_every = (len(trainloader) / int(batch_size)) / 5

    train(model=model, trainloader=trainloader, valloader=valloader, device=device,
            loss_fn=criterion, optimizer=optimizer, n_epochs=int(args.n_epochs), result_manager=result_manager,
            testloader=testloader, eval_valid_every=eval_valid_every)
