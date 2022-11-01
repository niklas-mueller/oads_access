import argparse, os, sys
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import numpy as np
from model import visTensor, TestModel, evaluate
from result_manager.result_manager import ResultManager
from oads_access.oads_access import OADS_Access, OADSImageDataset, plot_image_in_color_spaces
from retina_model import RetinaCortexModel
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from pytorch_utils.pytorch_utils import *

if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', help='Path to input directory.')
    parser.add_argument('--output_dir', help='Path to output directory.')
    parser.add_argument('--model_name', help='Model name to save the model under', default=f'model_{datetime.now().strftime("%d-%m-%y-%H:%M:%S")}')
    parser.add_argument('--n_epochs', help='Number of epochs for training.')
    parser.add_argument('--optimizer', help='Optimizer to use for training', default='sgd')
    parser.add_argument('--model_type', help='Model to use for training. Can be "test" or "retina_cortex"', default='retina_cortex')

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
    oads.prepare_crops()

    result_manager = ResultManager(root=args.output_dir)

    # fig = oads.plot_image_size_distribution(use_crops=True, figsize=(30, 30))
    # result_manager.save_pdf(figs=[fig], filename='image_size_distribution.pdf')
    # exit(1)

    # get train, val, test split, using crops if specific size
    # train_data, val_data, test_data = oads.get_train_val_test_split(use_crops=True, min_size=size, max_size=size)
    train_ids, val_ids, test_ids = oads.get_train_val_test_split_indices(use_crops=True)
    print(f"Loaded data with train_data.shape: {len(train_ids)}")

    
    # figs = []
    # for img in train_data:
    #     fig = plot_image_in_color_spaces(np.array(img[0]), cmap_opponent='gray')
    #     figs.append(fig)
    # result_manager.save_pdf(figs=figs, filename=f'image_in_color_spaces_{size[0]}x{size[1]}.pdf')
    # exit(1)

    input_channels = size[0] #np.array(train_data[0][0]).shape[-1]

    output_channels = len(oads.get_class_mapping())
    class_index_mapping = {key: index for index, key in enumerate(list(oads.get_class_mapping().keys()))}

    # Initialize model
    if args.model_type == 'test':
        model = TestModel(input_channels=input_channels, output_channels=output_channels, input_shape=size)
        model = torch.nn.DataParallel(model)
        model = model.to(device)
    elif args.model_type == 'retina_cortex':
        model = RetinaCortexModel(n_retina_layers=2, n_retina_in_channels=input_channels, n_retina_out_channels=2, retina_width=32,
                                input_shape=size, kernel_size=(9,9), n_vvs_layers=2, out_features=output_channels, vvs_width=32)


    batch_size = 32

    # means, stds = oads.get_dataset_stats(train_data)
    # if not means.shape == (3,):
    #     print(means.shape, stds.shape)

    # Get the custom dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(means.mean(axis=0), stds.mean(axis=0))
    ])

    # traindataset = OADSImageDataset(data=train_data, class_index_mapping=class_index_mapping, transform=transform, device=device)
    # valdataset = OADSImageDataset(data=val_data, class_index_mapping=class_index_mapping, transform=transform, device=device)
    # testdataset = OADSImageDataset(data=test_data, class_index_mapping=class_index_mapping, transform=transform, device=device)

    traindataset = OADSImageDataset(oads_access=oads, item_ids=train_ids, use_crops=True, class_index_mapping=class_index_mapping, transform=transform, device=device)
    valdataset = OADSImageDataset(oads_access=oads, item_ids=val_ids, use_crops=True, class_index_mapping=class_index_mapping, transform=transform, device=device)
    testdataset = OADSImageDataset(oads_access=oads, item_ids=test_ids, use_crops=True, class_index_mapping=class_index_mapping, transform=transform, device=device)


    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=1)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True, num_workers=1)
    testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True, num_workers=1)


    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9)

    eval_valid_every = (len(trainloader) / int(batch_size)) / 5

    train(model=model, trainloader=trainloader, valloader=valloader, device=device,
            loss_fn=criterion, optimizer=optimizer, n_epochs=int(args.n_epochs), result_manager=result_manager,
            testloader=testloader, eval_valid_every=eval_valid_every)

    # results = {}
    # eval_begin = evaluate(testloader, model, criterion=criterion)
    # results['eval_pre_training'] = eval_begin

    # print(f"Succesfully loaded everything with random test accuracy of {eval_begin['accuracy']}")
    # # exit(1)

    # epoch_times = []
    # for epoch in range(int(args.n_epochs)):  # loop over the dataset multiple times
    #     start_time_epoch = time.time()
    #     print(f"Running epoch {epoch}")
    #     running_loss = 0.0
    #     for i, data in enumerate(trainloader):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)

    #         # zero the parameter gradients
    #         optimizer.zero_grad()

    #         # forward + backward + optimize
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         # print statistics
    #         running_loss += loss.item()
    #         if i % 2000 == 1999:    # print every 2000 mini-batches
    #             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
    #             running_loss = 0.0

    #     end_time_epoch = time.time()
    #     epoch_times.append(end_time_epoch - start_time_epoch)

    # print(f'Finished Training with loss: {loss.item()}')
    # print(f'Average time per epoch for {args.n_epochs} epochs: {np.mean(epoch_times)}')

    # eval = evaluate(loader=testloader, model=model)
    # eval['training_loss'] = loss.item()

    # results['eval_post_training'] = eval
    # results['average_epoch_time'] = np.mean(epoch_times)

    # # torch.save(model.state_dict(), os.path.join(args.output_dir, f'{args.model_name}.pth'))

    # result_manager.save_result(results, filename='result_dict.yml', overwrite=True)
    # result_manager.save_model(model, filename='model.pth', overwrite=True)

    # figs = []
    # for name, module in model.module.named_modules():
    #     if not isinstance(module, nn.Sequential):
    #         if type(module) == nn.modules.conv.Conv2d or type(module) == nn.Conv2d:
    #             filter = module.weight.cpu().data.clone()
    #         else:
    #             continue
    #         fig = visTensor(filter, ch=0, allkernels=True)
    #         figs.append(fig)
    #         plt.axis('off')
    #         plt.title(f'Layer: {name}')
    #         plt.ioff()
    #         # plt.show()

    # result_manager.save_pdf(figs=figs, filename='layer_visualisation_after_training.pdf')