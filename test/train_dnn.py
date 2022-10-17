import argparse, os
import numpy as np
from oads_access.oads_access import OADS_Access, OADSImageDataset, TestModel
import torchvision.transforms as transforms
import torch
from torch import nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', help='Path to input directory.')
    parser.add_argument('--output_dir', help='Path to output directory.')
    parser.add_argument('--n_epochs', help='Number of epochs for training.')

    args = parser.parse_args()

    # Check if GPU is available
    if not torch.cuda.is_available():
        print("GPU not available. Exiting ....")
        exit(1)
    print("Using GPU!")

    # Setting weird stuff
    torch.multiprocessing.set_start_method('spawn')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # initialize data access
    # home = '../../data/oads/mini_oads/'
    home = args.input_dir
    oads = OADS_Access(home)

    # get train, val, test split, using crops if specific size
    size = (100, 100)
    train_data, val_data, test_data = oads.get_train_val_test_split(use_crops=True, min_size=size, max_size=size)
    input_channels = np.array(train_data[0][0]).shape[-1]

    output_channels = len(oads.get_class_mapping())
    class_index_mapping = {key: index for index, key in enumerate(list(oads.get_class_mapping().keys()))}

    # Initialize model
    model = TestModel(input_channels=input_channels, output_channels=output_channels, input_shape=size)
    model = model.to('cuda:0')

    batch_size = 10

    # Get the custom dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    traindataset = OADSImageDataset(data=train_data, class_index_mapping=class_index_mapping, transform=transform)
    valdataset = OADSImageDataset(data=val_data, class_index_mapping=class_index_mapping, transform=transform)
    testdataset = OADSImageDataset(data=test_data, class_index_mapping=class_index_mapping, transform=transform)

    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True, num_workers=2)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(int(args.n_epochs)):  # loop over the dataset multiple times
        print(f"Running epoch {epoch}")
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            print(f"Running {i}")
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # inputs = inputs.float()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            print(f"Got output!")
            loss = criterion(outputs, labels)
            print(f"Got loss!")
            loss.backward()
            print(f"Got backward!")
            optimizer.step()
            print(f"Got optimizer step!")

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print(f'Finished Training with loss: {loss.item()}')