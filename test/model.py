from matplotlib import pyplot as plt
import numpy as np
from torch import nn as nn
import torch
from torchvision import utils

class TestModel(nn.Module):
    def __init__(self, input_channels, output_channels, input_shape, kernel_size=(3,3)) -> None:
        super(TestModel, self).__init__()

        conv_output_channels = 4
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=kernel_size, padding='same'),       # ((100,100) - (3,3)) / 1 + 1   = (98,98)
            nn.Conv2d(32, conv_output_channels, kernel_size=kernel_size, padding='same'),                    # ((98,98) - (3,3)) / 1 + 1     = (96,96)
            nn.Flatten(),
            nn.Linear(in_features=input_shape[0]*input_shape[1]*conv_output_channels, out_features=output_channels),
        )


    def forward(self, x):
        return self.layers(x)

def evaluate(loader, model:nn.DataParallel, criterion):
    num_correct = 0
    num_samples = 0
    losses = {}
    model.eval()
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device=model.src_device_obj)
            y = y.to(device=model.src_device_obj)
            
            scores = model(x)
            loss = criterion(scores, y)
            losses[i] = loss.cpu().tolist()
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        accuracy = float(num_correct)/float(num_samples)
        print(f'Got {num_correct} / {num_samples} with accuracy {accuracy*100:.2f}%') 
    
    model.train()

    return {'accuracy': accuracy, 'batch_losses': losses}

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    fig = plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    return fig