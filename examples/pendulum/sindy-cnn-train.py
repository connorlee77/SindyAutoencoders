import sys
sys.path.append("../../src")
import os
import datetime
import pandas as pd
import numpy as np
from example_pendulum import get_pendulum_data
from example_reactiondiffusion import get_rd_data
import matplotlib.pyplot as plt 
from sindy_cnn import CNNAutoEncoder, SindyLibrary

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd.functional import jacobian, hessian


class PendulumDataset(Dataset):

    def __init__(self, n_ics):
        data = get_pendulum_data(n_ics)
        B, H, W = data['x'].shape
        self.x = torch.Tensor(data['x']).unsqueeze(1)
        self.dx = torch.Tensor(data['dx']).unsqueeze(1)
        self.ddx = torch.Tensor(data['ddx']).unsqueeze(1)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.dx[idx], self.ddx[idx]

class RDDataset(Dataset):

    def __init__(self, train=True):
        training, val, testing = get_rd_data(random=True)

        data = training
        if not train:
            data = testing

        B, N = data['x'].shape
        self.x = torch.Tensor(data['x']).reshape(B, np.sqrt(N), np.sqrt(N))
        self.dx = torch.Tensor(data['dx']).reshape(B, np.sqrt(N), np.sqrt(N))
        print(self.x.shape)
        exit(0)
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.dx[idx], self.ddx[idx]


def evaluate(model, dataloader, train, device='cuda:0'):



    for i, data in enumerate(dataloader):
        x, dx, ddx = data[0].to(device), data[1].to(device), data[2].to(device)

        x.requires_grad = True
        z, xhat = model(x) 
        B, C = z.shape

        zs = []
        ones = []
        xs = []
        for c in range(C):
            zs.append(z[:,c])
            ones.append(torch.ones_like(z[:,c]))
            xs.append(x)

        dzdx = torch.autograd.grad(zs, xs, grad_outputs=ones, create_graph=True)

        d2zdx2 = torch.autograd.grad(zs, xs, grad_outputs=ones, create_graph=True)
        print(len(d2zdx2), d2zdx2[0].shape)
        exit(0)
        ### Compute dz/dt
        dzdt = []
        for i in range(C):
            temp = torch.sum(dzdx[i].view(B, -1) * dx.view(B, -1), dim=1)
            dzdt.append(temp)
        dzdt = torch.stack(dzdt, dim=1)


        ### SINDY library
        theta_z = SindyLibrary(z, dzdt, latent_dim=LATENT_DIM, poly_order=POLYORDER, include_sine=INCLUDE_SIN, device=device)
        print(theta_z.shape)
        exit(0)

if __name__ == "__main__":

    POLYORDER = 3
    INCLUDE_SIN = True
    LATENT_DIM = 1
    
    device = 'cuda:0'


    trainset = RDDataset(train=True)
    testset = RDDataset(train=False)

    trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=0)

    model = CNNAutoEncoder(latent_dim=LATENT_DIM)
    model.to(device)

    evaluate(model, trainloader, train=True, device=device)

