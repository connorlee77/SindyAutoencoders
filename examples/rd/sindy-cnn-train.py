import sys
sys.path.append("../../src")
import os
import datetime
import pandas as pd
import numpy as np
from example_reactiondiffusion import get_rd_data
import matplotlib.pyplot as plt 
from sindy_cnn import CNNAutoEncoder, SindyLibrary
from sindy_utils import library_size

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.autograd.functional import jacobian, hessian
from torch.autograd.gradcheck import zero_gradients



class RDDataset(Dataset):

	def __init__(self, data):

		B, N = data['x'].shape
		self.x = torch.Tensor(data['x']).view(B, 1, 100, 100)
		self.dx = torch.Tensor(data['dx']).view(B, 1, 100, 100)

	def __len__(self):
		return len(self.x)
	
	def __getitem__(self, idx):
		return self.x[idx], self.dx[idx]

def compute_jacobian(inputs, output):
	"""
	:param inputs: Batch X Size (e.g. Depth X Width X Height)
	:param output: Batch X Classes
	:return: jacobian: Batch X Classes X Size
	"""
	assert inputs.requires_grad

	num_classes = output.size()[1]

	jacobian = torch.zeros(num_classes, *inputs.size())
	grad_output = torch.zeros(*output.size())
	if inputs.is_cuda:
		grad_output = grad_output.cuda()
		jacobian = jacobian.cuda()

	for i in range(num_classes):
		zero_gradients(inputs)
		grad_output.zero_()
		grad_output[:, i] = 1
		output.backward(grad_output, create_graph=True)
		jacobian[i] = inputs.grad.data

	return torch.transpose(jacobian, dim0=0, dim1=1)


def evaluate(model, sindy_model, dataloader, train, device='cuda:0'):



	for i, data in enumerate(dataloader):

		x, dxdt = data[0].to(device), data[1].to(device)

		x.requires_grad = True
		z, xhat, encoded = model(x) 
		B, C, H, W = z.shape

		# dzdx = compute_jacobian(x, z)
		# dzdt = torch.matmul(dzdx.view(B, LATENT_DIM, -1), dxdt.view(B, -1, 1)).view(B, C)

		dzdx = jacobian(lambda x: model(x)[0], x, create_graph=True)
		a = torch.diagonal(dzdx, offset=0, dim1=0, dim2=4).squeeze()
		a = a.permute(3, 0, 1, 2)
		dzdt = torch.bmm(a.view(B, C, -1), dxdt.view(B, -1, 1))

		dxdz = jacobian(lambda z: model.decoder(F.upsample(z, size=encoded.shape[2:]))[1], z, create_graph=True)
		print(dxdz.shape)
		exit(0)
		# ### SINDY library
		theta_z = SindyLibrary(z.view(B, C), latent_dim=LATENT_DIM, poly_order=POLYORDER, include_sine=INCLUDE_SIN, device=device)
		zdot_hat = sindy_model(theta_z)

		print(x.shape, xhat.shape, zdot_hat.shape, dzdt.shape)
		sindy_weights = sindy_model.weight
		print(sindy_weights.shape)
		sindy_regularization = torch.linalg.norm(sindy_weights,view(1, -1), 1)

		exit(0)

if __name__ == "__main__":

	POLYORDER = 2
	INCLUDE_SIN = False
	LATENT_DIM = 2
	
	device = 'cuda:0'

	training, val, testing = get_rd_data(random=True)
	trainset = RDDataset(training)
	testset = RDDataset(testing)

	trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=0)
	testloader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=0)

	model = CNNAutoEncoder(latent_dim=LATENT_DIM)
	model.to(device)

	library_n = library_size(LATENT_DIM, POLYORDER, use_sine=INCLUDE_SIN, include_constant=True)
	sindy_model = nn.Linear(library_n, LATENT_DIM, bias=False)
	sindy_model.to(device)

	evaluate(model, sindy_model, trainloader, train=True, device=device)

