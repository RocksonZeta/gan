import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np


LR = 0.001
DOWNLOAD_MNIST = False
BATCH_SIZE = 20

X_DIM = 28*28
Z_DIM = 100

mnist_root = '/Users/ququ/data/mnist'

train_data = torchvision.datasets.MNIST(
	root = mnist_root,
	train=True,
	transform=torchvision.transforms.ToTensor()
)
train_loader = Data.DataLoader(train_data , BATCH_SIZE , shuffle=True)

def sample_z(m , n):
	return Variable(torch.FloatTensor(m,n).uniform_(-1,1))
#generator
g = nn.Sequential(
		nn.Linear(Z_DIM, 128),
		nn.ReLU(),
		nn.Linear(128,X_DIM),
		nn.Sigmoid()
	)
#discriminator
d = nn.Sequential(
		nn.Linear(X_DIM , 128),
		nn.ReLU(),
		nn.Linear(128,1),
		nn.Sigmoid()
	)

def init_param(layer):
	if type(layer) == nn.Linear :
		nn.init.normal(layer.weight.data , 0 , 0.1)
g.apply(init_param)
d.apply(init_param)

optimizer_d = torch.optim.Adam(d.parameters(), LR)
optimizer_g = torch.optim.Adam(g.parameters(), LR)


def loss_func(x,z):
	d_loss = -(torch.log(d(x)) + torch.log(1-d(g(z)))).mean()
	g_loss = - torch.log(d(g(z))).mean()

	optimizer_d.zero_grad()
	d_loss.backward()
	optimizer_d.step()
	optimizer_g.zero_grad()
	g_loss.backward()
	optimizer_g.step()
	# print("d_loss:",d_loss , g_loss)
	
def show():
	im = g(sample_z(10 , Z_DIM))
	print(im.size())
	plt.imshow(im.data.numpy().reshape(10 *28,28), cmap="gray")
	plt.pause(0.1)

plt.ion()
plt.show()
i = 0 
for epi in range(1):
	for step , (xs,ys) in enumerate(train_loader):
		if i %100 ==0 :
			show()
		i+=1
		zs = sample_z(BATCH_SIZE , Z_DIM)
		loss_func(Variable(xs.squeeze().view(BATCH_SIZE , X_DIM)) , zs)
		
plt.ioff()
