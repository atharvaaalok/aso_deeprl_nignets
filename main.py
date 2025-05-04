import numpy as np
import torch
from torch import nn
import geosimilarity as gs
from NIGnets import NIGnet

from compute_L_by_D import compute_L_by_D


# Import original airfoil
Xt = np.loadtxt('assets/naca2412.dat')

# Compute L-by-D for the original shape
L_by_D_Xt = compute_L_by_D(Xt)
print(f'L_by_D_Xt: {L_by_D_Xt}')


# Import the NIGnet model that we trained to fit the airfoil
nig_net = NIGnet(layer_count = 4, act_fn = nn.Tanh)
nig_net.load_state_dict(torch.load('assets/nignet_fit_to_naca2412.pth', weights_only = True))

# Sample points on the curve
num_pts = 250
t = torch.linspace(0, 1, num_pts).reshape(-1, 1)
Xc = nig_net(t)
Xc = Xc.detach().cpu().numpy()

# Compute L-by-D for the nignet fitted shape
L_by_D_Xc = compute_L_by_D(Xc)
print(f'L_by_D_Xc: {L_by_D_Xc}')