''' this script works with CPU and GPUs
'''
from __future__ import print_function

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import random
import numpy as np
import scipy.io

import argparse
import os
import gflags
import sys


# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


from utils import Logger, AverageMeter
import models
import baro_topo

# hyper parameters for training
parser = argparse.ArgumentParser(description='model configuration')
# data loading parameters
parser.add_argument('--pred_length', default=200599, type=int, metavar='T',
                    help='sequence length for training samples')
parser.add_argument('--input_length', default=100, type=int, metavar='L',
                    help='model input state size')
parser.add_argument('--nskip', default=1, type=int, metavar='nk',
                    help='time step skip in the loaded raw data, dt=1e-2')
parser.add_argument('--npred', default=500, type=int, metavar='Np',
                    help='number of iterations to measure in the loss func.')
parser.add_argument('--kmax', default=10, type=int, metavar='km',
                    help='total number of spectral modes in the red. model')
parser.add_argument('--kred', default=2, type=int, metavar='krr',
                    help='total number of resolved modes in the red. model')
# (train-batch * iters = train_length - input_length - npred+1)


# model parameters
parser.add_argument('--epoch', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--nhid', default=100, type=int, metavar='nh',
                    help='hidden layer size in the network cell for variance')
parser.add_argument('--nhidm', default=20, type=int, metavar='nhm',
                    help='hidden layer size in the network cell for mean')
parser.add_argument('--nloss', default=100, type=int, metavar='nv',
                    help='number of steps to measure in the loss function')
# optimization parameters
parser.add_argument('--loss-type', '--lt', default='kld', type=str, metavar='LT',
                    help='choices of loss functions (state,flux,comb, kld,mixed)')

# checkpoints/data setting
parser.add_argument('-c', '--checkpoint', default='checkpoint/train_flow_K10sk1sU20_condGau', type=str, metavar='C_PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--data-file', default='data/baro_K10sk1sU20dk1dU1', type=str, metavar='DATA_PATH',
                    help='path to train data set pert forcing')
parser.add_argument('--resume', default=False, type=bool, metavar='R_PATH',
                    help='path to latest checkpoint (default: none)')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()
cfg = {k: v for k, v in args._get_kwargs()}

if not os.path.isdir(args.checkpoint):
    os.makedirs(args.checkpoint)
        
# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)

fname_load = 'flowred2_euler_adamlr01_ls2stg0hs{}hsm{}nl{}_seq{}nsk{}np10_epoch{}_'.format(args.nhid,args.nhidm, args.nloss,
            args.input_length, args.nskip, args.epoch) + args.loss_type
fname = 'flow{}_euler_lr01_ls2stg0hs{}hsm{}nl{}_seq{}nsk{}np10_epoch{}_'.format(args.npred, args.nhid,args.nhidm, args.nloss,
            args.input_length, args.nskip, args.epoch) + args.loss_type

def main(pretrained = False, valid = False):
    # models for unresolved processes
    model_m = models.LSTMresi(input_size = 2+3*args.kred, hidden_size = args.nhidm, output_size = 1, 
                                nlayers = 2, nstages = 0).double()
    model_v = models.LSTMresi(input_size = 1+8*args.kred, hidden_size = args.nhid, output_size = 4*args.kred, 
                                nlayers = 2, nstages = 0).double()
    # load model on GPU
    dev1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dev2 = torch.device("cuda:1" if torch.cuda.device_count()>1 else "cuda:0")
    device = (dev1, dev2)
    print('This code is run by {}: {} GPU(s)'.format(dev1, torch.cuda.device_count()))
    
    
    model_path1 = torch.load(os.path.join(cfg['checkpoint'], 'modelm_'+fname_load), map_location=dev1)
    model_m.load_state_dict(model_path1['model_state_dict'])
    model_path2 = torch.load(os.path.join(cfg['checkpoint'], 'modelv_'+fname_load), map_location=dev2)
    model_v.load_state_dict(model_path2['model_state_dict'])  
    model_m.to(dev1)
    model_v.to(dev2)
    model = (model_m, model_v)
        
    print('    Total vari model params.: {}'.format(sum(p.numel() for p in model_v.parameters())))
    
    
    # evaluating model in prediction data set
    # load evaluation dataset
    data_load = scipy.io.loadmat(args.data_file)
    params = data_load.get('params')[0,0]
    tt    = np.transpose(data_load.get('TT'), (1,0))
    dt    = data_load.get('Dt')[0,0]
    Us    = np.transpose(data_load.get('Uout'), (1,0))
    dotUs = np.transpose(data_load.get('dUout'), (1,0))
    noise = np.transpose(data_load.get('noise'), (1,0))
    unres = np.transpose(data_load.get('unres'), (1,0))
    mu    = np.transpose(data_load.get('umout'), (1,0))
    Ru    = np.transpose(data_load.get('Ruout'), (1,0))
    Cu    = np.transpose(data_load.get('Cuout'), (1,0))
    Ff    = np.transpose(data_load.get('dfout'), (1,0))
    Gf    = np.transpose(data_load.get('dgout'), (1,0))
    
    nb = 10000
    npred = args.npred 
    tt = tt[nb:args.pred_length*args.nskip+nb+1:args.nskip]
    args.dt = dt
    Ut   = np.empty(tt.shape[0])
    dotU = np.empty(tt.shape[0])
    Fu   = np.empty(tt.shape[0])
    dW   = np.empty(tt.shape[0])
    Ures = np.empty(tt.shape[0])
    Ut[:]   =    Us[nb:args.pred_length*args.nskip+nb+1  :args.nskip,0]
    dotU[:] = dotUs[nb:args.pred_length*args.nskip+nb+1  :args.nskip,0]
    Fu[:]   =    Ff[nb-1:args.pred_length*args.nskip+nb  :args.nskip,0]
    dW[:]   = noise[nb+1:args.pred_length*args.nskip+nb+2:args.nskip,0]
    Ures[:] = unres[nb-1:args.pred_length*args.nskip+nb  :args.nskip,0]
    uk = np.empty((tt.shape[0], 4*args.kmax))
    rk = np.empty((tt.shape[0], 2*args.kmax))
    ck = np.empty((tt.shape[0], 2*args.kmax))
    Gk = np.empty((tt.shape[0], 4*args.kmax))
    uk[:,:args.kmax]              = mu[nb:args.pred_length*args.nskip+nb+1:args.nskip,:args.kmax].real
    uk[:,args.kmax:2*args.kmax]   = mu[nb:args.pred_length*args.nskip+nb+1:args.nskip,:args.kmax].imag
    uk[:,2*args.kmax:3*args.kmax] = mu[nb:args.pred_length*args.nskip+nb+1:args.nskip,2*args.kmax:3*args.kmax].real
    uk[:,3*args.kmax:]            = mu[nb:args.pred_length*args.nskip+nb+1:args.nskip,2*args.kmax:3*args.kmax].imag
    Gk[:,:args.kmax]              = Gf[nb-1:args.pred_length*args.nskip+nb:args.nskip,:args.kmax].real
    Gk[:,args.kmax:2*args.kmax]   = Gf[nb-1:args.pred_length*args.nskip+nb:args.nskip,:args.kmax].imag
    Gk[:,2*args.kmax:3*args.kmax] = Gf[nb-1:args.pred_length*args.nskip+nb:args.nskip,2*args.kmax:3*args.kmax].real
    Gk[:,3*args.kmax:4*args.kmax] = Gf[nb-1:args.pred_length*args.nskip+nb:args.nskip,2*args.kmax:3*args.kmax].imag
    for ii in range(args.kmax):
        rk[:,ii]           = Ru[nb:args.pred_length*args.nskip+nb+1:args.nskip, ii            ].real
        rk[:,ii+args.kmax] = Ru[nb:args.pred_length*args.nskip+nb+1:args.nskip, ii+2*args.kmax].real
        ck[:,ii]           = Cu[nb:args.pred_length*args.nskip+nb+1:args.nskip, ii].real
        ck[:,ii+args.kmax] = Cu[nb:args.pred_length*args.nskip+nb+1:args.nskip, ii].imag
    
    
    nskip = 20
    Nsamp  = (args.pred_length-args.input_length - args.npred+1) // nskip
    traj_set   = torch.zeros(args.input_length + args.npred, Nsamp, 5+12*args.kred, dtype=torch.double)
    ind1 = np.concatenate([range(0,args.kred),range(args.kmax,args.kmax+args.kred)])
    ind2 = np.concatenate([range(2*args.kmax,2*args.kmax+args.kred),range(3*args.kmax,3*args.kmax+args.kred)])
    for l in range(Nsamp):
        traj_set[:,l, 0]                            = torch.from_numpy(  Ut[l*nskip:l*nskip+args.input_length + args.npred])
        traj_set[:,l, 1]                            = torch.from_numpy(dotU[l*nskip:l*nskip+args.input_length + args.npred])
        traj_set[:,l, 2:2+2*args.kred]              = torch.from_numpy(  uk[l*nskip:l*nskip+args.input_length + args.npred,ind1])
        traj_set[:,l, 2+2*args.kred:2+4*args.kred]  = torch.from_numpy(  uk[l*nskip:l*nskip+args.input_length + args.npred,ind2])
        traj_set[:,l, 2+4*args.kred:2+6*args.kred]  = torch.from_numpy(  rk[l*nskip:l*nskip+args.input_length + args.npred,ind1])
        traj_set[:,l, 2+6*args.kred:2+8*args.kred]  = torch.from_numpy(  ck[l*nskip:l*nskip+args.input_length + args.npred,ind1])
        traj_set[:,l, 2+8*args.kred]                = torch.from_numpy(  Fu[l*nskip:l*nskip+args.input_length + args.npred])
        traj_set[:,l, 3+8*args.kred:3+10*args.kred] = torch.from_numpy(  Gk[l*nskip:l*nskip+args.input_length + args.npred,ind1])
        traj_set[:,l, 3+10*args.kred:3+12*args.kred]= torch.from_numpy(  Gk[l*nskip:l*nskip+args.input_length + args.npred,ind2])
        traj_set[:,l, 3+12*args.kred]               = torch.from_numpy(Ures[l*nskip:l*nskip+args.input_length + args.npred])
        traj_set[:,l, 4+12*args.kred]               = torch.from_numpy(  dW[l*nskip:l*nskip+args.input_length + args.npred])
    del(data_load,Ut,dotU,uk,rk,ck,Fu,Gk, dW,Ures)
    
    valid_pred, valid_err = prediction(traj_set, npred, model, params, device)

    
    datapath = os.path.join(cfg['checkpoint'], 'leading1_' + fname)
    np.savez(datapath, tt = tt, pred = valid_pred[:,:,:,0], gt = valid_pred[:,:,:,1], valid_err = valid_err)

def prediction(input_set, npred, model, params, device):
    dev1, dev2 = device
    model_m, model_v = model
    with torch.no_grad():
        model_m.eval()
        model_v.eval()
    baro_cond = baro_topo.CondGau(dt=args.dt, params=params, kred=args.kred, device=dev1)
    baro_flow = baro_topo.BaroTopo(dt=args.dt, params=params, kred=args.kred, device=dev1)
 
    kmax = params['kmax'][0,0]
    kred = args.kred
    Nsamp = input_set.shape[1]
    valid_pred = np.zeros((10, Nsamp,3+12*kred, 2))
    valid_err  = np.zeros((npred,3+12*kred))

    ind1 = np.concatenate([range(1),range(2,2+2*kred),range(2+4*kred,2+5*kred),range(3+12*kred,4+12*kred)])
    ind2 = np.concatenate([range(1),range(2+4*kred,2+8*kred),range(3+8*kred,3+12*kred)])
    indt = np.concatenate([range(1),range(2,4+12*kred)])

    ic=0
    with torch.no_grad():
        istate   = torch.empty(args.input_length, Nsamp, 5+12*kred, dtype=torch.double, device=dev1)
        istate_m = torch.empty(args.input_length, Nsamp, 2+3*kred, dtype=torch.double, device=dev1)
        istate_v = torch.empty(args.input_length, Nsamp, 1+8*kred, dtype=torch.double, device=dev1)
    
        istate_m[:,:,:] = input_set[:args.input_length,:,ind1].clone()
        istate_v[:,:,:] = input_set[:args.input_length,:,ind2].clone()
        istate[:,:,:]   = input_set[:args.input_length,:,:].clone()
        # target   = input_set[-1, :,2:]

        hidden_m, hidden_v = (), ()
        for istep in range(npred):
            ############################################################
            # run model in one forward iteration
            Ur_out, hidden_m = model_m(istate_m, hidden_m, device=dev1)
            Ur_out = torch.squeeze(Ur_out)
            Gk_out, hidden_v = model_v(istate_v, hidden_v, device=dev2)

            istate[:,:,3+8*args.kred:3+12*args.kred] = Gk_out.to(dev1)
            istate[:,:,3+12*args.kred] = Ur_out.to(dev1)
            istate1, incre = baro_cond.update_condGau_step(istate,istate, args.dt)
            U1, increU = baro_flow.update_flow_step(istate,istate, args.dt)
            (inc_vk, inc_Tk, inc_rk, inc_ck) = incre
            Fu_out = istate1[:,:,2+8*args.kred]
            Gk_p = Gk_out
            Ur_p = Ur_out

            dW = input_set[istep:args.input_length+istep,:,4+12*args.kred].to(dev1)
            U_out  = istate[:,:,0] + args.dt * increU + dW
            dotU_out = increU + dW/args.dt
            vk_out = istate[:,:,2       :2+2*kred] + args.dt * inc_vk
            Tk_out = istate[:,:,2+2*kred:2+4*kred] + args.dt * inc_Tk
            rk_out = istate[:,:,2+4*kred:2+6*kred] + args.dt * inc_rk
            ck_out = istate[:,:,2+6*kred:2+8*kred] + args.dt * inc_ck
            ############################################################
            # run model in 4-stage runge-kutta
            '''
            Ur1, hidden_m = model_m(istate_m, hidden_m, device=dev1)
            Ur1 = torch.squeeze(Ur1)
            Gk1, hidden_v = model_v(istate_v, hidden_v, device=dev2)
            istate[:,:,3+8*args.kred:3+12*args.kred] = Gk1.to(dev1)
            istate[:,:,3+12*args.kred] = Ur1.to(dev1)
            # rk4 -- step 1
            istate1, incre1 = baro_cond.update_condGau_step(istate,istate, args.dt/2)
            U1, increU1 = baro_flow.update_flow_step(istate,istate, args.dt/2)
            istate1[:,:,0] = U1
            (inc_vk1, inc_Tk1, inc_rk1, inc_ck1) = incre1
            Fu1 = istate1[:,:,2+8*args.kred]
            istate_m1 = istate1[:,:,ind1].clone().to(dev1)
            istate_v1 = istate1[:,:,ind2].clone().to(dev2)

            Ur2, hidden_m = model_m(istate_m1, hidden_m, device=dev1)
            Ur2 = torch.squeeze(Ur2)
            Gk2, hidden_v = model_v(istate_v1, hidden_v, device=dev2)
            istate1[:,:,3+8*args.kred:3+12*args.kred] = Gk2.to(dev1)
            istate1[:,:,3+12*args.kred] = Ur2.to(dev1)
            # rk4 -- step 2
            istate2, incre2 = baro_cond.update_condGau_step(istate1,istate, args.dt/2)
            U2, increU2 = baro_flow.update_flow_step(istate1,istate, args.dt/2)
            istate2[:,:,0] = U2
            (inc_vk2, inc_Tk2, inc_rk2, inc_ck2) = incre2
            Fu2 = istate2[:,:,2+8*args.kred]
            istate_m2 = istate2[:,:,ind1].clone().to(dev1)
            istate_v2 = istate2[:,:,ind2].clone().to(dev2)

            Ur3, hidden_m = model_m(istate_m2, hidden_m, device=dev1)
            Ur3 = torch.squeeze(Ur3)
            Gk3, hidden_v = model_v(istate_v2, hidden_v, device=dev2)
            istate2[:,:,3+8*args.kred:3+12*args.kred] = Gk3.to(dev1)
            istate2[:,:,3+12*args.kred] = Ur3.to(dev1)
            # rk4 -- step 3
            istate3, incre3 = baro_cond.update_condGau_step(istate2,istate, args.dt)
            U3, increU3 = baro_flow.update_flow_step(istate2,istate, args.dt)
            istate3[:,:,0] = U3
            (inc_vk3, inc_Tk3, inc_rk3, inc_ck3) = incre3
            Fu3 = istate3[:,:,2+8*args.kred]
            istate_m3 = istate3[:,:,ind1].clone().to(dev1)
            istate_v3 = istate3[:,:,ind2].clone().to(dev2)

            Ur4, hidden_m = model_m(istate_m3, hidden_m, device=dev1)
            Ur4 = torch.squeeze(Ur4)
            Gk4, hidden_v = model_v(istate_v3, hidden_v, device=dev2)
            istate3[:,:,3+8*args.kred:3+12*args.kred] = Gk4.to(dev1)
            istate3[:,:,3+12*args.kred] = Ur4.to(dev1)
            # rk4 -- step 4
            istate4, incre4 = baro_cond.update_condGau_step(istate3,istate, args.dt)
            U4, increU4 = baro_flow.update_flow_step(istate3,istate, args.dt)
            istate4[:,:,0] = U4
            (inc_vk4, inc_Tk4, inc_rk4, inc_ck4) = incre4
            Fu4 = istate4[:,:,2+8*args.kred]
            
            Fu_out = Fu1 #(Fu1+2*Fu2+2*Fu3+Fu4)/6
            Gk_out = Gk1 #(Gk1+2*Gk2+2*Gk3+Gk4)/6
            Ur_out = Ur1 #(Ur1+2*Ur2+2*Ur3+Ur4)/6
            Gk_p = (Gk1+2*Gk2+2*Gk3+Gk4)/6
            Ur_p = (Ur1+2*Ur2+2*Ur3+Ur4)/6
            
            dW = input_set[istep:args.input_length+istep,:,4+12*args.kred].to(dev1)
            U_out  = istate[:,:,0] + args.dt * (increU1 + 2*increU2 + 2*increU3 + increU4) / 6 + dW
            dotU_out = (increU1 + 2*increU2 + 2*increU3 + increU4) / 6 + dW/args.dt
            vk_out = istate[:,:,2:2+2*kred] + args.dt * (inc_vk1 + 2*inc_vk2 + 2*inc_vk3 + inc_vk4) / 6
            Tk_out = istate[:,:,2+2*kred:2+4*kred] + args.dt * (inc_Tk1 + 2*inc_Tk2 + 2*inc_Tk3 + inc_Tk4) / 6
            rk_out = istate[:,:,2+4*kred:2+6*kred] + args.dt * (inc_rk1 + 2*inc_rk2 + 2*inc_rk3 + inc_rk4) / 6
            ck_out = istate[:,:,2+6*kred:2+8*kred] + args.dt * (inc_ck1 + 2*inc_ck2 + 2*inc_ck3 + inc_ck4) / 6
            '''
            ######################################################################################################
            
            
            istate_m[:-1,:,:] = istate_m[1:,:,:].clone()
            istate_v[:-1,:,:] = istate_v[1:,:,:].clone()
            istate[:-1,:,:]   = istate[1:,:,:].clone()
            
            istate[-1,:,2+8*kred] = input_set[args.input_length+istep,:,2+8*kred]
            istate[-1,:,4+12*kred] = input_set[args.input_length+istep,:,4+12*kred]
            
            istate_m[-1,:,0]                 = U_out[-1]
            istate_m[-1,:,1:1+2*kred]        = vk_out[-1]
            istate_m[-1,:,1+2*kred:1+3*kred] = rk_out[-1,:,:kred]
            istate_m[-1,:,1+3*kred]          = Ur_p[-1]
            istate_v[-1,:,0]                 = U_out[-1].to(dev2)
            istate_v[-1,:,1:1+2*kred]        = rk_out[-1].to(dev2)
            istate_v[-1,:,1+2*kred:1+4*kred] = ck_out[-1].to(dev2)
            istate_v[-1,:,1+4*kred:1+8*kred] = Gk_p[-1]
            istate[-1,:,0]                  = U_out[-1]
            istate[-1,:,1]                  = dotU_out[-1]
            istate[-1,:,2:2+2*kred]         = vk_out[-1]
            istate[-1,:,2+2*kred:2+4*kred]  = Tk_out[-1]
            istate[-1,:,2+4*kred:2+6*kred]  = rk_out[-1]
            istate[-1,:,2+6*kred:2+8*kred]  = ck_out[-1]
            # istate[-1,:,2+8*kred]           = Fu_out[-1]
            istate[-1,:,3+8*kred:3+12*kred] = Gk_out[-1].to(dev1)
            istate[-1,:,3+12*kred]          = Ur_out[-1]

            
            targ = input_set[args.input_length+istep,:,indt]
            valid_err[istep,:2*kred]            = np.square(vk_out.data.cpu().numpy()[-1] - targ.data.cpu().numpy()[:,1       :1+2*kred]).mean(0)
            valid_err[istep,2*kred:4*kred]      = np.square(Tk_out.data.cpu().numpy()[-1] - targ.data.cpu().numpy()[:,1+2*kred:1+4*kred]).mean(0)
            valid_err[istep,4*kred:6*kred]      = np.square(rk_out.data.cpu().numpy()[-1] - targ.data.cpu().numpy()[:,1+4*kred:1+6*kred]).mean(0)
            valid_err[istep,6*kred:8*kred]      = np.square(ck_out.data.cpu().numpy()[-1] - targ.data.cpu().numpy()[:,1+6*kred:1+8*kred]).mean(0)
            valid_err[istep,8*kred]             = np.square(Fu_out.data.cpu().numpy()[-1] - targ.data.cpu().numpy()[:,1+8*kred]).mean(0)
            valid_err[istep,1+8*kred:1+12*kred] = np.square(Gk_out.data.cpu().numpy()[-1] - targ.data.cpu().numpy()[:,2+8*kred:2+12*kred]).mean(0)
            valid_err[istep,1+12*kred]          = np.square(Ur_out.data.cpu().numpy()[-1] - targ.data.cpu().numpy()[:,2+12*kred]).mean(0)
            valid_err[istep,2+12*kred]          = np.square( U_out.data.cpu().numpy()[-1] - targ.data.cpu().numpy()[:,0]).mean(0)
            print('step {}: err_U = {:.6f} err_Ur = {:.6f} err_u = {:.6f} err_Fu = {:.6f} error_R = {:.6f} error_Gk = {:.6f}'.format(istep, 
                  valid_err[istep,2+12*kred],valid_err[istep,1+12*kred], valid_err[istep,:2*kred].sum(), 
                  valid_err[istep,8*kred], valid_err[istep,4*kred:6*kred].sum(), valid_err[istep,1+8*kred:1+12*kred].sum()) )
                
            if (istep+1) in [20,50,80,100,150,200,250,300,400,500]:
                predv  = vk_out.data.cpu().numpy()[-1]
                predT  = Tk_out.data.cpu().numpy()[-1]
                predrk  = rk_out.data.cpu().numpy()[-1]
                predck  = ck_out.data.cpu().numpy()[-1]
                predFu = Fu_out.data.cpu().numpy()[-1]
                predGk = Gk_out.data.cpu().numpy()[-1]
                predU = U_out.data.cpu().numpy()[-1]
                predUr = Ur_out.data.cpu().numpy()[-1]
                targ  = targ.data.cpu().numpy()
                valid_pred[ic,:,:2*kred, 0] = predv
                valid_pred[ic,:,2*kred:4*kred, 0] = predT
                valid_pred[ic,:,4*kred:6*kred, 0] = predrk
                valid_pred[ic,:,6*kred:8*kred, 0] = predck
                valid_pred[ic,:,8*kred, 0] = predFu
                valid_pred[ic,:,8*kred+1:12*kred+1, 0] = predGk
                valid_pred[ic,:,12*kred+1, 0] = predUr
                valid_pred[ic,:,12*kred+2, 0] = predU
                valid_pred[ic,:,:, 1] = targ
                ic = ic+1
            


        
    return valid_pred, valid_err

if __name__ == '__main__':
    gflags.FLAGS(sys.argv)
    gflags.DEFINE_boolean('pretrained', False, 'Use pretrained model')
    gflags.DEFINE_boolean('eval', True, 'Run tests with the network')
    
    main(pretrained = gflags.FLAGS.pretrained, valid = gflags.FLAGS.eval)
