''' this script works with CPU or single GPU
'''
from __future__ import print_function

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
import dyad

# hyper parameters for training
parser = argparse.ArgumentParser(description='model configuration')
# data loading parameters
parser.add_argument('--pred_length', default=10599, type=int, metavar='T',
                    help='sequence length for training samples')
parser.add_argument('--input_length', default=100, type=int, metavar='L',
                    help='model input state size')
parser.add_argument('--nskip', default=1, type=int, metavar='nk',
                    help='time step skip in the loaded raw data, dt=1e-2')
parser.add_argument('--npred', default=1000, type=int, metavar='Np',
                    help='number of iterations to measure in the loss func.')
# (train-batch * iters = train_length - input_length - npred+1)


# model parameters
parser.add_argument('--epoch', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--nhid', default=20, type=int, metavar='nh',
                    help='hidden layer size in the network cell for variance')
parser.add_argument('--nhidm', default=20, type=int, metavar='nhm',
                    help='hidden layer size in the network cell for mean')
parser.add_argument('--nloss', default=100, type=int, metavar='nv',
                    help='number of steps to measure in the loss function')
# optimization parameters
parser.add_argument('--loss-type', '--lt', default='mixed', type=str, metavar='LT',
                    help='choices of loss functions (state,kld,mixed)')

# checkpoints/data setting
parser.add_argument('-c', '--checkpoint', default='checkpoint/train_dyad_su05sv2_condGau', type=str, metavar='C_PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--data-file', default='data/dyad_su05sv2', type=str, metavar='DATA_PATH',
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

fname_load = 'dyad_adamlr01_hs{}hsm{}nl{}_seq{}nsk{}np10_epoch{}_'.format(args.nhid,args.nhidm, args.nloss,
            args.input_length, args.nskip, args.epoch) + args.loss_type
fname = '{}_dyad_hs{}hsm{}nl{}_seq{}nsk{}np10_epoch{}_'.format(args.npred, args.nhid,args.nhidm, args.nloss,
            args.input_length, args.nskip,  args.epoch) + args.loss_type

def main(pretrained = False, valid = False):
    # models for unresolved processes
    model_m = models.LSTMresi(input_size = 3, hidden_size = args.nhidm, output_size = 1, 
                                nlayers = 1, nstages = 0).double()
    model_v = models.LSTMresi(input_size = 3, hidden_size = args.nhid, output_size = 1, 
                                nlayers = 1, nstages = 0).double()
    # load model on GPU
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('This code is run by {}: {} GPU(s)'.format(dev, torch.cuda.device_count()))
    
    
    model_path1 = torch.load(os.path.join(cfg['checkpoint'], 'modelm_'+fname_load), map_location=dev)
    model_m.load_state_dict(model_path1['model_state_dict'])
    model_path2 = torch.load(os.path.join(cfg['checkpoint'], 'modelv_'+fname_load), map_location=dev)
    model_v.load_state_dict(model_path2['model_state_dict'])  
    model_m.to(dev)
    model_v.to(dev)
    model = (model_m, model_v)
        
    print('    Total vari model params.: {}'.format(sum(p.numel() for p in model_v.parameters())))
    
    
    # evaluating model in prediction data set
    # load evaluation dataset
    data_load = scipy.io.loadmat(args.data_file)
    params = data_load.get('params')[0,0]
    tt = np.transpose(data_load.get('tout'), (1,0))
    dt = data_load.get('dt')[0,0]
    Us = np.transpose(data_load.get('u_truth'), (1,0))
    # Vs = np.transpose(data_load.get('v_truth'), (1,0))
    dotUs = np.transpose(data_load.get('dotu'), (1,0))
    noise = np.transpose(data_load.get('noise'), (1,0))
    unres = np.transpose(data_load.get('unres'), (1,0))
    mu = np.transpose(data_load.get('gamma_mean_trace'), (1,0))
    Ru = np.transpose(data_load.get('gamma_cov_trace'),  (1,0))
    Ff = np.transpose(data_load.get('dfout'), (1,0))
    Gf = np.transpose(data_load.get('dgout'), (1,0))
    
    
    nb = 10000
    tt = tt[nb:args.pred_length*args.nskip+nb+1:args.nskip]
    args.dt = dt
    Ut   = np.empty(tt.shape[0])
    dotU = np.empty(tt.shape[0])
    Fu   = np.empty(tt.shape[0])
    dW   = np.empty(tt.shape[0])
    Ures = np.empty(tt.shape[0])
    Ut[:]   =    Us[nb  :args.pred_length*args.nskip+nb+1:args.nskip,0]
    dotU[:] = dotUs[nb  :args.pred_length*args.nskip+nb+1:args.nskip,0]
    Fu[:]   =    Ff[nb  :args.pred_length*args.nskip+nb+1:args.nskip,0]
    dW[:]   = noise[nb+1:args.pred_length*args.nskip+nb+2:args.nskip,0]
    Ures[:] = unres[nb  :args.pred_length*args.nskip+nb+1:args.nskip,0]
    mv = np.empty(tt.shape[0])
    rv = np.empty(tt.shape[0])
    Gk = np.empty(tt.shape[0])
    mv[:] = mu[nb:args.pred_length*args.nskip+nb+1:args.nskip,0]
    rv[:] = Ru[nb:args.pred_length*args.nskip+nb+1:args.nskip,0]
    Gk[:] = Gf[nb:args.pred_length*args.nskip+nb+1:args.nskip,0]
    del(Us,dotUs,noise,unres, mu,Ru, Ff,Gf)
    
    npred = args.npred #10 #args.pred_length-args.input_length
    nskip = 10
    Nsamp  = (args.pred_length-args.input_length - args.npred+1) // nskip
    traj_set   = torch.zeros(args.input_length + args.npred, Nsamp, 8, dtype=torch.double)
    for l in range(Nsamp):
        traj_set[:,l, 0] = torch.from_numpy(   Ut[l*nskip:l*nskip+args.input_length + npred])
        traj_set[:,l, 1] = torch.from_numpy( dotU[l*nskip:l*nskip+args.input_length + npred])
        traj_set[:,l, 2] = torch.from_numpy(   mv[l*nskip:l*nskip+args.input_length + npred])
        traj_set[:,l, 3] = torch.from_numpy(   rv[l*nskip:l*nskip+args.input_length + npred])
        traj_set[:,l, 4] = torch.from_numpy(   Fu[l*nskip:l*nskip+args.input_length + npred])
        traj_set[:,l, 5] = torch.from_numpy(   Gk[l*nskip:l*nskip+args.input_length + npred])
        traj_set[:,l, 6] = torch.from_numpy( Ures[l*nskip:l*nskip+args.input_length + npred])
        traj_set[:,l, 7] = torch.from_numpy(   dW[l*nskip:l*nskip+args.input_length + npred])
    del(data_load,Ut,dotU,mv,rv, Fu,Gk, dW,Ures)
    
    valid_pred, valid_err = prediction(traj_set, npred, model, params, dev)
    # tt = tt[::nskip]
    
    datapath = os.path.join(cfg['checkpoint'], 'leading' + fname)
    np.savez(datapath, tt = tt, pred = valid_pred[:,:,:,0], gt = valid_pred[:,:,:,1], valid_err = valid_err)

def prediction(input_set, npred, model, params, dev):
    model_m, model_v = model
    with torch.no_grad():
        model_m.eval()
        model_v.eval()
    dyad_cond = dyad.CondGau(dt=args.dt, params=params, device=dev)
    dyad_u    = dyad.Dyn_u(dt=args.dt, params=params, device=dev)
 
    Nsamp = input_set.shape[1]
    valid_pred = np.zeros((10, Nsamp,6, 2))
    valid_err  = np.zeros((npred,6))
    
    ind1 = [0, 2, 6]
    ind2 = [0, 3, 5]
    indt = [0, 2,3, 4,5, 6]
    istate   = torch.empty(args.input_length, Nsamp, 8, dtype=torch.double, device=dev)
    istate_m = torch.empty(args.input_length, Nsamp, 3, dtype=torch.double, device=dev)
    istate_v = torch.empty(args.input_length, Nsamp, 3, dtype=torch.double, device=dev)
    istate_m = input_set[:args.input_length,:,ind1].clone().to(dev)
    istate_v = input_set[:args.input_length,:,ind2].clone().to(dev)
    istate   = input_set[:args.input_length,:,:].clone().to(dev)

    ic=0
    hidden_m, hidden_v = (), ()
    with torch.no_grad():
        for istep in range(npred):   
            # target set data
            # target  = input_set[(istep+1): args.input_length + (istep+1), :, indt]
            # run model in one forward iteration -- Euler
            Ur_out, hidden_m = model_m(istate_m, hidden_m, device=dev)
            Ur_out = torch.squeeze(Ur_out)
            Gk_out, hidden_v = model_v(istate_v, hidden_v, device=dev)
            Gk_out = torch.squeeze(Gk_out)
            # forward euler
            istate[:,:,5] = Gk_out
            istate[:,:,6] = Ur_out
            U_out, dotU_out = dyad_u.update_u_euler(istate)
            istate[:,:,1] = dotU_out
            mv_out, rv_out, Fu_out = dyad_cond.update_condGau_euler(istate)
            
            
            istate_m[:-1,:,:] = istate_m[1:,:,:].clone()
            istate_v[:-1,:,:] = istate_v[1:,:,:].clone()
            istate[:-1,:,:]   = istate[1:,:,:].clone()
            
            istate[-1,:,7] = input_set[args.input_length+istep,:,7]
            
            istate_m[-1,:,0] = U_out[-1]
            istate_m[-1,:,1] = mv_out[-1]
            istate_m[-1,:,2] = Ur_out[-1]
            istate_v[-1,:,0] = U_out[-1]
            istate_v[-1,:,1] = rv_out[-1]
            istate_v[-1,:,2] = Gk_out[-1]
            istate[-1,:,0]   = U_out[-1]
            istate[-1,:,1]   = dotU_out[-1]
            istate[-1,:,2]   = mv_out[-1]
            istate[-1,:,3]   = rv_out[-1]
            istate[-1,:,4]   = Fu_out[-1]
            istate[-1,:,5]   = Gk_out[-1]
            istate[-1,:,6]   = Ur_out[-1]
            

            
            targ = input_set[args.input_length+istep,:,indt]
            valid_err[istep,1] = np.square(mv_out.data.cpu().numpy()[-1]  - targ.data.cpu().numpy()[:,1]).mean(0)
            valid_err[istep,2] = np.square(rv_out.data.cpu().numpy()[-1]  - targ.data.cpu().numpy()[:,2]).mean(0)
            valid_err[istep,3] = np.square(Fu_out.data.cpu().numpy()[-1]  - targ.data.cpu().numpy()[:,3]).mean(0)
            valid_err[istep,4] = np.square(Gk_out.data.cpu().numpy()[-1]  - targ.data.cpu().numpy()[:,4]).mean(0)
            valid_err[istep,5] = np.square(Ur_out.data.cpu().numpy()[-1]  - targ.data.cpu().numpy()[:,5]).mean(0)
            valid_err[istep,0] = np.square(U_out.data.cpu().numpy()[-1]   - targ.data.cpu().numpy()[:,0]).mean(0)
            print('step {}: err_U = {:.6f} err_Ur = {:.6f} err_u = {:.6f} err_Fu = {:.6f} error_R = {:.6f} error_Gk = {:.6f}'.format(istep, 
                  valid_err[istep,0], valid_err[istep,5], valid_err[istep,1], 
                  valid_err[istep,3], valid_err[istep,2], valid_err[istep,4]) )
                
            if (istep+1) in [10,20,50,100,200,250,500,750,800,1000]: #[10,20,50,100,150,200,250,300,400,500]:
                predv   = mv_out.data.cpu().numpy()[-1]
                predrv  = rv_out.data.cpu().numpy()[-1]
                predFu  = Fu_out.data.cpu().numpy()[-1]
                predGk  = Gk_out.data.cpu().numpy()[-1]
                predU   = U_out.data.cpu().numpy()[-1]
                predUr  = Ur_out.data.cpu().numpy()[-1]
                targ    = targ.data.cpu().numpy()
                valid_pred[ic,:,1, 0] = predv
                valid_pred[ic,:,2, 0] = predrv
                valid_pred[ic,:,3, 0] = predFu
                valid_pred[ic,:,4, 0] = predGk
                valid_pred[ic,:,5, 0] = predUr
                valid_pred[ic,:,0, 0] = predU
                valid_pred[ic,:,:, 1] = targ
                ic = ic+1

        # err_ave = valid_err.mean(1)

        
    return valid_pred, valid_err

if __name__ == '__main__':
    gflags.FLAGS(sys.argv)
    gflags.DEFINE_boolean('pretrained', False, 'Use pretrained model')
    gflags.DEFINE_boolean('eval', True, 'Run tests with the network')
    
    main(pretrained = gflags.FLAGS.pretrained, valid = gflags.FLAGS.eval)
