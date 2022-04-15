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

from utils import Logger, AverageMeter
import models
import baro_topo

# hyper parameters for training
parser = argparse.ArgumentParser(description='model configuration')
# data loading parameters
parser.add_argument('--train_length', default=200109, type=int, metavar='T',
                    help='sequence length for training samples')
parser.add_argument('--pred_length', default=200000, type=int, metavar='T',
                    help='sequence length for training samples')
parser.add_argument('--input_length', default=100, type=int, metavar='L',
                    help='model input state size')
parser.add_argument('--iters', default=100, type=int, metavar='I',
                    help='number of iterations for each epoch')
parser.add_argument('--train-batch', default=100, type=int, metavar='B',
                    help='each training batch size')
parser.add_argument('--nskip', default=1, type=int, metavar='nk',
                    help='time step skip in the loaded raw data, dt=1e-2')
parser.add_argument('--npred', default=10, type=int, metavar='Np',
                    help='number of iterations to measure in the loss func.')
parser.add_argument('--kmax', default=10, type=int, metavar='km',
                    help='total number of spectral modes in the full model')
parser.add_argument('--kred', default=2, type=int, metavar='kr',
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
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[40,75],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.5, 
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.5, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--loss-type', '--lt', default='kld', type=str, metavar='LT',
                    help='choices of loss functions (state,flux,comb, kld,mixed)')

# checkpoints/data setting
parser.add_argument('-c', '--checkpoint', default='checkpoint/train_flow_K10sk20sU10_condGau', type=str, metavar='C_PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--data-file', default='data/baro_K10sk20sU10dk1dU1', type=str, metavar='DATA_PATH',
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

# save config
fname = 'flowred{}_euler_adamlr01_ls2stg0hs{}hsm{}nl{}_seq{}nsk{}np{}_epoch{}_'.format(args.kred, args.nhid,args.nhidm, args.nloss,
            args.input_length, args.nskip, args.npred, args.epoch) + args.loss_type
with open(args.checkpoint + "/config_"+fname+".txt", 'w') as f:
    for (k, v) in args._get_kwargs():
        f.write(k + ' : ' + str(v) + '\n')
    f.write('\n')

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
    print('This code is run by {} and {}: {} GPU(s)'.format(dev1, dev2, torch.cuda.device_count()))
    if  torch.cuda.device_count() > 1:
        model_m = nn.DataParallel(model_m).to(dev1)
        model_v = nn.DataParallel(model_v).to(dev2)
    cudnn.benchmark = True
    cudnn.enabled = True
    
    if pretrained:
        # load the pretrained model
        model_path1 = torch.load(os.path.join(cfg['checkpoint'], 'modelm_'+fname), map_location=dev1)
        model_m.load_state_dict(model_path1['model_state_dict'])
        model_path2 = torch.load(os.path.join(cfg['checkpoint'], 'modelv_'+fname), map_location=dev2)
        model_v.load_state_dict(model_path2['model_state_dict'])  
    ires = 0
    if args.resume == True:
        model_path1 = torch.load(os.path.join(cfg['checkpoint'], 'modelm_'+fname), map_location=dev1)
        model_m.load_state_dict(model_path1['model_state_dict'])
        model_path2 = torch.load(os.path.join(cfg['checkpoint'], 'modelv_'+fname), map_location=dev2)
        model_v.load_state_dict(model_path2['model_state_dict'])
        log = np.loadtxt(os.path.join(cfg['checkpoint'], 'log_'+fname+'.txt'), skiprows=1)
        ires = int(log[-1, 0]) + 1
    model_m.to(dev1)
    model_v.to(dev2)
    model = (model_m, model_v)
        
    with open(args.checkpoint + "/config_"+fname+".txt", 'a') as f:
        f.write('Total model params. for mean: {}'.format(sum(p.numel() for p in model_m.parameters())) + '\n')
        f.write('Total model params. for vari: {}'.format(sum(p.numel() for p in model_v.parameters())) + '\n')
    print('    Total mean model params.: {}'.format(sum(p.numel() for p in model_m.parameters())))
    print('    Total vari model params.: {}'.format(sum(p.numel() for p in model_v.parameters())))
    
    # loss function and optimizer
    if args.loss_type == 'state' or args.loss_type == 'flux' or args.loss_type == 'comb':
        criterion = nn.MSELoss(reduction='mean')
    elif args.loss_type == 'kld':
        criterion = nn.KLDivLoss(reduction='batchmean')
    elif args.loss_type == 'mixed':
        crion1 = nn.KLDivLoss(reduction='batchmean')
        crion2 = nn.MSELoss(reduction='mean')
        criterion = (crion1, crion2)
        

    optim_m = optim.Adam(model_m.parameters(), lr=args.lr, betas=(0.9,.99), weight_decay=args.weight_decay, amsgrad = True)
    optim_v = optim.Adam(model_v.parameters(), lr=args.lr, betas=(0.9,.99), weight_decay=args.weight_decay, amsgrad = True)
    optimizer = (optim_m, optim_v)

    # logger
    logger = Logger(os.path.join(args.checkpoint, 'log_'+fname+'.txt'), title = 'log', resume=args.resume)
    if ires == 0:
        logger.set_names(['Epoch', '        Learning Rate.', 'Train Loss.', 'Accu. U','        Accu. n_U', 'Accu. mu', 'Accu. f_u', 
                          'Accu. Ru', 'Accu. g_v'])
    
    # load dataset
    data_load = scipy.io.loadmat(args.data_file)
    params = data_load.get('params')[0,0]
    tt = np.transpose(data_load.get('TT'), (1,0))
    dt = data_load.get('Dt')[0,0]
    Us = np.transpose(data_load.get('Uout'), (1,0))
    dotUs = np.transpose(data_load.get('dUout'), (1,0))
    noise = np.transpose(data_load.get('noise'), (1,0))
    unres = np.transpose(data_load.get('unres'), (1,0))
    mu = np.transpose(data_load.get('umout'), (1,0))
    Ru = np.transpose(data_load.get('Ruout'), (1,0))
    Cu = np.transpose(data_load.get('Cuout'), (1,0))
    Ff = np.transpose(data_load.get('dfout'), (1,0))
    Gf = np.transpose(data_load.get('dgout'), (1,0))

    # load data in the observed step
    if args.kmax != params['kmax'][0,0]:
        print('Error: inconsistent maximum mode!')
        return 0
    nb = 10000
    tt = tt[nb:args.train_length*args.nskip+nb:args.nskip]
    args.dt = dt
    Ut   = np.empty(tt.shape[0])
    dotU = np.empty(tt.shape[0])
    Fu   = np.empty(tt.shape[0])
    Ures = np.empty(tt.shape[0])
    dW   = np.empty(tt.shape[0])
    Ut[:]   =    Us[nb  :args.train_length*args.nskip+nb  :args.nskip,0]
    dotU[:] = dotUs[nb  :args.train_length*args.nskip+nb  :args.nskip,0]
    Fu[:]   =    Ff[nb-1:args.train_length*args.nskip+nb-1:args.nskip,0]
    dW[:]   = noise[nb+1:args.train_length*args.nskip+nb+1:args.nskip,0]
    Ures[:] = unres[nb-1:args.train_length*args.nskip+nb-1:args.nskip,0]
    uk = np.empty((tt.shape[0], 4*args.kmax))
    rk = np.empty((tt.shape[0], 2*args.kmax))
    ck = np.empty((tt.shape[0], 2*args.kmax))
    Gk = np.empty((tt.shape[0], 4*args.kmax))
    uk[:,:args.kmax]              = mu[nb  :args.train_length*args.nskip+nb:args.nskip,:args.kmax].real
    uk[:,args.kmax:2*args.kmax]   = mu[nb  :args.train_length*args.nskip+nb:args.nskip,:args.kmax].imag
    uk[:,2*args.kmax:3*args.kmax] = mu[nb  :args.train_length*args.nskip+nb:args.nskip,2*args.kmax:3*args.kmax].real
    uk[:,3*args.kmax:]            = mu[nb  :args.train_length*args.nskip+nb:args.nskip,2*args.kmax:3*args.kmax].imag
    Gk[:,:args.kmax]              = Gf[nb-1:args.train_length*args.nskip+nb-1:args.nskip,:args.kmax].real
    Gk[:,args.kmax:2*args.kmax]   = Gf[nb-1:args.train_length*args.nskip+nb-1:args.nskip,:args.kmax].imag
    Gk[:,2*args.kmax:3*args.kmax] = Gf[nb-1:args.train_length*args.nskip+nb-1:args.nskip,2*args.kmax:3*args.kmax].real
    Gk[:,3*args.kmax:4*args.kmax] = Gf[nb-1:args.train_length*args.nskip+nb-1:args.nskip,2*args.kmax:3*args.kmax].imag
    for ii in range(args.kmax):
        rk[:,ii]           = Ru[nb:args.train_length*args.nskip+nb:args.nskip, ii            ].real
        rk[:,ii+args.kmax] = Ru[nb:args.train_length*args.nskip+nb:args.nskip, ii+2*args.kmax].real
        ck[:,ii]           = Cu[nb:args.train_length*args.nskip+nb:args.nskip, ii].real
        ck[:,ii+args.kmax] = Cu[nb:args.train_length*args.nskip+nb:args.nskip, ii].imag
    del(Us,dotUs,noise,unres, mu,Ru,Cu, Ff,Gf)


    nskip = 20 
    Nsamp  = (args.train_length-args.input_length - args.npred+1) // nskip
    args.iters = Nsamp//args.train_batch
    train_set   = torch.zeros(args.input_length + args.npred-1, Nsamp, 12*args.kred+5, dtype=torch.double)
    target_set  = torch.zeros(args.input_length + args.npred-1, Nsamp, 12*args.kred+5, dtype=torch.double)
    ind1 = np.concatenate([range(0,args.kred),range(args.kmax,args.kmax+args.kred)])
    ind2 = np.concatenate([range(2*args.kmax,2*args.kmax+args.kred),range(3*args.kmax,3*args.kmax+args.kred)])
    for l in range(Nsamp): 
        train_set[:, l, 0] = torch.from_numpy(  Ut[l*nskip:l*nskip+args.input_length + args.npred-1])
        train_set[:, l, 1] = torch.from_numpy(dotU[l*nskip:l*nskip+args.input_length + args.npred-1])
        train_set[:, l, 2:2+2*args.kred]             = torch.from_numpy(uk[l*nskip:l*nskip+args.input_length + args.npred-1,ind1])
        train_set[:, l, 2+2*args.kred:2+4*args.kred] = torch.from_numpy(uk[l*nskip:l*nskip+args.input_length + args.npred-1,ind2])
        train_set[:, l, 2+4*args.kred:2+6*args.kred] = torch.from_numpy(rk[l*nskip:l*nskip+args.input_length + args.npred-1,ind1])
        train_set[:, l, 2+6*args.kred:2+8*args.kred] = torch.from_numpy(ck[l*nskip:l*nskip+args.input_length + args.npred-1,ind1])
        train_set[:, l, 2+8*args.kred]               = torch.from_numpy(Fu[l*nskip:l*nskip+args.input_length + args.npred-1])
        train_set[:, l, 3+8*args.kred:3+10*args.kred]  = torch.from_numpy(Gk[l*nskip:l*nskip+args.input_length + args.npred-1,ind1])
        train_set[:, l, 3+10*args.kred:3+12*args.kred] = torch.from_numpy(Gk[l*nskip:l*nskip+args.input_length + args.npred-1,ind2])
        train_set[:, l, 3+12*args.kred] = torch.from_numpy(Ures[l*nskip:l*nskip+args.input_length + args.npred-1])
        train_set[:, l, 4+12*args.kred] = torch.from_numpy(  dW[l*nskip:l*nskip+args.input_length + args.npred-1])
        
        target_set[:, l, 0] = torch.from_numpy(  Ut[l*nskip+1:l*nskip+args.input_length + args.npred])
        target_set[:, l, 1] = torch.from_numpy(dotU[l*nskip+1:l*nskip+args.input_length + args.npred])
        target_set[:, l, 2:2+2*args.kred]             = torch.from_numpy(uk[l*nskip+1:l*nskip+args.input_length + args.npred,ind1])
        target_set[:, l, 2+2*args.kred:2+4*args.kred] = torch.from_numpy(uk[l*nskip+1:l*nskip+args.input_length + args.npred,ind2])
        target_set[:, l, 2+4*args.kred:2+6*args.kred] = torch.from_numpy(rk[l*nskip+1:l*nskip+args.input_length + args.npred,ind1])
        target_set[:, l, 2+6*args.kred:2+8*args.kred] = torch.from_numpy(ck[l*nskip+1:l*nskip+args.input_length + args.npred,ind1])
        target_set[:, l, 2+8*args.kred]               = torch.from_numpy(Fu[l*nskip+1:l*nskip+args.input_length + args.npred])
        target_set[:, l, 3+8*args.kred:3+10*args.kred]  = torch.from_numpy(Gk[l*nskip+1:l*nskip+args.input_length + args.npred,ind1])
        target_set[:, l, 3+10*args.kred:3+12*args.kred] = torch.from_numpy(Gk[l*nskip+1:l*nskip+args.input_length + args.npred,ind2])
        target_set[:, l, 3+12*args.kred] = torch.from_numpy(Ures[l*nskip+1:l*nskip+args.input_length + args.npred])
        target_set[:, l, 4+12*args.kred] = torch.from_numpy(  dW[l*nskip+1:l*nskip+args.input_length + args.npred])

    train_loader = (train_set, target_set)
    del(data_load,Ut,dotU,uk,rk,ck,Fu,Gk, dW,Ures)
    

    # training performance measure
    epoch_loss = np.zeros((args.epoch, 2))
    epoch_accU = np.zeros((args.epoch, 2))
    epoch_accm = np.zeros((args.epoch, 4*args.kred, 2))
    epoch_accv = np.zeros((args.epoch, 4*args.kred, 2))
    epoch_accth = np.zeros((args.epoch,2+4*args.kred, 2))
    for epoch in range(ires, args.epoch):
        adjust_learning_rate(optimizer, epoch)
        print('\nEpoch: [{} | {}] LR: {:.8f} loss: {}'.format(epoch + 1, cfg['epoch'], cfg['lr'], cfg['loss_type']))
        train_loss,vloss, train_accU,vaccU, train_accm,vaccm, train_accFu,vaccFu, train_accv,vaccv, train_accGk,vaccGk, \
        train_accUr,vaccUr, pred, gt = train(train_loader, model, criterion, optimizer, params, device)

        # save accuracy
        epoch_loss[epoch,0]  = train_loss
        epoch_accU[epoch,0] = train_accU
        epoch_accm[epoch,:,0]  = train_accm
        epoch_accv[epoch,:,0]  = train_accv
        epoch_accth[epoch,0,0] = train_accFu
        epoch_accth[epoch,1:-1,0] = train_accGk
        epoch_accth[epoch,-1,0] = train_accUr
        epoch_loss[epoch,1]  = vloss
        epoch_accU[epoch,1] = vaccU
        epoch_accm[epoch,:,1]  = vaccm
        epoch_accv[epoch,:,1]  = vaccv
        epoch_accth[epoch,0,1] = vaccFu
        epoch_accth[epoch,1:-1,1] = vaccGk
        epoch_accth[epoch,-1,1] = vaccUr
        
        # append logger file
        logger.append([epoch, cfg['lr'], train_loss, train_accU,train_accUr, train_accm.sum(),train_accFu, 
                       train_accv.sum(),train_accGk.sum()])
        filepath1 = os.path.join(cfg['checkpoint'], 'modelm_' + fname)
        torch.save({'model_state_dict': model_m.state_dict(), 
                    'optimizer_state_dict': optim_m.state_dict(),}, filepath1)
        filepath2 = os.path.join(cfg['checkpoint'], 'modelv_' + fname)
        torch.save({'model_state_dict': model_v.state_dict(), 
                    'optimizer_state_dict': optim_v.state_dict(),}, filepath2)
        
    datapath = os.path.join(cfg['checkpoint'], 'train1_' + fname)
    np.savez(datapath, tt = tt, epoch_loss = epoch_loss, epoch_accU = epoch_accU, epoch_accm = epoch_accm,
             epoch_accv = epoch_accv, epoch_accth = epoch_accth, pred = pred, gt = gt) 
    
    # evaluating model in prediction data set
    if valid:
        # load evaluation dataset
        data_load = scipy.io.loadmat(args.data_file)
        params = data_load.get('params')[0,0]
        tt = np.transpose(data_load.get('TT'), (1,0))
        dt = data_load.get('Dt')[0,0]
        Us = np.transpose(data_load.get('Uout'), (1,0))
        dotUs = np.transpose(data_load.get('dUout'), (1,0))
        noise = np.transpose(data_load.get('noise'), (1,0))
        unres = np.transpose(data_load.get('unres'), (1,0))
        mu = np.transpose(data_load.get('umout'), (1,0))
        Ru = np.transpose(data_load.get('Ruout'), (1,0))
        Cu = np.transpose(data_load.get('Cuout'), (1,0))
        Ff = np.transpose(data_load.get('dfout'), (1,0))
        Gf = np.transpose(data_load.get('dgout'), (1,0))
        
        nb = 10000
        tt = tt[nb:args.pred_length*args.nskip+nb+1:args.nskip]
        args.dt = dt
        Ut = np.empty(tt.shape[0])
        dotU = np.empty(tt.shape[0])
        Fu = np.empty(tt.shape[0])
        dW = np.empty(tt.shape[0])
        Ures = np.empty(tt.shape[0])
        Ut[:] = Us[nb:args.pred_length*args.nskip+nb+1:args.nskip,0]
        dotU[:] = dotUs[nb:args.pred_length*args.nskip+nb+1:args.nskip,0]
        Fu[:] = Ff[nb-1:args.pred_length*args.nskip+nb:args.nskip,0]
        dW[:] = noise[nb+1:args.pred_length*args.nskip+nb+2:args.nskip,0]
        Ures[:] = unres[nb-1:args.pred_length*args.nskip+nb:args.nskip,0]
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
        del(Us,dotUs,noise,unres, mu,Ru,Cu, Ff,Gf)
        
        npred = args.pred_length-args.input_length
        ind1 = np.concatenate([range(0,args.kred),range(args.kmax,args.kmax+args.kred)])
        ind2 = np.concatenate([range(2*args.kmax,2*args.kmax+args.kred),range(3*args.kmax,3*args.kmax+args.kred)])
        traj_set = torch.zeros(args.pred_length+1, 1, 5+12*args.kred, dtype=torch.double)
        traj_set[:,0, 0] = torch.from_numpy(Ut)
        traj_set[:,0, 1] = torch.from_numpy(dotU)
        traj_set[:,0, 2:2+2*args.kred] = torch.from_numpy(uk)[:,ind1]
        traj_set[:,0, 2+2*args.kred:2+4*args.kred] = torch.from_numpy(uk)[:,ind2]
        traj_set[:,0, 2+4*args.kred:2+6*args.kred] = torch.from_numpy(rk[:,ind1])
        traj_set[:,0, 2+6*args.kred:2+8*args.kred] = torch.from_numpy(ck[:,ind1])
        traj_set[:,0, 2+8*args.kred] = torch.from_numpy(Fu)
        traj_set[:,0, 3+8*args.kred:3+10*args.kred] = torch.from_numpy(Gk[:,ind1])
        traj_set[:,0, 3+10*args.kred:3+12*args.kred] = torch.from_numpy(Gk[:,ind2])
        traj_set[:,0, 3+12*args.kred] = torch.from_numpy(Ures)
        traj_set[:,0, 4+12*args.kred] = torch.from_numpy(dW)
        del(data_load,Ut,dotU,uk,rk,ck,Fu,Gk, dW,Ures)
        
        
        logger.file.write('\n')
        logger.set_names(['Model eval.', 'total', '        error U', '        error U_unres', '        error u', 
                          '        error theta_m', ' error r', ' error theta_v'])
        valid_pred, valid_err = prediction(traj_set, npred, model, params, logger, device)
        
        datapath = os.path.join(cfg['checkpoint'], 'pred1_' + fname)
        np.savez(datapath, tt = tt, pred = valid_pred[:,:,:,0], gt = valid_pred[:,:,:,1], valid_err = valid_err)

    logger.close()
    
def prediction(input_set, npred, model, params, logger, device):
    dev1, dev2 = device
    model_m, model_v = model
    with torch.no_grad():
        model_m.eval()
        model_v.eval()
    baro_cond = baro_topo.CondGau(dt=args.dt, params=params, kred=args.kred, device=dev1)
    baro_flow = baro_topo.BaroTopo(dt=args.dt, params=params, kred=args.kred, device=dev1)
 
    kred = args.kred
    valid_pred = np.zeros((npred, 1,3+12*kred, 2))
    valid_err  = np.zeros((npred, 1,3+12*kred))

    ind1 = np.concatenate([range(1),range(2,2+2*kred),range(2+4*kred,2+5*kred),range(3+12*kred,4+12*kred)])
    ind2 = np.concatenate([range(1),range(2+4*kred,2+8*kred),range(3+8*kred,3+12*kred)])
    indt = np.concatenate([range(1),range(2,4+12*kred)])
    istate_m = input_set[:args.input_length,:,ind1].clone().to(dev1)
    istate_v = input_set[:args.input_length,:,ind2].clone().to(dev2)
    istate   = input_set[:args.input_length,:,:].clone().to(dev1)
        
    hidden_m, hidden_v = (), ()
    with torch.no_grad():
        for istep in range(npred):
            # target set data
            target  = input_set[(istep+1): args.input_length + (istep+1), :, indt]

            ############################################################
            # run model in one forward iteration
            Ur_out, hidden_m = model_m(istate_m, hidden_m, device=dev1)
            Ur_out = torch.squeeze(Ur_out)
            Gk_out, hidden_v = model_v(istate_v, hidden_v, device=dev2)

            istate[:,:,3+8*args.kred:3+12*args.kred] = Gk_out.to(dev1)
            istate[:,0,3+12*args.kred] = Ur_out.to(dev1)
            
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
            istate[:,0,3+12*args.kred] = Ur1.to(dev1)
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
            istate1[:,0,3+12*args.kred] = Ur2.to(dev1)
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
            istate2[:,0,3+12*args.kred] = Ur3.to(dev1)
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
            istate3[:,0,3+12*args.kred] = Ur4.to(dev1)
            # rk4 -- step 4
            istate4, incre4 = baro_cond.update_condGau_step(istate3,istate, args.dt)
            U4, increU4 = baro_flow.update_flow_step(istate3,istate, args.dt)
            istate4[:,:,0] = U4
            (inc_vk4, inc_Tk4, inc_rk4, inc_ck4) = incre4
            Fu4 = istate4[:,:,2+8*args.kred]
            
            
            Fu_out = Fu1 #(Fu1+2*Fu2+2*Fu3+Fu4)/6
            Gk_out = Gk1 #(Gk1+2*Gk2+2*Gk3+Gk4) / 6
            Ur_out = Ur1 #(Ur1+2*Ur2+2*Ur3+Ur4) / 6
            Gk_p = (Gk1+2*Gk2+2*Gk3+Gk4) / 6
            Ur_p = (Ur1+2*Ur2+2*Ur3+Ur4) / 6
            
            dW = input_set[istep:args.input_length+istep,:,4+12*args.kred].to(dev1)
            U_out  = istate[:,:,0] + args.dt * (increU1 + 2*increU2 + 2*increU3 + increU4) / 6 + dW
            dotU_out = (increU1 + 2*increU2 + 2*increU3 + increU4) / 6 + dW/args.dt
            vk_out = istate[:,:,2       :2+2*kred] + args.dt * (inc_vk1 + 2*inc_vk2 + 2*inc_vk3 + inc_vk4) / 6
            Tk_out = istate[:,:,2+2*kred:2+4*kred] + args.dt * (inc_Tk1 + 2*inc_Tk2 + 2*inc_Tk3 + inc_Tk4) / 6
            rk_out = istate[:,:,2+4*kred:2+6*kred] + args.dt * (inc_rk1 + 2*inc_rk2 + 2*inc_rk3 + inc_rk4) / 6
            ck_out = istate[:,:,2+6*kred:2+8*kred] + args.dt * (inc_ck1 + 2*inc_ck2 + 2*inc_ck3 + inc_ck4) / 6
            '''
            ##################################################################################################
            
            
            istate_m[:-1,:,:] = istate_m[1:,:,:].clone()
            istate_v[:-1,:,:] = istate_v[1:,:,:].clone()
            istate[:-1,:,:] = istate[1:,:,:].clone()
            
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
            istate[-1,:,3+8*kred:3+12*kred] = Gk_out[-1].to(dev1)
            istate[-1,:,3+12*kred]          = Ur_out[-1]
            

            predv  = vk_out.data.cpu().numpy()[-1]
            predT  = Tk_out.data.cpu().numpy()[-1]
            predrk  = rk_out.data.cpu().numpy()[-1]
            predck  = ck_out.data.cpu().numpy()[-1]
            predFu = Fu_out.data.cpu().numpy()[-1]
            predGk = Gk_out.data.cpu().numpy()[-1]
            predUr = Ur_out.data.cpu().numpy()[-1]
            predU  = U_out.data.cpu().numpy()[-1]
            targ  = target.data.cpu().numpy()[-1]
            valid_pred[istep, :,:2*kred, 0] = predv
            valid_pred[istep, :,2*kred:4*kred, 0] = predT
            valid_pred[istep, :,4*kred:6*kred, 0] = predrk
            valid_pred[istep, :,6*kred:8*kred, 0] = predck
            valid_pred[istep, :,8*kred, 0] = predFu
            valid_pred[istep, :,8*kred+1:12*kred+1, 0] = predGk
            valid_pred[istep, :,12*kred+1, 0] = predUr
            valid_pred[istep, :,12*kred+2, 0] = predU
            valid_pred[istep, :,:, 1] = targ
            
            valid_err[istep, :,:2*kred] = ( np.square(predv  - targ[:,1:1+2*kred]) )
            valid_err[istep, :,2*kred:4*kred] = ( np.square(predT  - targ[:,1+2*kred:1+4*kred]) )
            valid_err[istep, :,4*kred:6*kred] = ( np.square(predrk  - targ[:,1+4*kred:1+6*kred]) )
            valid_err[istep, :,6*kred:8*kred] = ( np.square(predck  - targ[:,1+6*kred:1+8*kred]) )
            valid_err[istep, :,8*kred] = ( np.square(predFu - targ[:,1+8*kred]) )
            valid_err[istep, :,1+8*kred:1+12*kred] = ( np.square(predGk - targ[:,2+8*kred:2+12*kred]) )
            valid_err[istep, :,1+12*kred] = ( np.square(predUr - targ[:,2+12*kred]) )
            valid_err[istep, :,2+12*kred] = ( np.square(predU - targ[:,0]) )

            err_ave = valid_err.mean(1)
            print('step {}: err_U = {:.6f} err_Ur = {:.6f} err_u = {:.6f} err_Fu = {:.6f} error_R = {:.6f} error_Gk = {:.6f}'.format(istep, 
                  err_ave[istep,2+12*kred], err_ave[istep,1+12*kred], err_ave[istep,:2*kred].sum(), 
                  err_ave[istep,8*kred], err_ave[istep,4*kred:6*kred].sum(), err_ave[istep,1+8*kred:1+12*kred].sum()) )
            logger.append([istep, err_ave[istep,:].sum(), err_ave[istep,2+12*kred],err_ave[istep,1+12*kred], err_ave[istep,:4*kred].sum(), 
                           err_ave[istep,8*kred], err_ave[istep,4*kred:8*kred].sum(), err_ave[istep,8*kred+1:12*kred+1].sum() ])
        
    return valid_pred, valid_err
    
def train(train_loader, model, criterion, optimizer, params, device):
    dev1, dev2 = device
    model_m, model_v = model
    optim_m, optim_v = optimizer
    model_m.train()
    model_v.train()
    baro_cond = baro_topo.CondGau( dt=args.dt, params=params, kred=args.kred, device=dev1)
    baro_flow = baro_topo.BaroTopo(dt=args.dt, params=params, kred=args.kred, device=dev1)
    
    batch_time = AverageMeter()
    losses     = AverageMeter()
    accsU      = AverageMeter()
    accsm      = AverageMeter()
    accsv      = AverageMeter()
    accsFu     = AverageMeter()
    accsGk     = AverageMeter()
    accsUr     = AverageMeter()
    end = time.time()
    
    kmax = params['kmax'][0,0]
    kred = args.kred
    input_full, target_full = train_loader
    dsize = args.train_batch*args.iters
    s_idx = random.sample(range(0,input_full.size(1)), dsize)
    input_iter   = input_full[:, s_idx,:].pin_memory()
    target_iter  = target_full[:,s_idx,:].pin_memory()
    for ib in range(0, args.iters):
        ind1 = np.concatenate([range(1),range(2,2+2*kred),range(2+4*kred,2+5*kred),range(3+12*kred,4+12*kred)])
        ind2 = np.concatenate([range(1),range(2+4*kred,2+8*kred),range(3+8*kred,3+12*kred)])
        inputs   = input_iter[:, ib*args.train_batch:(ib+1)*args.train_batch, :].to(dev1, non_blocking=True)
        inputs_m = input_iter[:, ib*args.train_batch:(ib+1)*args.train_batch, ind1].to(dev1, non_blocking=True)
        inputs_v = input_iter[:, ib*args.train_batch:(ib+1)*args.train_batch, ind2].to(dev2, non_blocking=True)
        indt = np.concatenate([range(1),range(2,4+12*kred)])
        targets = target_iter[:, ib*args.train_batch:(ib+1)*args.train_batch, indt].to(dev1, non_blocking=True)
        
        optim_m.zero_grad()
        optim_v.zero_grad()  # zero the gradient buffers
        # iteration the model in npred steps
        hidden_m, hidden_v = (), ()
        istate   = torch.empty(args.input_length, args.train_batch, 5+12*kred, dtype=torch.double, device=dev1)
        istate_m = torch.empty(args.input_length, args.train_batch, 2+3*kred,  dtype=torch.double, device=dev1)
        istate_v = torch.empty(args.input_length, args.train_batch, 1+8*kred,  dtype=torch.double, device=dev2)
        istate_m[:,:,:] = inputs_m[:args.input_length,:,:]
        istate_v[:,:,:] = inputs_v[:args.input_length,:,:]
        istate[:,:,:]   = inputs[:args.input_length,:,:]
        
        pred = torch.empty(args.input_length+args.npred, args.train_batch, 3+12*kred, dtype=torch.double, device=dev1)
        pred[:args.input_length,:,:-1] = inputs[:args.input_length,:,2:-1].clone()
        pred[:args.input_length,:,-1] = inputs[:args.input_length,:,0].clone()
        loss = 0
        for ip in range(args.npred):
            ###########################################################
            # One-step
            Ur_out, hidden_m = model_m(istate_m, hidden_m, device=dev1)
            Ur_out = torch.squeeze(Ur_out)
            Gk_out, hidden_v = model_v(istate_v, hidden_v, device=dev2)

            istate[:,:,3+8*args.kred:3+12*args.kred] = Gk_out.to(dev1)
            istate[:,:,3+12*args.kred] = Ur_out.to(dev1)
            istate1, incre = baro_cond.update_condGau_step(istate,istate, args.dt)
            U1, increU  = baro_flow.update_flow_step(istate,istate, args.dt)
            (inc_vk, inc_Tk, inc_rk, inc_ck) = incre
            Fu_out = istate1[:,:,2+8*args.kred]
            Ur_p = Ur_out
            Gk_p = Gk_out

            
            dW = inputs[ip:args.input_length+ip,:,4+12*args.kred]
            U_out  = istate[:,:,0] + args.dt * increU + dW
            dotU_out = increU + dW/args.dt
            vk_out = istate[:,:,2            :2+2*args.kred] + args.dt * inc_vk
            Tk_out = istate[:,:,2+2*args.kred:2+4*args.kred] + args.dt * inc_Tk
            rk_out = istate[:,:,2+4*args.kred:2+6*args.kred] + args.dt * inc_rk
            ck_out = istate[:,:,2+6*args.kred:2+8*args.kred] + args.dt * inc_ck
            ###########################################################
            # 4-stage runge-kutta
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
            Gk_out = Gk1 #(Gk1+2*Gk2+2*Gk3+Gk4) / 6
            Ur_out = Ur1 #(Ur1+2*Ur2+2*Ur3+Ur4) / 6
            Gk_p = (Gk1+2*Gk2+2*Gk3+Gk4) / 6
            Ur_p = (Ur1+2*Ur2+2*Ur3+Ur4) / 6
            
            dW = inputs[ip:args.input_length+ip,:,4+12*args.kred]
            U_out  = istate[:,:,0] + args.dt * (increU1 + 2*increU2 + 2*increU3 + increU4) / 6 + dW
            dotU_out = (increU1 + 2*increU2 + 2*increU3 + increU4) / 6 + dW/args.dt
            vk_out = istate[:,:,2            :2+2*args.kred] + args.dt * (inc_vk1 + 2*inc_vk2 + 2*inc_vk3 + inc_vk4) / 6
            Tk_out = istate[:,:,2+2*args.kred:2+4*args.kred] + args.dt * (inc_Tk1 + 2*inc_Tk2 + 2*inc_Tk3 + inc_Tk4) / 6
            rk_out = istate[:,:,2+4*args.kred:2+6*args.kred] + args.dt * (inc_rk1 + 2*inc_rk2 + 2*inc_rk3 + inc_rk4) / 6
            ck_out = istate[:,:,2+6*args.kred:2+8*args.kred] + args.dt * (inc_ck1 + 2*inc_ck2 + 2*inc_ck3 + inc_ck4) / 6
            '''
            #############################################################################################################
            
            

            
            pred[args.input_length+ip,:, :2*kred]       = vk_out[-1]
            pred[args.input_length+ip,:, 2*kred:4*kred] = Tk_out[-1]
            pred[args.input_length+ip,:, 4*kred:6*kred] = rk_out[-1]
            pred[args.input_length+ip,:, 6*kred:8*kred] = ck_out[-1]
            pred[args.input_length+ip,:, 8*kred]        = Fu_out[-1]
            pred[args.input_length+ip,:, 8*kred+1:12*kred+1] = Gk_out[-1].to(dev1)
            pred[args.input_length+ip,:, 12*kred+1]       = Ur_out[-1]
            pred[args.input_length+ip,:, 12*kred+2]       = U_out[-1]
            
            if ip < args.npred-1:
                istate_m = torch.empty_like(istate_m)
                istate_v = torch.empty_like(istate_v)
                istate = torch.empty_like(istate)
                # update with final model output
                istate[:,:,2+8*kred] = inputs[ip+1:args.input_length+ip+1,:,2+8*kred]
                istate[:,:,4+12*kred] = inputs[ip+1:args.input_length+ip+1,:,4+12*kred]
                
                istate_m[:,:,0]                  = U_out
                istate_m[:,:,1:1+2*kred]         = vk_out
                istate_m[:,:,1+2*kred:1+3*kred]  = rk_out[:,:,:kred]
                istate_m[:,:,1+3*kred]           = Ur_p
                istate_v[:,:,0]                  = U_out
                istate_v[:,:,1:1+2*kred]         = rk_out
                istate_v[:,:,1+2*kred:1+4*kred]  = ck_out
                istate_v[:,:,1+4*kred:1+8*kred]  = Gk_p
                istate[:,:,0]                  = U_out
                istate[:,:,1]                  = dotU_out
                istate[:,:,2:2+2*kred]         = vk_out
                istate[:,:,2+2*kred:2+4*kred]  = Tk_out
                istate[:,:,2+4*kred:2+6*kred]  = rk_out
                istate[:,:,2+6*kred:2+8*kred]  = ck_out
                # istate[:,:,2+8*kmax]           = Fu_out
                istate[:,:,3+8*kred:3+12*kred] = Gk_out
                istate[:,:,3+12*kred] = Ur_out

            output = torch.transpose(torch.cat([U_out[:,:,None], vk_out, Tk_out, rk_out, ck_out, 
                                                Fu_out[:,:,None],Gk_out, Ur_out[:,:,None]],2), 0,1)
            target = torch.transpose(targets[ip:args.input_length+ip,:,:], 0,1)
            if args.loss_type == 'flux':
                out1 = output[:, -args.nloss:, 12*kred+2]
                tag1 = target[:, -args.nloss:, 12*kred+2]
                out2 = output[:, -args.nloss:, 8*kred+2:12*kred+2]
                tag2 = target[:, -args.nloss:, 8*kred+2:12*kred+2]
                loss += 1.*criterion(out2, tag2) + 1.*criterion(out1, tag1)
            elif args.loss_type == 'state':
                out1 = output[:, -args.nloss:, :1+4*kred]
                tag1 = target[:, -args.nloss:, :1+4*kred]
                out2 = output[:, -args.nloss:, 1+4*kred:1+8*kred]
                tag2 = target[:, -args.nloss:, 1+4*kred:1+8*kred]
                loss += 1*criterion(out1, tag1) + 1.*criterion(out2, tag2)
            elif args.loss_type == 'comb':
                out1 = output[:, -args.nloss:, 8*kred+2:12*kred+3]
                tag1 = target[:, -args.nloss:, 8*kred+2:12*kred+3]
                out2 = output[:, -args.nloss:, :8*kred+1]
                tag2 = target[:, -args.nloss:, :8*kred+1]
                loss += 1.*criterion(out1, tag1) + 1.*criterion(out2, tag2)
            elif args.loss_type == 'kld':
                out1 = output[:, -args.nloss:, :1+4*kred]
                tag1 = target[:, -args.nloss:, :1+4*kred]
                out2 = output[:, -args.nloss:, 1+4*kred:1+8*kred]
                tag2 = target[:, -args.nloss:, 1+4*kred:1+8*kred]
                out1_p = F.log_softmax(1. * out1, dim=1)
                tag1_p = F.softmax(1. * tag1, dim=1)
                out1_n = F.log_softmax(-1. * out1, dim=1)
                tag1_n = F.softmax(-1. * tag1, dim=1)
                out2_p = F.log_softmax(1. * out2, dim=1)
                tag2_p = F.softmax(1. * tag2, dim=1)
                loss += criterion(out1_p,tag1_p)+1.*criterion(out1_n,tag1_n) + \
                        1.*criterion(out2_p,tag2_p)

                
        
        loss.backward()
        optim_m.step()
        optim_v.step()
        
        # get trained output
        losses.update(loss.item() )
        pred_out = pred[args.input_length:]
        gt_out   = targets[args.input_length-1:]
        accsm.update( ((pred_out[:,:,:4*kred]-gt_out[:,:,1:1+4*kred]).square().mean(1).mean(0) ).data.cpu().numpy() )
        accsv.update( ((pred_out[:,:,4*kred:8*kred]-gt_out[:,:,1+4*kred:1+8*kred]).square().mean(1).mean(0) ).data.cpu().numpy() )
        accsFu.update( ((pred_out[:,:,8*kred]-gt_out[:,:,1+8*kred]).square().mean() ).item())
        accsGk.update( ((pred_out[:,:,8*kred+1:12*kred+1]-gt_out[:,:,8*kred+2:12*kred+2]).square().mean(1).mean(0) ).data.cpu().numpy() ) 
        accsUr.update( ((pred_out[:,:,12*kred+1]-gt_out[:,:,2+12*kred]).square().mean() ).item())
        accsU.update( ((pred_out[:,:,12*kred+2]-gt_out[:,:,0]).square().mean() ).item())
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        torch.cuda.empty_cache()
        
        #LR = optimizer.param_groups[0]['lr']
        suffix = 'Iter{iter}: Loss = {loss:.5e} U = {accU:.5e} Ur = {accUr:.5e} u = {accu:.5e} Fu = {accthm:.5e} R = {accr:.5e} Gk = {accthv:.5e}  run_time = {bt:.2f}'.format(
                  iter = ib, loss = losses.val, accU=accsU.val, accUr=accsUr.val, accu=accsm.val.sum(),accthm=accsFu.val, 
                  accr=accsv.val.sum(), accthv=accsGk.val.sum(), bt = batch_time.sum)
        if ib % 20 == 0:
            print(suffix)
            
    # get trained output
    pred = pred_out.data.cpu().numpy()
    gt   = gt_out.data.cpu().numpy()

    return losses.avg,losses.var, accsU.avg,accsU.var, accsm.avg,accsm.var,accsFu.avg,accsFu.var, accsv.avg,accsv.var, accsGk.avg,accsGk.var, \
           accsUr.avg,accsUr.var, pred, gt
 
def adjust_learning_rate(optimizer, epoch):
    global cfg
    if epoch in cfg['schedule']:
        cfg['lr'] *= args.gamma
        for ioptim in optimizer:
            for param_group in ioptim.param_groups:
                param_group['lr'] = cfg['lr']   

if __name__ == '__main__':
    gflags.FLAGS(sys.argv)
    gflags.DEFINE_boolean('pretrained', False, 'Use pretrained model')
    gflags.DEFINE_boolean('eval', True, 'Run tests with the network')
    
    main(pretrained = gflags.FLAGS.pretrained, valid = gflags.FLAGS.eval)
