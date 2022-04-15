''' this script works with CPU or single GPU
    training the conditional Gaussian equations for the dyad model

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

from utils import Logger, AverageMeter
import models
import dyad

# hyper parameters for training
parser = argparse.ArgumentParser(description='model configuration')
# data loading parameters
parser.add_argument('--train_length', default=1000104, type=int, metavar='T',
                    help='sequence length for training samples')
parser.add_argument('--pred_length', default=100000, type=int, metavar='T',
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
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[25,45],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.5, 
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.5, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--loss-type', '--lt', default='state', type=str, metavar='LT',
                    help='choices of loss functions (state, kld, mixed)')

# checkpoints/data setting
parser.add_argument('-c', '--checkpoint', default='checkpoint/train_dyad_su05sv2_condGau', type=str, metavar='C_PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--data-file', default='data/dyad_su05sv2', type=str, metavar='DATA_PATH',
                    help='path to train data set')
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
fname = 'dyad_adamlr01_hs{}hsm{}nl{}_seq{}nsk{}np{}_epoch{}_'.format(args.nhid,args.nhidm, args.nloss,
            args.input_length, args.nskip, args.npred, args.epoch) + args.loss_type
with open(args.checkpoint + "/config_"+fname+".txt", 'w') as f:
    for (k, v) in args._get_kwargs():
        f.write(k + ' : ' + str(v) + '\n')
    f.write('\n')

def main(pretrained = False, valid = False):
    # models for unresolved processes
    model_m = models.LSTMresi(input_size = 3, hidden_size = args.nhidm, output_size = 1, 
                                nlayers = 1, nstages = 0).double()
    model_v = models.LSTMresi(input_size = 3, hidden_size = args.nhid, output_size = 1, 
                                nlayers = 1, nstages = 0).double()
    # load model on GPU
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('This code is run by {}: {} GPU(s)'.format(dev, torch.cuda.device_count()))

    
    if pretrained:
        # load the pretrained model
        model_path1 = torch.load(os.path.join(cfg['checkpoint'], 'modelm_'+fname), map_location=dev)
        model_m.load_state_dict(model_path1['model_state_dict'])
        model_path2 = torch.load(os.path.join(cfg['checkpoint'], 'modelv_'+fname), map_location=dev)
        model_v.load_state_dict(model_path2['model_state_dict'])  
    ires = 0
    if args.resume == True:
        model_path1 = torch.load(os.path.join(cfg['checkpoint'], 'modelm_'+fname), map_location=dev)
        model_m.load_state_dict(model_path1['model_state_dict'])
        model_path2 = torch.load(os.path.join(cfg['checkpoint'], 'modelv_'+fname), map_location=dev)
        model_v.load_state_dict(model_path2['model_state_dict'])
        log = np.loadtxt(os.path.join(cfg['checkpoint'], 'log_'+fname+'.txt'), skiprows=1)
        ires = int(log[-1, 0]) + 1
    model_m.to(dev)
    model_v.to(dev)
    model = (model_m, model_v)
        
    with open(args.checkpoint + "/config_"+fname+".txt", 'a') as f:
        f.write('Total model params. for mean: {}'.format(sum(p.numel() for p in model_m.parameters())) + '\n')
        f.write('Total model params. for vari: {}'.format(sum(p.numel() for p in model_v.parameters())) + '\n')
    print('    Total mean model params.: {}'.format(sum(p.numel() for p in model_m.parameters())))
    print('    Total vari model params.: {}'.format(sum(p.numel() for p in model_v.parameters())))
    
    # loss function and optimizer
    if args.loss_type == 'state':
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
        logger.set_names(['Epoch', '        Learning Rate.', 'Train Loss.', '    Accu. U','        Accu. U_r', '    Accu. mv',
                          '    Accu. f_v', 'Accu. rv', 'Accu. g_v'])
    
    # load dataset
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

    # load data in the observed step
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
    Fu[:]   =    Ff[nb  :args.train_length*args.nskip+nb  :args.nskip,0]
    dW[:]   = noise[nb+1:args.train_length*args.nskip+nb+1:args.nskip,0]
    Ures[:] = unres[nb  :args.train_length*args.nskip+nb  :args.nskip,0]
    mv = np.empty(tt.shape[0])
    rv = np.empty(tt.shape[0])
    Gk = np.empty(tt.shape[0])
    mv[:] = mu[nb  :args.train_length*args.nskip+nb:args.nskip,0]
    rv[:] = Ru[nb  :args.train_length*args.nskip+nb:args.nskip,0]
    Gk[:] = Gf[nb  :args.train_length*args.nskip+nb:args.nskip,0]
    del(Us,dotUs,noise,unres, mu,Ru, Ff,Gf)


    nskip = 100 
    Nsamp  = (args.train_length-args.input_length - args.npred+1) // nskip
    args.iters = Nsamp//args.train_batch
    train_set   = torch.zeros(args.input_length + args.npred-1, Nsamp, 8, dtype=torch.double)
    target_set  = torch.zeros(args.input_length + args.npred-1, Nsamp, 8, dtype=torch.double)
    for l in range(Nsamp): 
        train_set[:, l, 0] = torch.from_numpy(  Ut[l*nskip:l*nskip+args.input_length + args.npred-1])
        train_set[:, l, 1] = torch.from_numpy(dotU[l*nskip:l*nskip+args.input_length + args.npred-1])
        train_set[:, l, 2] = torch.from_numpy(  mv[l*nskip:l*nskip+args.input_length + args.npred-1])
        train_set[:, l, 3] = torch.from_numpy(  rv[l*nskip:l*nskip+args.input_length + args.npred-1])
        train_set[:, l, 4] = torch.from_numpy(  Fu[l*nskip:l*nskip+args.input_length + args.npred-1])
        train_set[:, l, 5] = torch.from_numpy(  Gk[l*nskip:l*nskip+args.input_length + args.npred-1])
        train_set[:, l, 6] = torch.from_numpy(Ures[l*nskip:l*nskip+args.input_length + args.npred-1])
        train_set[:, l, 7] = torch.from_numpy(  dW[l*nskip:l*nskip+args.input_length + args.npred-1])
        
        target_set[:, l, 0] = torch.from_numpy(  Ut[l*nskip+1:l*nskip+args.input_length + args.npred])
        target_set[:, l, 1] = torch.from_numpy(dotU[l*nskip+1:l*nskip+args.input_length + args.npred])
        target_set[:, l, 2] = torch.from_numpy(  mv[l*nskip+1:l*nskip+args.input_length + args.npred])
        target_set[:, l, 3] = torch.from_numpy(  rv[l*nskip+1:l*nskip+args.input_length + args.npred])
        target_set[:, l, 4] = torch.from_numpy(  Fu[l*nskip+1:l*nskip+args.input_length + args.npred])
        target_set[:, l, 5] = torch.from_numpy(  Gk[l*nskip+1:l*nskip+args.input_length + args.npred])
        target_set[:, l, 6] = torch.from_numpy(Ures[l*nskip+1:l*nskip+args.input_length + args.npred])
        target_set[:, l, 7] = torch.from_numpy(  dW[l*nskip+1:l*nskip+args.input_length + args.npred])

    train_loader = (train_set, target_set)
    del(data_load,Ut,dotU,mv,rv,Fu,Gk, dW,Ures)
    

    # training performance measure
    epoch_loss = np.zeros((args.epoch, 2))
    epoch_accU = np.zeros((args.epoch, 2))
    epoch_accm = np.zeros((args.epoch, 2))
    epoch_accv = np.zeros((args.epoch, 2))
    epoch_accth = np.zeros((args.epoch,3, 2))
    for epoch in range(ires, args.epoch):
        adjust_learning_rate(optimizer, epoch)
        print('\nEpoch: [{} | {}] LR: {:.8f} loss: {}'.format(epoch + 1, cfg['epoch'], cfg['lr'], cfg['loss_type']))
        train_loss,vloss, train_accU,vaccU, train_accm,vaccm, train_accFu,vaccFu, train_accv,vaccv, train_accGk,vaccGk, \
        train_accUr,vaccUr, pred, gt = train(train_loader, model, criterion, optimizer, params, dev)

        # save accuracy
        epoch_loss[epoch,0]  = train_loss
        epoch_accU[epoch,0] = train_accU
        epoch_accm[epoch,0]  = train_accm
        epoch_accv[epoch,0]  = train_accv
        epoch_accth[epoch,0,0] = train_accFu
        epoch_accth[epoch,1,0] = train_accGk
        epoch_accth[epoch,2,0] = train_accUr
        epoch_loss[epoch,1]  = vloss
        epoch_accU[epoch,1] = vaccU
        epoch_accm[epoch,1]  = vaccm
        epoch_accv[epoch,1]  = vaccv
        epoch_accth[epoch,0,1] = vaccFu
        epoch_accth[epoch,1,1] = vaccGk
        epoch_accth[epoch,2,1] = vaccUr
        
        # append logger file
        logger.append([epoch, cfg['lr'], train_loss, train_accU,train_accUr, train_accm,train_accFu, 
                       train_accv,train_accGk])
        filepath1 = os.path.join(cfg['checkpoint'], 'modelm_' + fname)
        torch.save({'model_state_dict': model_m.state_dict(), 
                    'optimizer_state_dict': optim_m.state_dict(),}, filepath1)
        filepath2 = os.path.join(cfg['checkpoint'], 'modelv_' + fname)
        torch.save({'model_state_dict': model_v.state_dict(), 
                    'optimizer_state_dict': optim_v.state_dict(),}, filepath2)
        
    datapath = os.path.join(cfg['checkpoint'], 'train_' + fname)
    np.savez(datapath, tt = tt, epoch_loss = epoch_loss, epoch_accU = epoch_accU, epoch_accm = epoch_accm,
             epoch_accv = epoch_accv, epoch_accth = epoch_accth, pred = pred, gt = gt) 
    
    # evaluating model in prediction data set
    if valid:
        # load evaluation dataset
        data_load = scipy.io.loadmat(args.data_file)
        params = data_load.get('params')[0,0]
        tt = np.transpose(data_load.get('tout'), (1,0))
        dt = data_load.get('dt')[0,0]
        Us = np.transpose(data_load.get('u_truth'), (1,0))
        dotUs = np.transpose(data_load.get('dotu'), (1,0))
        noise = np.transpose(data_load.get('noise'), (1,0))
        unres = np.transpose(data_load.get('unres'), (1,0))
        mu = np.transpose(data_load.get('gamma_mean_trace'), (1,0))
        Ru = np.transpose(data_load.get('gamma_cov_trace'), (1,0))
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
        
        npred = 1000 #args.pred_length-args.input_length
        ntraj = 10
        nskip = 100
        traj_set = torch.zeros(args.input_length+npred, ntraj, 8, dtype=torch.double)
        for l in range(ntraj):
            traj_set[:,l, 0] = torch.from_numpy(   Ut[l*nskip:l*nskip+args.input_length + npred])
            traj_set[:,l, 1] = torch.from_numpy( dotU[l*nskip:l*nskip+args.input_length + npred])
            traj_set[:,l, 2] = torch.from_numpy(   mv[l*nskip:l*nskip+args.input_length + npred])
            traj_set[:,l, 3] = torch.from_numpy(   rv[l*nskip:l*nskip+args.input_length + npred])
            traj_set[:,l, 4] = torch.from_numpy(   Fu[l*nskip:l*nskip+args.input_length + npred])
            traj_set[:,l, 5] = torch.from_numpy(   Gk[l*nskip:l*nskip+args.input_length + npred])
            traj_set[:,l, 6] = torch.from_numpy( Ures[l*nskip:l*nskip+args.input_length + npred])
            traj_set[:,l, 7] = torch.from_numpy(   dW[l*nskip:l*nskip+args.input_length + npred])
        del(data_load,Ut,dotU,mv,rv, Fu,Gk, dW,Ures)
        
        
        logger.file.write('\n')
        logger.set_names(['Model eval.', 'total', '        error U', '        error U_unres', '        error v', 
                          '        error theta_m', ' error r', ' error theta_v'])
        valid_pred, valid_err = prediction(traj_set, npred,ntraj, model, params, logger, dev)
        
        datapath = os.path.join(cfg['checkpoint'], 'pred_' + fname)
        np.savez(datapath, tt = tt, pred = valid_pred[:,:,:,0], gt = valid_pred[:,:,:,1], valid_err = valid_err)

    logger.close()
    
def prediction(input_set, npred,ntraj, model, params, logger, dev):
    model_m, model_v = model
    with torch.no_grad():
        model_m.eval()
        model_v.eval()
    dyad_cond = dyad.CondGau(dt=args.dt, params=params, device=dev)
    dyad_u    = dyad.Dyn_u(dt=args.dt, params=params, device=dev)
 
    valid_pred = np.zeros((npred, ntraj,6, 2))
    valid_err  = np.zeros((npred, ntraj,6))

    ind1 = [0, 2, 6]
    ind2 = [0, 3, 5]
    indt = [0, 2,3, 4,5, 6]
    istate_m = input_set[:args.input_length,:,ind1].clone().to(dev)
    istate_v = input_set[:args.input_length,:,ind2].clone().to(dev)
    istate   = input_set[:args.input_length,:,:].clone().to(dev)
        
    hidden_m, hidden_v = (), ()
    with torch.no_grad():
        for istep in range(npred):
            # target set data
            target  = input_set[(istep+1): args.input_length + (istep+1), :, indt]

            # run model in one forward iteration
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
            istate[:-1,:,:] = istate[1:,:,:].clone()
            
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
            

            predv   = mv_out.data.cpu().numpy()[-1]
            predrv  = rv_out.data.cpu().numpy()[-1]
            predFu  = Fu_out.data.cpu().numpy()[-1]
            predGk  = Gk_out.data.cpu().numpy()[-1]
            predUr  = Ur_out.data.cpu().numpy()[-1]
            predU   = U_out.data.cpu().numpy()[-1]
            targ    = target.data.cpu().numpy()[-1]
            valid_pred[istep, :,1, 0] = predv
            valid_pred[istep, :,2, 0] = predrv
            valid_pred[istep, :,3, 0] = predFu
            valid_pred[istep, :,4, 0] = predGk
            valid_pred[istep, :,5, 0] = predUr
            valid_pred[istep, :,0, 0] = predU
            valid_pred[istep, :,:, 1] = targ
            
            valid_err[istep, :,1] = ( np.square(predv  - targ[:,1]) )
            valid_err[istep, :,2] = ( np.square(predrv - targ[:,2]) )
            valid_err[istep, :,3] = ( np.square(predFu - targ[:,3]) )
            valid_err[istep, :,4] = ( np.square(predGk - targ[:,4]) )
            valid_err[istep, :,5] = ( np.square(predUr - targ[:,5]) )
            valid_err[istep, :,0] = ( np.square(predU  - targ[:,0]) )

            err_ave = valid_err.mean(1)
            print('step {}: err_U = {:.6f} err_Ur = {:.6f} err_v = {:.6f} err_Fu = {:.6f} error_R = {:.6f} error_Gk = {:.6f}'.format(istep, 
                  err_ave[istep,0], err_ave[istep,5], err_ave[istep,1], err_ave[istep,3], err_ave[istep,2], err_ave[istep,4] ) )
            logger.append([istep, err_ave[istep,:].sum(), err_ave[istep,0],err_ave[istep,5], err_ave[istep,1], 
                           err_ave[istep,3], err_ave[istep,2], err_ave[istep,4] ])
        
    return valid_pred, valid_err
    
def train(train_loader, model, criterion, optimizer, params, dev):
    model_m, model_v = model
    optim_m, optim_v = optimizer
    model_m.train()
    model_v.train()
    dyad_cond = dyad.CondGau(dt=args.dt, params=params, device=dev)
    dyad_u    = dyad.Dyn_u(dt=args.dt, params=params, device=dev)
    
    batch_time = AverageMeter()
    losses     = AverageMeter()
    accsU      = AverageMeter()
    accsm      = AverageMeter()
    accsv      = AverageMeter()
    accsFu     = AverageMeter()
    accsGk     = AverageMeter()
    accsUr     = AverageMeter()
    end = time.time()
    
    input_full, target_full = train_loader
    dsize = args.train_batch*args.iters
    s_idx = random.sample(range(0,input_full.size(1)), dsize)
    input_iter   = input_full[:, s_idx,:].pin_memory()
    target_iter  = target_full[:,s_idx,:].pin_memory()
    for ib in range(0, args.iters):
        ind1 = [0, 2, 6]
        ind2 = [0, 3, 5]
        inputs   = input_iter[:, ib*args.train_batch:(ib+1)*args.train_batch, :].to(dev, non_blocking=True)
        inputs_m = input_iter[:, ib*args.train_batch:(ib+1)*args.train_batch, ind1].to(dev, non_blocking=True)
        inputs_v = input_iter[:, ib*args.train_batch:(ib+1)*args.train_batch, ind2].to(dev, non_blocking=True)
        indt = [0, 2,3, 4,5, 6]
        targets = target_iter[:, ib*args.train_batch:(ib+1)*args.train_batch, indt].to(dev, non_blocking=True)
        
        optim_m.zero_grad()
        optim_v.zero_grad()  # zero the gradient buffers
        # iteration the model in npred steps
        hidden_m, hidden_v = (), ()
        istate   = torch.empty(args.input_length, args.train_batch, 8, dtype=torch.double, device=dev)
        istate_m = torch.empty(args.input_length, args.train_batch, 3, dtype=torch.double, device=dev)
        istate_v = torch.empty(args.input_length, args.train_batch, 3, dtype=torch.double, device=dev)
        istate_m[:,:,:] = inputs_m[:args.input_length,:,:]
        istate_v[:,:,:] = inputs_v[:args.input_length,:,:]
        istate[:,:,:]   = inputs[:args.input_length,:,:]
        
        pred = torch.empty(args.input_length+args.npred, args.train_batch, 7, dtype=torch.double, device=dev)
        pred[:args.input_length,:,:] = inputs[:args.input_length,:,0:-1].clone()
        loss = 0
        for ip in range(args.npred):
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
        

            pred[args.input_length+ip,:, 0] = U_out[-1]
            pred[args.input_length+ip,:, 1] = dotU_out[-1]
            pred[args.input_length+ip,:, 2] = mv_out[-1]
            pred[args.input_length+ip,:, 3] = rv_out[-1]
            pred[args.input_length+ip,:, 4] = Fu_out[-1]
            pred[args.input_length+ip,:, 5] = Gk_out[-1]
            pred[args.input_length+ip,:, 6] = Ur_out[-1]
            
            if ip < args.npred-1:
                istate_m = torch.empty_like(istate_m)
                istate_v = torch.empty_like(istate_v)
                istate   = torch.empty_like(istate)
                # update with full model output
                istate[:,:,7] = inputs[ip+1:args.input_length+ip+1,:,7]
                
                istate_m[:,:,0] = U_out
                istate_m[:,:,1] = mv_out
                istate_m[:,:,2] = Ur_out
                istate_v[:,:,0] = U_out
                istate_v[:,:,1] = rv_out
                istate_v[:,:,2] = Gk_out
                istate[:,:,0]   = U_out
                istate[:,:,1]   = dotU_out
                istate[:,:,2]   = mv_out
                istate[:,:,3]   = rv_out
                istate[:,:,4]   = Fu_out
                istate[:,:,5]   = Gk_out
                istate[:,:,6]   = Ur_out

            output = torch.transpose(torch.cat([U_out[:,:,None], mv_out[:,:,None], rv_out[:,:,None],
                                                Gk_out[:,:,None], Ur_out[:,:,None]],2), 0,1)
            target = torch.transpose(targets[ip:args.input_length+ip,:,[0,1,2,4,5]], 0,1)
            if args.loss_type == 'state':
                out = output[:, -args.nloss:, :]
                tag = target[:, -args.nloss:, :]
                loss += 1*criterion(out, tag)
            elif args.loss_type == 'kld':
                ind = [0,1,3,4]
                out1 = output[:, -args.nloss:, ind]
                tag1 = target[:, -args.nloss:, ind]
                out1_p = F.log_softmax(1. * out1, dim=1)
                tag1_p = F.softmax(1. * tag1, dim=1)
                out1_n = F.log_softmax(-1. * out1, dim=1)
                tag1_n = F.softmax(-1. * tag1, dim=1)
                out2 = output[:, -args.nloss:, 2]
                tag2 = target[:, -args.nloss:, 2]
                out2_p = F.log_softmax(1. * out2, dim=1)
                tag2_p = F.softmax(1. * tag2, dim=1)
                loss += criterion(out1_p,tag1_p)+1.*criterion(out1_n,tag1_n) + \
                        1.*criterion(out2_p,tag2_p)
            elif args.loss_type == 'mixed':
                ind = [0,1,3,4]
                crion1, crion2 = criterion
                out1 = output[:, -args.nloss:, ind]
                tag1 = target[:, -args.nloss:, ind]
                out1_p = F.log_softmax(1. * out1, dim=1)
                tag1_p = F.softmax(1. * tag1, dim=1)
                out1_n = F.log_softmax(-1. * out1, dim=1)
                tag1_n = F.softmax(-1. * tag1, dim=1)
                out2 = output[:, -args.nloss:, 2]
                tag2 = target[:, -args.nloss:, 2]
                out2_p = F.log_softmax(1. * out2, dim=1)
                tag2_p = F.softmax(1. * tag2, dim=1)
                
                out = output[:, -args.nloss:, :]
                tag = target[:, -args.nloss:, :]
                loss += crion1(out1_p,tag1_p)+1.*crion1(out1_n,tag1_n) + \
                        crion1(out2_p,tag2_p) + 1.*crion2(out,tag)

                
        
        loss.backward()
        optim_m.step()
        optim_v.step()
        
        # get trained output
        losses.update(loss.item() )
        pred_out = pred[args.input_length:]
        gt_out   = targets[args.input_length-1:]
        accsm.update(  ((pred_out[:,:,2]-gt_out[:,:,1]).square().mean() ).item() )
        accsv.update(  ((pred_out[:,:,3]-gt_out[:,:,2]).square().mean() ).item() )
        accsFu.update( ((pred_out[:,:,4]-gt_out[:,:,3]).square().mean() ).item() )
        accsGk.update( ((pred_out[:,:,5]-gt_out[:,:,4]).square().mean() ).item() ) 
        accsUr.update( ((pred_out[:,:,6]-gt_out[:,:,5]).square().mean() ).item() )
        accsU.update(  ((pred_out[:,:,0]-gt_out[:,:,0]).square().mean() ).item() )
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        #LR = optimizer.param_groups[0]['lr']
        suffix = 'Iter{iter}: Loss = {loss:.5e} U = {accU:.5e} Ur = {accUr:.5e} v = {accu:.5e} Fu = {accthm:.5e} r = {accr:.5e} Gk = {accthv:.5e}  run_time = {bt:.2f}'.format(
                  iter = ib, loss = losses.val, accU=accsU.val, accUr=accsUr.val, accu=accsm.val,accthm=accsFu.val, 
                  accr=accsv.val, accthv=accsGk.val, bt = batch_time.sum)
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
