"""
Forward Euler integrator for the dyad model with conditional Gaussian dynamics
"""
import numpy as np
import torch

__all__ = ['Dyn_u', 'CondGau']

class Dyn_u(object):
    "compute the one-step forward integration for the resolved component u of dyad model"
    def __init__(self, dt, params, device='cpu'):
        self.device = device
        
        self.dt = dt
        self.fu = params['F_u'][0,0].astype(float)
        self.fv = params['F_v'][0,0].astype(float)
        self.du = params['d_u'][0,0].astype(float)
        self.dv = params['d_v'][0,0].astype(float)
        self.sig_u = params['sig_u'][0,0].astype(float)
        self.sig_v = params['sig_v'][0,0].astype(float)
        self.c = params['c'][0,0].astype(float)
        
    def baro_mean_flow(self, U, Ures):
        
        incre = -self.du*U + Ures + self.fu
        return incre

    def update_u_euler(self, inputs):
        U = inputs[:,:,0]
        # dotU = inputs[:,:,1]
        # mv = inputs[:,:,2]
        # rv = inputs[:,:,3]
        # Fu = inputs[:,:,4]
        # Gk = inputs[:,:,5]
        Ures = inputs[:,:,6]
        dW   = inputs[:,:,7]
        
        incre_U = self.baro_mean_flow(U,Ures)
        Up = U + self.dt * incre_U + dW
        Udot = incre_U + dW / self.dt
        return Up, Udot
    
class CondGau(object):
    "compute the one-step forward integration for the conditional Gaussian model"
    # NOTICE here is only for uniform damping & noise
    def __init__(self, dt, params, device='cpu'):
        self.device = device
        
        self.dt = dt
        self.fu = params['F_u'][0,0].astype(float)
        self.fv = params['F_v'][0,0].astype(float)
        self.du = params['d_u'][0,0].astype(float)
        self.dv = params['d_v'][0,0].astype(float)
        self.sig_u = params['sig_u'][0,0].astype(float)
        self.sig_v = params['sig_v'][0,0].astype(float)
        self.c = params['c'][0,0].astype(float)
        
    def cond_mean_v(self, U,mv, dotU,Gk):
        
        FU = dotU[:,:] - (-self.du*U+self.fu+self.c*U*mv) 
        incre = -self.c*U**2 + self.fv - self.dv*mv  \
                + (self.sig_u)**(-2)*FU*Gk

        return incre, FU

    def cond_var_v(self, rv, Gk):
        
        incre = -2*self.dv*rv + (self.sig_v)**2 - (self.sig_u)**(-2)*(Gk**2)
        return incre

    def update_condGau_euler(self, inputs):
        U = inputs[:,:,0]
        dotU = inputs[:,:,1]
        mv = inputs[:,:,2]
        rv = inputs[:,:,3]
        # Fu = inputs[:,:,4]
        Gk = inputs[:,:,5]
        
        incre_v, FU = self.cond_mean_v(U,mv, dotU,Gk)
        mvp = mv + self.dt * incre_v
        rvp = rv + self.dt * self.cond_var_v(rv, Gk)
        
        return mvp, rvp, FU