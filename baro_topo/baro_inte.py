"""
Forward Euler integrator for the topographic barotropic model with conditional Gaussian dynamics
"""
import numpy as np
import torch

__all__ = ['BaroTopo', 'CondGau']

class BaroTopo(object):
    "compute the one-step forward integration for the topographic barotropic model"
    def __init__(self, dt, params, kred=2, device='cpu'):
        kmax = params['kmax'][0,0]
        self.kmax = kmax
        self.kred = kred
        self.device = device
        
        self.dt = dt
        self.lx  = params['lvec'][0,0].astype(np.double)
        self.d_U = params['d_U'][0,0]
        self.sig_U = params['sig_U'][0,0]
        hkc = params['hk']
        self.hk = torch.empty(2*kred, dtype=torch.double, device=device)
        self.hk[:kred] = torch.from_numpy(hkc[:kred,0].real)
        self.hk[kred:] = torch.from_numpy(hkc[:kred,0].imag)
        
    def baro_mean_flow(self, U,vk, Ures):
        nseq  = U.shape[0]
        nsamp = U.shape[1]
        hkm = (self.hk).repeat(nseq, nsamp, 1)
        
        incre = -self.d_U*U + 2*(hkm*vk).sum(2) + Ures
        return incre
    def update_flow_step(self, inputs,inputs0, dt):
        U = inputs[:,:,0]
        vk = inputs[:,:,2:2+2*self.kred]
        # dotU = inputs[:,:,1]
        # Fu = inputs[:,:,2+8*self.kred]
        Ures = inputs[:,:,3+12*self.kred]
        # dW = inputs[:,:,4+12*self.kred]
        U0 = inputs0[:,:,0]
        
        incre_U = self.baro_mean_flow(U,vk,Ures)
        Up = U0 + dt * incre_U
        # dotU = incre_U + dW/self.dt
        return Up, incre_U #, dotU
    def update_flow_euler(self, inputs):
        U = inputs[:,:,0]
        vk = inputs[:,:,2:2+2*self.kred]
        # dotU = inputs[:,:,1]
        # Fu = inputs[:,:,2+8*self.kred]
        Ures = inputs[:,:,3+12*self.kred]
        dW = inputs[:,:,4+12*self.kred]
        
        incre_U = self.baro_mean_flow(U,vk,Ures)
        Up = U + self.dt * incre_U + dW
        Udot = incre_U + dW / self.dt
        return Up, Udot
    
class CondGau(object):
    "compute the one-step forward integration for the conditional Gaussian model"
    # NOTICE here is only for uniform damping & noise
    def __init__(self, dt, params, kred=2, device='cpu'):
        kmax = params['kmax'][0,0]
        kvec = params['kvec'][:kred,0]
        self.kmax = kmax
        self.kred = kred
        self.kvec = torch.from_numpy(np.concatenate([kvec,kvec]).astype(np.double)).to(device)
        self.device = device
        
        self.dt = dt
        self.beta = params['beta'][0,0].astype(np.double)
        self.lx  = params['lvec'][0,0].astype(np.double)
        self.d_U = params['d_U'][0,0]
        self.sig_U = params['sig_U'][0,0]
        self.d_k = params['d_k'][0,0]
        self.sig_k = params['sig_k'][:kred,0]
        self.alpha = params['alpha'][0,0].astype(np.double)
        
        self.gam_T = params['gamma_T'][:kred,0]
        hkc = params['hk']
        self.hk = torch.empty(2*kred, dtype=torch.double, device=device)
        self.hk[:kred] = torch.from_numpy(hkc[:kred,0].real)
        self.hk[kred:] = torch.from_numpy(hkc[:kred,0].imag)
        
    def cond_mean_v(self, U,vk, FU,Gk):
        nseq  = U.shape[0]
        nsamp = U.shape[1]
        hkm = (self.hk).repeat(nseq, nsamp, 1)
        kvec = (self.kvec).repeat(nseq, nsamp, 1)
        ome_v = self.beta/kvec-kvec*U[:,:,None]
        
        ind = np.concatenate([range(self.kred,2*self.kred),range(0,self.kred)])
        vki = vk[:,:,ind]
        vki[:,:,:self.kred] = -vki[:,:,:self.kred]
        
        incre = -(self.lx**2)*hkm*U[:,:,None] - self.d_k*vk + ome_v*vki  \
                + (self.sig_U)**(-2)*FU[:,:,None]*Gk[:,:,:2*self.kred]
        # vkp = vk + self.dt * incre
        return incre
    def cond_mean_v_full(self, U,vk, dotU,Gk):
        nseq  = U.shape[0]
        nsamp = U.shape[1]
        hkm = (self.hk).repeat(nseq, nsamp, 1)
        kvec = (self.kvec).repeat(nseq, nsamp, 1)
        ome_v = self.beta/kvec-kvec*U[:,:,None]
        
        ind = np.concatenate([range(self.kred,2*self.kred),range(0,self.kred)])
        vki = vk[:,:,ind]
        vki[:,:,:self.kred] = -vki[:,:,:self.kred]
        
        FU = dotU[:,:] - 2*(hkm*vk).sum(2) + self.d_U*U
        incre = -(self.lx**2)*hkm*U[:,:,None] - self.d_k*vk + ome_v*vki  \
                + (self.sig_U)**(-2)*FU[:,:,None]*Gk[:,:,:2*self.kred]
        # vkp = vk + self.dt * incre
        return incre, FU
    def cond_mean_T(self, U,vk,Tk, FU,Gk):
        nseq  = U.shape[0]
        nsamp = U.shape[1]
        omeT = -(self.kvec).repeat(nseq, nsamp, 1)*U[:,:,None]
        gamT = torch.empty(2*self.kred, dtype=torch.double, device=self.device)
        gamT[:self.kred] = torch.from_numpy(self.gam_T)
        gamT[self.kred:] = torch.from_numpy(self.gam_T)
        gamT = (gamT).repeat(nseq, nsamp, 1)
        
        ind = np.concatenate([range(self.kred,2*self.kred),range(0,self.kred)])
        Tki = Tk[:,:,ind]
        Tki[:,:,:self.kred] = -Tki[:,:,:self.kred]
        
        incre = -gamT*Tk + omeT*Tki -self.alpha*vk \
            + (self.sig_U)**(-2)*FU[:,:,None]*Gk[:,:,2*self.kred:]
        # Tkp = Tk + self.dt * incre
        return incre
    def cond_mean_T_full(self, U,vk,Tk, dotU,Gk):
        nseq  = U.shape[0]
        nsamp = U.shape[1]
        hkm = (self.hk).repeat(nseq, nsamp, 1)
        omeT = -(self.kvec).repeat(nseq, nsamp, 1)*U[:,:,None]
        gamT = torch.empty(2*self.kred, dtype=torch.double, device=self.device)
        gamT[:self.kred] = torch.from_numpy(self.gam_T)
        gamT[self.kred:] = torch.from_numpy(self.gam_T)
        gamT = (gamT).repeat(nseq, nsamp, 1)
        
        ind = np.concatenate([range(self.kred,2*self.kred),range(0,self.kred)])
        Tki = Tk[:,:,ind]
        Tki[:,:,:self.kred] = -Tki[:,:,:self.kred]
        
        FU = dotU[:,:] - 2*(hkm*vk).sum(2) + self.d_U*U
        incre = -gamT*Tk + omeT*Tki -self.alpha*vk \
            + (self.sig_U)**(-2)*FU[:,:,None]*Gk[:,:,2*self.kred:]
        # Tkp = Tk + self.dt * incre
        return incre
    def cond_var_v(self, rvk, Gk):
        nseq  = rvk.shape[0]
        nsamp = rvk.shape[1]
        sigkm = torch.from_numpy(self.sig_k).to(self.device)
        sigkm = (sigkm).repeat(nseq, nsamp, 1)
        Gkv2 = (Gk[:,:,:self.kred])**2+(Gk[:,:,self.kred:2*self.kred])**2
        
        incre = -2*self.d_k*rvk + (sigkm)**2 - (self.sig_U)**(-2)*Gkv2
        # rvkp = rvk + self.dt * incre
        return incre
    def cond_var_T(self, rTk,ck, Gk):
        nseq  = rTk.shape[0]
        nsamp = rTk.shape[1]
        gamT = torch.from_numpy(self.gam_T).to(self.device)
        gamT = (gamT).repeat(nseq, nsamp, 1)
        GkT2 = (Gk[:,:,2*self.kred:3*self.kred])**2+(Gk[:,:,3*self.kred:])**2
        
        incre = -2*gamT*rTk - 2*self.alpha*ck[:,:,:self.kred] - (self.sig_U)**(-2)*GkT2
        # rTkp = rTk + self.dt * incre
        return incre
    def cond_cov(self, U,rvk,ck, Gk):
        nseq  = U.shape[0]
        nsamp = U.shape[1]
        ome_c = -self.beta/(self.kvec).repeat(nseq, nsamp, 1)
        gamT = torch.empty(2*self.kred, dtype=torch.double, device=self.device)
        gamT[:self.kred] = torch.from_numpy(self.gam_T)
        gamT[self.kred:] = torch.from_numpy(self.gam_T)
        gamT = (gamT).repeat(nseq, nsamp, 1)
        
        ind = np.concatenate([range(self.kred,2*self.kred),range(0,self.kred)])
        cki = ck[:,:,ind]
        cki[:,:,:self.kred] = -cki[:,:,:self.kred]
        rvkc = torch.zeros(nseq,nsamp, 2*self.kred, dtype=torch.double, device=self.device)
        rvkc[:,:,:self.kred] = rvk
        
        Gkx2r = Gk[:,:,:2*self.kred]*Gk[:,:,2*self.kred:]
        ind = np.concatenate([range(3*self.kred,4*self.kred),range(2*self.kred,3*self.kred)])
        Gkx2i = Gk[:,:,:2*self.kred]*Gk[:,:,ind]
        Gkc2 = torch.empty(nseq,nsamp, 2*self.kred, dtype=torch.double, device=self.device)
        Gkc2[:,:,:self.kred] = Gkx2r[:,:,:self.kred]+Gkx2r[:,:,self.kred:]
        Gkc2[:,:,self.kred:] = Gkx2i[:,:,:self.kred]-Gkx2i[:,:,self.kred:]
        
        incre = -(self.d_k+gamT)*ck + ome_c*cki - self.alpha*rvkc \
                - (self.sig_U)**(-2)*Gkc2
        # ckp = ck + self.dt * incre
        return incre
    
    def update_condGau(self, inputs):
        U = inputs[:,:,0]
        vk = inputs[:,:,1:1+2*self.kred]
        Tk = inputs[:,:,1+2*self.kred:1+4*self.kred]
        rvk = inputs[:,:,1+4*self.kred:1+5*self.kred]
        rTk = inputs[:,:,1+5*self.kred:1+6*self.kred]
        ck = inputs[:,:,1+6*self.kred:1+8*self.kred]
        FU = inputs[:,:,1+8*self.kred]
        Gk = inputs[:,:,2+8*self.kred:2+12*self.kred]
        
        vkp = vk + self.dt * self.cond_mean_v(U,vk, FU,Gk)
        Tkp = Tk + self.dt *self.cond_mean_T(U,vk,Tk, FU,Gk)
        rvp = rvk + self.dt *self.cond_var_v(rvk, Gk)
        rTp = rTk + self.dt *self.cond_var_T(rTk,ck,Gk)
        ckp = ck + self.dt *self.cond_cov(U,rvk,ck, Gk)
        
        nseq  = U.shape[0]
        nsamp = U.shape[1]
        rkp = torch.empty(nseq,nsamp,2*self.kred, dtype=torch.double, device=self.device)
        rkp[:,:,:self.kred] = rvp
        rkp[:,:,self.kred:] = rTp
        
        return vkp, Tkp, rkp, ckp
    
    def update_condGau_step(self, inputs, inputs0, dt):
        U = inputs[:,:,0]
        dotU = inputs[:,:,1]
        vk = inputs[:,:,2:2+2*self.kred]
        Tk = inputs[:,:,2+2*self.kred:2+4*self.kred]
        rvk = inputs[:,:,2+4*self.kred:2+5*self.kred]
        rTk = inputs[:,:,2+5*self.kred:2+6*self.kred]
        ck = inputs[:,:,2+6*self.kred:2+8*self.kred]
        # Fu = inputs[:,:,2+8*self.kmax]
        Gk = inputs[:,:,3+8*self.kred:3+12*self.kred]
        Ures = inputs[:,:,3+12*self.kred]
        
        # initial states
        vk0 = inputs0[:,:,2:2+2*self.kred]
        Tk0 = inputs0[:,:,2+2*self.kred:2+4*self.kred]
        rvk0 = inputs0[:,:,2+4*self.kred:2+5*self.kred]
        rTk0 = inputs0[:,:,2+5*self.kred:2+6*self.kred]
        ck0 = inputs0[:,:,2+6*self.kred:2+8*self.kred]
        
        incre_vk, FU = self.cond_mean_v_full(U,vk, dotU,Gk)
        incre_Tk = self.cond_mean_T_full(U,vk,Tk, dotU,Gk)
        incre_rv = self.cond_var_v(rvk, Gk)
        incre_rT = self.cond_var_T(rTk,ck,Gk)
        incre_ck = self.cond_cov(U,rvk,ck, Gk)
        
        vkp = vk0 +  dt*incre_vk
        Tkp = Tk0 +  dt*incre_Tk
        rvp = rvk0 + dt*incre_rv
        rTp = rTk0 + dt*incre_rT
        ckp = ck0 +  dt*incre_ck
        
        nseq  = inputs.shape[0]
        nsamp = inputs.shape[1]
        ndim = inputs.shape[2]
        incre_rk = torch.empty(nseq,nsamp,2*self.kred, dtype=torch.double, device=self.device)
        incre_rk[:,:,:self.kred] = incre_rv
        incre_rk[:,:,self.kred:] = incre_rT
        
        # rkp = torch.empty(nseq,nsamp,2*self.kred, dtype=torch.double, device=self.device)
        # rkp[:,:,:self.kred] = rvp
        # rkp[:,:,self.kred:] = rTp
        outputs = torch.empty(nseq,nsamp,ndim, dtype=torch.double, device=self.device)
        outputs[:,:,0] = U
        outputs[:,:,1] = dotU
        outputs[:,:,2:2+2*self.kred] = vkp
        outputs[:,:,2+2*self.kred:2+4*self.kred] = Tkp
        outputs[:,:,2+4*self.kred:2+5*self.kred] = rvp
        outputs[:,:,2+5*self.kred:2+6*self.kred] = rTp
        outputs[:,:,2+6*self.kred:2+8*self.kred] = ckp
        outputs[:,:,2+8*self.kred] = FU
        outputs[:,:,3+8*self.kred:3+12*self.kred] = Gk
        outputs[:,:,3+12*self.kred] = Ures
        
        incre = (incre_vk, incre_Tk, incre_rk, incre_ck)
        # outputs = (vkp, Tkp, rkp, ckp)
        
        return outputs, incre #, FU
    def update_condGau_euler(self, inputs):
        U = inputs[:,:,0]
        dotU = inputs[:,:,1]
        vk = inputs[:,:,2:2+2*self.kred]
        Tk = inputs[:,:,2+2*self.kred:2+4*self.kred]
        rvk = inputs[:,:,2+4*self.kred:2+5*self.kred]
        rTk = inputs[:,:,2+5*self.kred:2+6*self.kred]
        ck = inputs[:,:,2+6*self.kred:2+8*self.kred]
        # Fu = inputs[:,:,2+8*self.kred]
        Gk = inputs[:,:,3+8*self.kred:3+12*self.kred]
        
        incre_v, FU = self.cond_mean_v_full(U,vk, dotU,Gk)
        vkp = vk + self.dt * incre_v
        Tkp = Tk + self.dt * self.cond_mean_T_full(U,vk,Tk, dotU,Gk)
        rvp = rvk + self.dt * self.cond_var_v(rvk, Gk)
        rTp = rTk + self.dt * self.cond_var_T(rTk,ck,Gk)
        ckp = ck + self.dt * self.cond_cov(U,rvk,ck, Gk)
        
        nseq  = U.shape[0]
        nsamp = U.shape[1]
        rkp = torch.empty(nseq,nsamp,2*self.kred, dtype=torch.double, device=self.device)
        rkp[:,:,:self.kred] = rvp
        rkp[:,:,self.kred:] = rTp
        
        return vkp, Tkp, rkp, ckp, FU
