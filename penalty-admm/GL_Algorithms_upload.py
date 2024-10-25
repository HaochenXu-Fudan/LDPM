#%%
# from msilib.schema import Class
from re import M
import time # record the efficiency
import itertools
import numpy as np
import pandas as pd
import cvxpy as cp
from hyperopt import tpe, hp, fmin  # for Bayesian method

from HC_GL import GL_Hillclimb

from tqdm import tqdm

from utils import Monitor, Monitor_DC

#%%
def train_error(settings, data, x):
    return .5 / settings.num_train * np.sum(np.square( data.y_train - data.X_train @ x ))

def validation_error(settings, data, x):
    return .5 / settings.num_validate * np.sum(np.square( data.y_validate - data.X_validate @ x ))

def test_error(settings, data, x):
    return .5 / settings.num_test * np.sum(np.square( data.y_test - data.X_test @ x ))

class Training_model:
    def __init__(self, data_info) -> None:  
        settings = data_info.settings
        data = data_info.data
        n, p, M = settings.num_train, settings.num_features, settings.num_experiment_groups
        cal_group_sizes = [p//M] * M
        group_ind = np.concatenate( [[0], np.cumsum(cal_group_sizes)] )

        self.x = cp.Variable(p)
        self.lam = cp.Parameter(M+1, nonneg=True)
        LS_lower = .5/n * cp.sum_squares( data.y_train - data.X_train @ self.x )
        group_lasso_regularization = cp.sum([self.lam[i]*cp.pnorm(self.x[group_ind[i]:group_ind[i+1]], 2) for i in range(M)])
        sparsity_regularization = cp.pnorm(self.x, 1)
        self.training_problem = cp.Problem(cp.Minimize(LS_lower + group_lasso_regularization + self.lam[-1]*sparsity_regularization))

    def solve_training(self, lam):
        self.lam.value = lam
        self.training_problem.solve()
        return self.x.value

class Training_model_simple:
    def __init__(self, data_info) -> None:  
        settings = data_info.settings
        data = data_info.data
        n, p, M = settings.num_train, settings.num_features, settings.num_experiment_groups
        cal_group_sizes = [p//M] * M
        group_ind = np.concatenate( [[0], np.cumsum(cal_group_sizes)] )

        self.x = cp.Variable(p)
        self.lam = cp.Parameter(2, nonneg=True)
        LS_lower = .5/n * cp.sum_squares( data.y_train - data.X_train @ self.x )
        group_lasso_regularization = cp.sum([cp.pnorm(self.x[group_ind[i]:group_ind[i+1]], 2) for i in range(M)])
        sparsity_regularization = cp.pnorm(self.x, 1)
        self.training_problem = cp.Problem(cp.Minimize(LS_lower + self.lam[0]*group_lasso_regularization + self.lam[-1]*sparsity_regularization))

    def solve_training(self, lam):
        self.lam.value = lam
        self.training_problem.solve()
        return self.x.value

#%%
# MMCauchy_penalty
def MMCauchy_penalty(data_info,MMP_Setting = dict()):
    # these settings are similar to those in iP_DCA
    settings = data_info.settings
    data = data_info.data
    p, M = settings.num_features, settings.num_experiment_groups
    cal_group_sizes = [p//M] * M
    group_ind = np.concatenate( [[0], np.cumsum(cal_group_sizes)] )

    MAX_ITERATION = MMP_Setting["MAX_ITERATION"] if "MAX_ITERATION" in MMP_Setting.keys() else 100
    TOL = MMP_Setting["TOL"] if "TOL" in MMP_Setting.keys() else 5e-2
    r = MMP_Setting["initial_guess_r"] if "initial_guess" in MMP_Setting.keys() else .1*np.ones(M+1)
    lbd = MMP_Setting["initial_guess_lbd"] if "initial_guess" in MMP_Setting.keys() else 5*np.ones(M+1)
    x = np.zeros(settings.num_features)
    epsilon = 1*np.ones(1)
    #define problems
    class MMP_approximated:
        def __init__(self, settings, data, MMP_Setting = dict()):
            self.delta =MMP_Setting["delta"] if "delta" in MMP_Setting.keys() else 5.
            self.c_alo = MMP_Setting["c"] if "c" in MMP_Setting.keys() else .1
            gamma = MMP_Setting["gamma"] if "gamma" in MMP_Setting.keys() else 5e-3
            beta_0 = MMP_Setting["beta_0"] if "beta_0" in MMP_Setting.keys() else 1
            
            self.x = cp.Variable(settings.num_features)
            self.r = cp.Variable(M+1)
            self.lbd = cp.Variable(M+1)
            self.rho1 = cp.Variable(settings.num_features)
            self.rho2 = cp.Variable(settings.num_features)
            self.w = cp.Variable(settings.num_train)
            self.coff_lbd = cp.Parameter(M+1,pos=True)
            self.coff_r = cp.Parameter(M+1,pos=True)
            #self.epsilon = cp.Parameter(1,nonneg=True)
            self.beta_k = cp.Parameter(pos=True)
            self.beta_k.value = beta_0
            # beta here is different from that in VF-DCA
            violate_1 = cp.maximum( *([cp.pnorm(self.x[group_ind[i]:group_ind[i+1]], 2) - self.r[i] for i in range(M)] + [cp.pnorm(self.x, 1) - self.r[M]]))
            violate_2 = cp.maximum( *([cp.pnorm(self.rho1[group_ind[i]:group_ind[i+1]], 2) - self.lbd[i] for i in range(M)] + [cp.pnorm(self.rho2, np.inf) - self.lbd[M]]))
            violate_3 = cp.norm2(data.X_train.T @ self.w + self.rho1 + self.rho2)
            bias_train = data.X_train @ self.x - data.y_train
            violate_4 =cp.maximum(0,cp.sum_squares(bias_train)+cp.sum_squares(cp.multiply(self.coff_r,self.r))+cp.sum_squares(cp.multiply(self.coff_lbd,self.lbd))+cp.sum_squares(self.w+data.y_train)-cp.sum_squares(data.y_train)-epsilon)
            self.violate = cp.maximum(0,violate_1,violate_2,violate_3,violate_4)
            loss_upper = .5/settings.num_validate*self.beta_k*cp.sum_squares(data.y_validate-data.X_validate @ self.x)+100*self.violate
            self.mmp_approximated = cp.Problem(cp.Minimize(loss_upper))

        def solve(self):
            ECOS_TOL = 1e-3
            ECOS_ITER = 100
            result = self.mmp_approximated.solve(solver = cp.ECOS, abstol=ECOS_TOL, reltol=ECOS_TOL, abstol_inacc=ECOS_TOL, reltol_inacc=ECOS_TOL,max_iters=ECOS_ITER)
            #mosek_params = {'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':8.0e-4}
            #result = self.mm_approximated.solve(solver = cp.MOSEK, mosek_params=mosek_params)
            return result, self.x.value, self.lbd.value, self.r.value
        
        def clare_variable_k(self, lbd, r, epsilon):
            lbd0 = np.maximum(lbd,1e-6)
            r0 = np.maximum(r,1e-6)
            temp = np.sqrt(lbd0/r0)
            self.coff_r.value = temp
            temp = np.sqrt(r0/lbd0)
            self.coff_lbd.value = temp
            #self.epsilon.value = epsilon
        def update_beta(self, err):
            if err / self.beta_k.value <= self.c_alo * min( 1., 1/(self.beta_k.value*self.violate.value) ):
                self.beta_k.value = 1/(1/self.beta_k.value + self.delta)
        def cal_penalty(self):
            return self.violate.value

    def iteration_err(x, lbd, r, xp, lbdp, rp):
        return np.sqrt(
            np.sum(np.square(x - xp)) + np.sum(np.square(r - rp))+np.sum(np.square(lbd - lbdp))
        ) / np.sqrt(
            np.sum(np.square(x)) + np.sum(np.square(r))+np.sum(np.square(lbd))
        )
    

    # preparation
    Timer = time.time
    monitor_mmp = Monitor()
    time_start = Timer()

    # main part
    approximated_problem = MMP_approximated(settings, data, MMP_Setting)

    for k in range(MAX_ITERATION):
        #epsilon = epsilon/2
        approximated_problem.clare_variable_k(lbd, r , epsilon)
        _, x_p, lbd_p, r_p = approximated_problem.solve()
        #r_p = np.maximum(r_p, 0)
        
        time_past = Timer() - time_start

        err = iteration_err(x, lbd, r, x_p, lbd_p, r_p)
        penalty = approximated_problem.cal_penalty()

        dic_for_monitor = {
            "time": time_past, 
            "train_error": train_error(settings, data, x_p),
            "validation_error": validation_error(settings, data, x_p),
            "test_error": test_error(settings, data, x_p),
        }

        monitor_mmp.append(dic_for_monitor)

        # Stopping Test
        if err < TOL and penalty < TOL:
            break 
        
        #approximated_problem.update_beta(err)

        x, lbd, r = x_p, lbd_p, r_p 

    return monitor_mmp.to_df()      
# MMCauchy
def MMCauchy(data_info,MM_Setting = dict()):
    # these settings are similar to those in iP_DCA
    settings = data_info.settings
    data = data_info.data
    p, M = settings.num_features, settings.num_experiment_groups
    cal_group_sizes = [p//M] * M
    group_ind = np.concatenate( [[0], np.cumsum(cal_group_sizes)] )

    MAX_ITERATION = MM_Setting["MAX_ITERATION"] if "MAX_ITERATION" in MM_Setting.keys() else 10
    TOL = MM_Setting["TOL"] if "TOL" in MM_Setting.keys() else 5e-2
    r = MM_Setting["initial_guess_r"] if "initial_guess" in MM_Setting.keys() else .1*np.ones(M)
    lbd = MM_Setting["initial_guess_lbd"] if "initial_guess" in MM_Setting.keys() else 5*np.ones(M)
    x = np.zeros(settings.num_features)
    epsilon = 1e-3*np.ones(1)
    #define problems
    class MM_approximated:
        def __init__(self, settings, data, MM_Setting = dict()):
            
            self.x = cp.Variable(settings.num_features)
            self.r = cp.Variable(M)
            self.lbd = cp.Variable(M)
            self.rho1 = cp.Variable(settings.num_features)
            # self.rho2 = cp.Variable(settings.num_features)
            self.w = cp.Variable(settings.num_train)
            self.coff_lbd = cp.Parameter(M,pos=True)
            self.coff_r = cp.Parameter(M,pos=True)
            self.epsilon = cp.Parameter(1,nonneg=True)

            loss_upper = 1/2*cp.sum_squares(data.y_validate-data.X_validate @ self.x)
            self.constraints = [cp.pnorm(self.x[group_ind[i]:group_ind[i+1]], 2) <= self.r[i] for i in range(M)] 
            self.constraints += [cp.pnorm(self.rho1[group_ind[i]:group_ind[i+1]], 2) <= self.lbd[i] for i in range(M)] 
            self.constraints +=[data.X_train.T @ self.w + self.rho1  == 0]
            bias_train = data.X_train @ self.x - data.y_train
            self.constraints +=[cp.sum_squares(bias_train)+cp.sum_squares(cp.multiply(self.coff_r,self.r))+cp.sum_squares(cp.multiply(self.coff_lbd,self.lbd))+cp.sum_squares(self.w+data.y_train)<=cp.sum_squares(data.y_train)+self.epsilon]

            self.mm_approximated = cp.Problem(cp.Minimize(loss_upper), self.constraints)

        def solve(self):
            ECOS_TOL = 1e-2
            ECOS_ITER = 50
            # result = self.mm_approximated.solve(solver = cp.ECOS, abstol=ECOS_TOL, reltol=ECOS_TOL, abstol_inacc=ECOS_TOL, reltol_inacc=ECOS_TOL,max_iters=ECOS_ITER)
            result = self.mm_approximated.solve(solver=cp.SCS)
            #mosek_params = {'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':8.0e-4}
            #result = self.mm_approximated.solve(solver = cp.MOSEK, mosek_params=mosek_params)
            return result, self.x.value, self.lbd.value, self.r.value
        
        def clare_variable_k(self, lbd, r, epsilon):
            lbd0 = np.maximum(lbd,1e-6)
            r0 = np.maximum(r,1e-6)
            temp = np.sqrt(lbd0/r0)
            self.coff_r.value = temp
            temp = np.sqrt(r0/lbd0)
            self.coff_lbd.value = temp
            self.epsilon.value = epsilon

    def iteration_err(x, lbd, r, xp, lbdp, rp):
        return np.sqrt(
            np.sum(np.square(x - xp)) + np.sum(np.square(r - rp))+np.sum(np.square(lbd - lbdp))
        ) / np.sqrt(
            np.sum(np.square(x)) + np.sum(np.square(r))+np.sum(np.square(lbd))
        )

    # preparation
    Timer = time.time
    monitor_mm = Monitor()
    time_start = Timer()

    # main part
    approximated_problem = MM_approximated(settings, data, MM_Setting)

    for k in range(MAX_ITERATION):
        #epsilon = epsilon/2
        approximated_problem.clare_variable_k(lbd, r , epsilon)
        _, x_p, lbd_p, r_p = approximated_problem.solve()
        #r_p = np.maximum(r_p, 0)
        
        time_past = Timer() - time_start

        err = iteration_err(x, lbd, r, x_p, lbd_p, r_p)

        dic_for_monitor = {
            "time": time_past, 
            "train_error": train_error(settings, data, x_p),
            "validation_error": validation_error(settings, data, x_p),
            "test_error": test_error(settings, data, x_p),
        }

        monitor_mm.append(dic_for_monitor)

        # Stopping Test
        if err < TOL:
            break 
        
        #approximated_problem.update_beta(err)

        x, lbd, r = x_p, lbd_p, r_p 
        # print(lbd, r)

    return monitor_mm.to_df()  




def LDPM(data_info,PM_Setting = dict()):
    settings = data_info.settings
    data = data_info.data
    TOL = PM_Setting["TOL"]
    
    class gl():
        def __init__(self,A_val,b_val,A_tr,b_tr,group_number):
            self.A_val=A_val
            self.b_val=b_val
            self.A_tr=A_tr
            self.b_tr=b_tr
            self.group_number=group_number
            m=A_tr.shape[0]
            n=A_tr.shape[1]
            self.group_size=n//group_number
            self.m=m
            self.n=n
            self.pinv=np.linalg.pinv(2*np.eye(n)+A_tr.T@A_tr)
            self.parameters=np.random.randn(n*3+m+2*(group_number+1))
            self.x=self.parameters[:n]
            self.rho1=self.parameters[n:2*n]
            self.rho2=self.parameters[2*n:3*n]
            self.w=self.parameters[3*n:3*n+m]
            self.r=self.parameters[3*n+m:3*n+m+group_number+1]
            self.lam=self.parameters[3*n+m+group_number+1:3*n+m+group_number+1+group_number+1]
            self.rp,self.lamp=0.1*np.ones(group_number+1),5*np.ones(group_number+1)
            self.xp=np.zeros(n)
            self.r[:],self.lam[:]=0.1*np.ones(group_number+1),5*np.ones(group_number+1)
            self.x[:]=np.zeros(n)
            self.beta=PM_Setting['beta0']
            self.prox=1

        def proj1(self):
            def alt(P,x0,tol=1e-6):
                x = x0.copy()
                p = len(P)
                y = np.zeros((p,x0.shape[0]))
                n = 0
                cI = float('inf')
                while n < 100 and cI >= tol:
                    cI = 0
                    for i in range(0,p):
                        prev_x = x.copy()
                        x = P[i](prev_x - y[i,:])
                        prev_y = y[i,:].copy()
                        y[i,:] = x - (prev_x - prev_y)
                        cI += np.linalg.norm(prev_y - y[i,:])**2
                        n += 1
                return x
            
            def p2(x0,r0):
                if r0==0:
                    return np.zeros_like(x0),0.
                if np.sum(x0**2)<=r0**2:
                    if r0>=0: return x0,r0
                    else: return np.zeros_like(x0),0.
                t=np.sqrt(np.sum(x0**2)/r0**2)
                if r0>0:
                    lam=(t-1)/(t+1)
                    return x0/(lam+1),r0/(1-lam)
                if r0<0:
                    lam=(t+1)/(t-1)
                    return x0/(lam+1),r0/(1-lam)
                        
            def p1(v,s):
                y=np.abs(v)
                if np.sum(y)<=s:
                    x=v
                    t=s
                    return (x,t)
                a,b=-s,1
                while y.size>0:
                    lam=np.sum(y[0])
                    y_upper = y[np.where(y>lam)]
                    sum_y = np.sum(y_upper) 
                    sum_index = y_upper.size
                    g = a + sum_y -lam*(b+sum_index)
                    if g<0:
                        a = a + sum_y +lam
                        b = b + sum_index+1
                        y = y[np.where(y<lam)]
                    else:
                        if g>0:
                            y = y_upper
                        else:
                            break

                lam = a/b
                x = np.where(np.zeros_like(v)> (v - lam),np.zeros_like(v), v - lam) - np.where(np.zeros_like(v)>( -v - lam),np.zeros_like(v), -v - lam)
                t = lam + s
                return x,t
            
            def p2f(x0):
                xx,rr=x0[:self.n].copy(),x0[self.n:].copy()
                for i in range(self.group_number):
                    xx[i*self.group_size:(i+1)*self.group_size],rr[i:i+1]=p2(xx[i*self.group_size:(i+1)*self.group_size].copy(),rr[i:i+1].copy())
                return np.concatenate([xx,rr])
            
            def p1f(x0):
                xx,rr=x0[:self.n].copy(),x0[self.n:].copy()
                xx[:],rr[-1:]=p1(xx.copy(),rr[-1:].copy())
                return np.concatenate([xx,rr])
            
            x0=np.concatenate([self.x,self.r])
            x=alt([p2f,p1f],x0)
            self.x[:],self.r[:]=x[:self.n].copy(),x[self.n:].copy()

        def proj2(self):
            def alt(P,x0,tol=1e-6):
                x = x0.copy()
                p = len(P)
                y = np.zeros((p,x0.shape[0]))
                n = 0
                cI = float('inf')
                while n < 100 and cI >= tol:
                    cI = 0
                    for i in range(0,p):
                        prev_x = x.copy()
                        x = P[i](prev_x - y[i,:])
                        prev_y = y[i,:].copy()
                        y[i,:] = x - (prev_x - prev_y)
                        cI += np.linalg.norm(prev_y - y[i,:])**2
                        n += 1
                return x
            
            def p2(x0,r0):
                if r0==0:
                    return np.zeros_like(x0),0.
                if np.sum(x0**2)<=r0**2:
                    if r0>=0: return x0,r0
                    else: return np.zeros_like(x0),0.
                t=np.sqrt(np.sum(x0**2)/r0**2)
                if r0>0:
                    lam=(t-1)/(t+1)
                    return x0/(lam+1),r0/(1-lam)
                if r0<0:
                    lam=(t+1)/(t-1)
                    return x0/(lam+1),r0/(1-lam)
                
            def pi(rho0,lam0):
                def end(mu):
                    t=-lam0+mu
                    v=-np.where(0>np.abs(rho0)-mu,0,np.abs(rho0)-mu)*np.sign(rho0)
                    rho,lam=rho0+v,lam0+t
                    return [rho,lam]               
                if -np.sum(np.abs(rho0))-lam0>=0:
                    return end(0.)
                else:
                    rho=np.abs(rho0)
                    upper=np.min(rho)
                    lower=0
                    rho_upper=rho[np.where(rho>=upper)]
                    upper=np.min(rho_upper)
                    while rho_upper.size>0:
                        upper=np.min(rho_upper)
                        mu=(np.sum(np.abs(rho_upper))+lam0)/(rho_upper.size+1)
                        if np.sum(mu)>=lower and np.sum(mu)<=upper:
                            return end(np.sum(mu))
                        else:
                            lower=upper
                            rho_upper=rho_upper[np.where(rho_upper>upper)]
                    return end(np.sum(lam0))
                

            def p2f(x0):
                rho1,lam=x0[:self.n].copy(),x0[-(self.group_number+1):].copy()
                for i in range(self.group_number):
                    rho1[i*self.group_size:(i+1)*self.group_size],lam[i:i+1]=p2(rho1[i*self.group_size:(i+1)*self.group_size].copy(),lam[i:i+1].copy())
                x0[:self.n]=rho1[:].copy()
                x0[-(self.group_number+1):]=lam[:].copy()
                return x0.copy()
            
            def pif(x0):
                rho2,lam=x0[self.n:2*self.n].copy(),x0[-(self.group_number+1):].copy()
                x0[self.n:2*self.n],x0[-1:]=pi(rho2.copy(),lam[-1:].copy())
                return x0.copy()
            
            def pn(x):
                w,rho1,rho2=x[2*self.n:self.n*2+self.m].copy(),x[:self.n].copy(),x[self.n:2*self.n].copy()
                t=self.pinv@(self.A_tr.T@w+rho1+rho2)
                x[2*self.n:2*self.n+self.m],x[:self.n],x[self.n:self.n*2]=w-self.A_tr@t,rho1-t,rho2-t
                return x
            
            x0=np.concatenate([self.rho1,self.rho2,self.w,self.lam])
            x=alt([p2f,pif,pn],x0)
            self.rho1[:],self.rho2[:],self.w[:],self.lam[:]=x[:self.n],x[self.n:self.n*2],x[2*self.n:2*self.n+self.m],x[-(self.group_number+1):]

            


    problem = gl(data.X_validate,data.y_validate,data.X_train,data.y_train,settings.num_experiment_groups)
    Timer = time.time
    monitor_mm = Monitor()
    time_start = Timer()
    gd_step = PM_Setting['gd_step']
    max_inner_iter=5
    max_outer_iter=10

    
    for j in range(max_outer_iter):
        parameters_pp,parameters_p=problem.parameters.copy(),problem.parameters.copy()
        for i in range(max_inner_iter):
            k=i+1
            problem.parameters[:]=parameters_p+(k-2)/(k+1)*(parameters_p-parameters_pp)
            grad=np.concatenate((problem.A_val.T@(problem.A_val@problem.x-problem.b_val)+problem.beta*problem.A_tr.T@(problem.A_tr@problem.x-problem.b_tr),
                                np.zeros_like(problem.rho1),
                                np.zeros_like(problem.rho1),
                                problem.beta*(problem.w+problem.b_tr),
                                problem.beta*(problem.lam+problem.prox*(problem.r-problem.rp)),
                                problem.beta*(problem.r+problem.prox*(problem.lam-problem.lamp)),
                                ))

            parameters_pp=parameters_p.copy()
            problem.parameters[:]-=gd_step*grad
            problem.proj1()
            problem.proj2()
            parameters_p=problem.parameters.copy()
            
        problem.parameters[:]=parameters_p.copy()

   
        def iteration_err(x, lbd, r, xp, lbdp, rp):
            return np.sqrt(
                np.sum(np.square(x - xp)) + np.sum(np.square(r - rp))+np.sum(np.square(lbd - lbdp))
            ) / np.sqrt(
                np.sum(np.square(x)) + np.sum(np.square(r))+np.sum(np.square(lbd))
            )

        err = iteration_err(problem.x, problem.lam, problem.r, problem.xp, problem.lamp, problem.rp)


        problem.rp=problem.r.copy()
        problem.lamp=problem.lam.copy()
        problem.xp=problem.x.copy()

        time_past = Timer() - time_start
 
        dic_for_monitor = {
            "time": time_past, 
            "train_error": train_error(settings, data, problem.x),
            "validation_error": validation_error(settings, data, problem.x),
            "test_error": test_error(settings, data, problem.x),
        }

        monitor_mm.append(dic_for_monitor)

        if err < TOL:
            break

    return monitor_mm.to_df()  





# iP-DCA
def iP_DCA(data_info, DC_Setting = dict()):
    settings = data_info.settings
    data = data_info.data
    p, M = settings.num_features, settings.num_experiment_groups
    cal_group_sizes = [p//M] * M
    group_ind = np.concatenate( [[0], np.cumsum(cal_group_sizes)] )

    MAX_ITERATION = DC_Setting["MAX_ITERATION"] if "MAX_ITERATION" in DC_Setting.keys() else 100
    TOL = DC_Setting["TOL"] if "TOL" in DC_Setting.keys() else 5e-2
    r = DC_Setting["initial_guess"] if "initial_guess" in DC_Setting.keys() else .1*np.ones(M)
    x = np.zeros(settings.num_features)

    # subproblems define
    class DC_lower:
        def __init__(self, settings, data) -> None:
            self.x_lower = cp.Variable(settings.num_features)
            self.r_lower = cp.Parameter(M, nonneg=True)
            self.constraints_lower = [cp.pnorm(self.x_lower[group_ind[i]:group_ind[i+1]], 2) <= self.r_lower[i] for i in range(M)] 
            LS_lower = .5 / settings.num_train * cp.sum_squares( data.y_train - data.X_train @ self.x_lower )
            self.dc_lower = cp.Problem(cp.Minimize(LS_lower), self.constraints_lower)

        def solve(self, r):
            ECOS_TOL = 1e-4
            ECOS_ITER = 100
            self.r_lower.value = r
            result = self.dc_lower.solve(solver = cp.ECOS, abstol=ECOS_TOL, reltol=ECOS_TOL, abstol_inacc=ECOS_TOL, reltol_inacc=ECOS_TOL, max_iters=ECOS_ITER)
            return result, self.x_lower.value

        def dual_value(self):
            return np.array([float(self.constraints_lower[i].dual_value) for i in range(M)])

    class DC_approximated:
        def __init__(self, settings, data, DC_Setting = dict()) -> None:
            self.delta = DC_Setting["delta"] if "delta" in DC_Setting.keys() else 5.
            self.c_alo = DC_Setting["c"] if "c" in DC_Setting.keys() else .1
            epsilon_alo = DC_Setting["epsilon"] if "epsilon" in DC_Setting.keys() else 0
            rho = DC_Setting["rho"] if "rho" in DC_Setting.keys() else 5e-3
            beta_0 = DC_Setting["beta_0"] if "beta_0" in DC_Setting.keys() else 1

            self.x_upper, self.r = cp.Variable(settings.num_features), cp.Variable(M)
            self.x_upper_k, self.r_k = cp.Parameter(settings.num_features), cp.Parameter(M, pos=True)
            self.gamma_k, self.bias_k = cp.Parameter(M), cp.Parameter()
            self.beta_k = cp.Parameter(pos=True)
            self.beta_k.value = beta_0

            LS_upper = .5/settings.num_validate * cp.sum_squares( data.y_validate - data.X_validate @ self.x_upper )
            prox = cp.sum_squares(self.x_upper - self.x_upper_k) + cp.sum_squares(self.r - self.r_k) 
            beta_k_V_k = self.beta_k * .5 / settings.num_train * cp.sum_squares( data.y_train - data.X_train @ self.x_upper ) + self.gamma_k @ self.r - self.bias_k - self.beta_k * epsilon_alo
            # violation = 0
            violation = cp.maximum( *([cp.pnorm(self.x_upper[group_ind[i]:group_ind[i+1]], 2) - self.r[i] for i in range(M)] ) )
            self.beta_k_penalty = cp.maximum(0, beta_k_V_k, 100*self.beta_k*violation)
            phi_k = LS_upper + rho/2 * prox + self.beta_k_penalty
            bi_constraints = [self.r >= 0]

            self.dc_approximated = cp.Problem(cp.Minimize(phi_k), bi_constraints)
        
        def solve(self, k):
            # try:
            # ECOS_TOL = 20/(k+1)
            # ECOS_ITER = 100
            # result = self.dc_approximated.solve(solver = cp.ECOS, feastol=ECOS_TOL,abstol=ECOS_TOL, reltol=ECOS_TOL, max_iters=ECOS_ITER, verbose=False)
            result = self.dc_approximated.solve(solver = cp.ECOS, verbose=False)
            # except:
                # result = self.dc_approximated.solve(solver=cp.SCS)
            return result, self.x_upper.value, self.r.value
        
        def update_beta(self, err):
            if err * self.beta_k.value <= self.c_alo * min( 1., self.beta_k_penalty.value ):
                self.beta_k.value = self.beta_k.value + self.delta
        
        def clare_V_k(self, gamma, obj_lower):
            self.gamma_k.value = gamma * self.beta_k.value
            self.bias_k.value = obj_lower * self.beta_k.value + self.gamma_k.value @ self.r_k.value 
        
        def clare_variable_k(self, x, r):
            self.x_upper_k.value = x 
            self.r_k.value = r
        
        def cal_penalty(self):
            return self.beta_k_penalty.value / self.beta_k.value

    def iteration_err(x, r, xp, rp):
        return np.sqrt(
            np.sum(np.square(x - xp)) + np.sum(np.square(r - rp))
        ) / np.sqrt(
            np.sum(np.square(x)) + np.sum(np.square(r))
        )

    # preparation
    Timer = time.time
    monitor_dc = Monitor_DC()
    time_start = Timer()

    # main part
    lower_problem = DC_lower(settings, data)
    approximated_problem = DC_approximated(settings, data, DC_Setting)

    for k in range(MAX_ITERATION):
        approximated_problem.clare_variable_k(x, r)
        obj_lower_k, x_k_tilde = lower_problem.solve(r)
        gamma = lower_problem.dual_value()
        approximated_problem.clare_V_k(gamma, obj_lower_k)
        _, x_p, r_p = approximated_problem.solve(k)
        r_p = np.maximum(r_p, 0)
        
        time_past = Timer() - time_start

        err = iteration_err(x, r, x_p, r_p)
        penalty = approximated_problem.cal_penalty()

        dic_for_monitor = {
            "time": time_past, 
            "train_error": train_error(settings, data, x_p),
            "validation_error": validation_error(settings, data, x_p),
            "test_error": test_error(settings, data, x_p),
            "lower_train_error": train_error(settings, data, x_k_tilde),
            "lower_validation_error": validation_error(settings, data, x_k_tilde), 
            "lower_test_error": test_error(settings, data, x_k_tilde),
            "diff_xk_xtilde": np.linalg.norm(x - x_k_tilde),
            "diff_xkp_xtilde": np.linalg.norm(x_p - x_k_tilde),
            "beta": approximated_problem.beta_k.value,
            "penalty": penalty
        }

        monitor_dc.append(dic_for_monitor)

        # Stopping Test
        if err < TOL and penalty < TOL:
            break 
        
        approximated_problem.update_beta(err)

        x, r = x_p, r_p 

    return monitor_dc.to_df()
#%%
# BCD_iP_DCA
def BCD_iP_DCA(data_info, DC_Setting = dict()):
    settings = data_info.settings
    data = data_info.data
    p, M = settings.num_features, settings.num_experiment_groups
    cal_group_sizes = [p//M] * M
    group_ind = np.concatenate( [[0], np.cumsum(cal_group_sizes)] )

    MAX_ITERATION = DC_Setting["MAX_ITERATION"] if "MAX_ITERATION" in DC_Setting.keys() else 100
    TOL = DC_Setting["TOL"] if "TOL" in DC_Setting.keys() else 5e-2
    r = DC_Setting["initial_guess"] if "initial_guess" in DC_Setting.keys() else .1*np.ones(M+1)
    x = np.zeros(settings.num_features)

    # subproblems define
    class DC_lower:
        def __init__(self, settings, data) -> None:
            self.x_lower = cp.Variable(settings.num_features)
            self.r_lower = cp.Parameter(M+1, nonneg=True)
            self.constraints_lower = [cp.pnorm(self.x_lower[group_ind[i]:group_ind[i+1]], 2) <= self.r_lower[i] for i in range(M)] + [cp.pnorm(self.x_lower, 1) <= self.r_lower[M]]
            LS_lower = .5 / settings.num_train * cp.sum_squares( data.y_train - data.X_train @ self.x_lower )
            self.dc_lower = cp.Problem(cp.Minimize(LS_lower), self.constraints_lower)

        def solve(self, r):
            ECOS_TOL = 1e-4
            ECOS_ITER = 100
            self.r_lower.value = r
            result = self.dc_lower.solve(solver = cp.ECOS, abstol=ECOS_TOL, reltol=ECOS_TOL, abstol_inacc=ECOS_TOL, reltol_inacc=ECOS_TOL, max_iters=ECOS_ITER)
            return result, self.x_lower.value

        def dual_value(self):
            return np.array([float(self.constraints_lower[i].dual_value) for i in range(M+1)])

    class DC_approximated:
        def __init__(self, settings, data, DC_Setting = dict()) -> None:
            self.delta = DC_Setting["delta"] if "delta" in DC_Setting.keys() else 5.
            self.c_alo = DC_Setting["c"] if "c" in DC_Setting.keys() else .1
            epsilon_alo = DC_Setting["epsilon"] if "epsilon" in DC_Setting.keys() else 0
            rho = DC_Setting["rho"] if "rho" in DC_Setting.keys() else 5e-3
            beta_0 = DC_Setting["beta_0"] if "beta_0" in DC_Setting.keys() else 1

            self.x_upper, self.r = cp.Variable(settings.num_features), cp.Variable(M+1)
            self.x_upper_k, self.r_k = cp.Parameter(settings.num_features), cp.Parameter(M+1, pos=True)
            self.gamma_k, self.bias_k = cp.Parameter(M+1), cp.Parameter()
            self.beta_k = cp.Parameter(pos=True)
            self.beta_k.value = beta_0
            self.constant_1 = cp.Parameter()
            self.constant_2 = cp.Parameter()
            self.vector_2 = cp.Parameter(M+1)
            self.beta_x_k = cp.Parameter(settings.num_features)
            self.beta_r_k = cp.Parameter(M+1)

            LS_upper = .5/settings.num_validate * cp.sum_squares( data.y_validate - data.X_validate @ self.x_upper )
            prox_x = cp.sum_squares(self.x_upper - self.x_upper_k)
            prox_r = cp.sum_squares(self.r - self.r_k)

            # beta_k_V_k_1 = self.beta_k * .5 / settings.num_train * cp.sum_squares( data.y_train - data.X_train @ self.x_upper ) + self.gamma_k @ self.r_k - self.bias_k - self.beta_k * epsilon_alo
            beta_k_V_k_1 = self.beta_k * .5 / settings.num_train * cp.sum_squares( data.y_train - data.X_train @ self.x_upper ) + self.constant_1
            # violation = 0
            violation_1 = cp.maximum( *([self.beta_k*cp.pnorm(self.x_upper[group_ind[i]:group_ind[i+1]], 2) - self.beta_r_k[i] for i in range(M)] + [self.beta_k*cp.pnorm(self.x_upper, 1) - self.beta_r_k[M]]) )
            self.beta_k_penalty_1 = cp.maximum(0, beta_k_V_k_1, violation_1)
            beta_k_V_k_2 = self.constant_2 + self.gamma_k @ self.r 
            violation_2 = cp.maximum( *([self.vector_2[i] - self.beta_k*self.r[i] for i in range(M)] + [self.vector_2[M] - self.beta_k*self.r[M]]) )
            self.beta_k_penalty_2 = cp.maximum(0, beta_k_V_k_2, violation_2)

            phi_k_1 = LS_upper + rho/2 * prox_x + self.beta_k_penalty_1
            phi_k_2 = rho/2 * prox_r + self.beta_k_penalty_2
            bi_constraints = [self.r >= 0]

            self.dc_approximated_1 = cp.Problem(cp.Minimize(phi_k_1))
            self.dc_approximated_2 = cp.Problem(cp.Minimize(phi_k_2), bi_constraints)
        
        def solve(self):
            self.constant_1.value = self.gamma_k.value @ self.r_k.value - self.bias_k.value
            self.beta_r_k.value = self.beta_k.value * self.r_k.value
            result = self.dc_approximated_1.solve(solver = cp.ECOS, verbose=False)
            
            self.x_upper_k.value = self.x_upper.value 
            self.constant_2.value = (self.beta_k * .5 / settings.num_train * cp.sum_squares( data.y_train - data.X_train @ self.x_upper_k ) - self.bias_k).value
            self.vector_2.value = self.beta_k.value * np.concatenate([np.array([cp.pnorm(self.x_upper_k[group_ind[i]:group_ind[i+1]], 2).value for i in range(M)]), np.array([cp.pnorm(self.x_upper, 1).value])])
            result = self.dc_approximated_2.solve(solver = cp.ECOS, verbose=False)

            return result, self.x_upper.value, self.r.value
        
        def update_beta(self, err):
            if err * self.beta_k.value <= self.c_alo * min( 1., self.beta_k_penalty_2.value ):
                self.beta_k.value = self.beta_k.value + self.delta
        
        def clare_V_k(self, gamma, obj_lower):
            self.gamma_k.value = gamma * self.beta_k.value
            self.bias_k.value = obj_lower * self.beta_k.value + self.gamma_k.value @ self.r_k.value 
        
        def clare_variable_k(self, x, r):
            self.x_upper_k.value = x 
            self.r_k.value = r
        
        def cal_penalty(self):
            return self.beta_k_penalty_2.value / self.beta_k.value

    def iteration_err(x, r, xp, rp):
        return np.sqrt(
            np.sum(np.square(x - xp)) + np.sum(np.square(r - rp))
        ) / np.sqrt(
            np.sum(np.square(x)) + np.sum(np.square(r))
        )

    # preparation
    Timer = time.time
    monitor_dc = Monitor_DC()
    time_start = Timer()

    # main part
    lower_problem = DC_lower(settings, data)
    approximated_problem = DC_approximated(settings, data, DC_Setting)

    for k in range(MAX_ITERATION):
        approximated_problem.clare_variable_k(x, r)
        obj_lower_k, x_k_tilde = lower_problem.solve(r)
        gamma = lower_problem.dual_value()
        approximated_problem.clare_V_k(gamma, obj_lower_k)
        _, x_p, r_p = approximated_problem.solve()
        r_p = np.maximum(r_p, 0)
        
        time_past = Timer() - time_start

        err = iteration_err(x, r, x_p, r_p)
        penalty = approximated_problem.cal_penalty()

        dic_for_monitor = {
            "time": time_past, 
            "train_error": train_error(settings, data, x_p),
            "validation_error": validation_error(settings, data, x_p),
            "test_error": test_error(settings, data, x_p),
            "beta": approximated_problem.beta_k.value,
            "penalty": penalty
        }

        monitor_dc.append(dic_for_monitor)

        # Stopping Test
        if err < TOL and penalty < TOL:
            break 
        
        approximated_problem.update_beta(err)

        x, r = x_p, r_p 

    return monitor_dc.to_df()

#%%
# Grid Search
def Grid_Search(data_info):
    # preparation
    settings = data_info.settings
    data = data_info.data
    training_process = Training_model_simple(data_info)

    Timer = time.time
    monitor = Monitor()
    time_start = Timer()

    # main part
    lam1s = np.power(10, np.linspace(-3, 1, 10)) 
    lam2s = lam1s
    for lam1, lam2 in tqdm(itertools.product(lam1s, lam2s)):
        x = training_process.solve_training(np.array([lam1, lam2]))
        monitor.append({
            "time": Timer() - time_start,
            "train_error": train_error(settings, data, x), 
            "validation_error": validation_error(settings, data, x), 
            "test_error": test_error(settings, data, x)
        })

    return monitor.to_df()

# Random Search
def Random_Search(data_info):
    # preparation
    settings = data_info.settings
    data = data_info.data
    training_process = Training_model(data_info)

    Timer = time.time
    Random_Generator = np.random.rand
    monitor = Monitor()
    time_start = Timer()

    # main part
    N = 100
    for _ in tqdm(range(N)):
        # lam1, lam2 = np.power(10, -3+4*Random_Generator()), np.power(10, -3+4*Random_Generator())
        x = training_process.solve_training(np.power(10, -3+4*Random_Generator(settings.num_experiment_groups+1)))
        monitor.append({
            "time": Timer() - time_start,
            "train_error": train_error(settings, data, x), 
            "validation_error": validation_error(settings, data, x), 
            "test_error": test_error(settings, data, x)
        })

    return monitor.to_df()

# Bayesian Method
def Bayesian_Method(data_info, Debug = False):
    # define the object for bayesian method 
    settings = data_info.settings
    data = data_info.data
    M = settings.num_experiment_groups
    training_process = Training_model(data_info)

    def Bayesian_obj(param):
        nonlocal monitor
        x = training_process.solve_training(np.power(10, np.array(param)))
        val_err = validation_error(settings, data, x)
        monitor.append({
            "time": Timer() - time_start,
            "train_error": train_error(settings, data, x), 
            "validation_error": val_err, 
            "test_error": test_error(settings, data, x)
        })
        return val_err

    # preparation
    Timer = time.time
    monitor = Monitor()

    # main part
    space = [hp.uniform(str(i), -3, 1) for i in range(M+1)]
    time_start = Timer()
    Best = fmin(
        fn=Bayesian_obj, # Objective Function to optimize
        space=space, # Hyperparameter's Search Space
        algo=tpe.suggest, # Optimization algorithm (representative TPE)
        max_evals=100 # Number of optimization attempts
        )
    
    if Debug: print(Best['1'], Best[str(M)])
    
    return monitor.to_df()  

# Implicit Differentiation
def Imlicit_Differntiation(data_info, HC_Setting = dict()):
    data = data_info.data 
    settings = data_info.settings
    initial_guess = HC_Setting.pop("initial_guess") if "initial_guess" in HC_Setting.keys() else .05*np.ones(settings.num_experiment_groups)
    HC_algo = GL_Hillclimb(data, settings, HC_Setting)
    HC_algo.run([initial_guess], debug=False, log_file=None)
    return HC_algo.monitor.to_df()

# GAFFA
def GF(data_info, GF_Setting = dict()):
    settings = data_info.settings
    data = data_info.data
    p, M = settings.num_features, settings.num_experiment_groups
    cal_group_sizes = [p//M] * M
    # group_ind = np.concatenate( [[0], np.cumsum(cal_group_sizes)] )

    # MAX_ITERATION = DC_Setting["MAX_ITERATION"] if "MAX_ITERATION" in DC_Setting.keys() else 100
    # TOL = DC_Setting["TOL"] if "TOL" in DC_Setting.keys() else 5e-2
    # r = DC_Setting["initial_guess"] if "initial_guess" in DC_Setting.keys() else .1*np.ones(M+1)
    x = np.zeros(settings.num_features)





    nIter = 100
    p = 0.3
    gam1, gam2 = 10, 1.

    gam1, gam2 = 100, .1
    ck0, lamU = 10, .5
    etak, alphak, betak = 1e-1, 1e-2, 1e-2

    def ls_err(X, y, w):
        return np.sum(np.square(X @ w - y))/len(y)
    
    class Training_model0:
        def __init__(self, data, settings) -> None:  
            n, p, M = settings.num_train, settings.num_features, settings.num_experiment_groups
            cal_group_sizes = [p//M] * M
            group_ind = np.concatenate( [[0], np.cumsum(cal_group_sizes)] )

            self.x = cp.Variable(p)
            self.lam = cp.Parameter(M+1, nonneg=True)
            LS_lower = .5/n * cp.sum_squares( data.y_train - data.X_train @ self.x )
            group_lasso_peanlty = [ self.lam[i] * cp.pnorm(self.x[group_ind[i]:group_ind[i+1]], 2) for i in range(M)]
            sparsity_penalty = self.lam[M] * cp.pnorm(self.x, 1)
            self.penalty = [cp.pnorm(self.x[group_ind[i]:group_ind[i+1]], 2) for i in range(M)] + [cp.pnorm(self.x, 1)]
            self.training_problem = cp.Problem(cp.Minimize(LS_lower + sparsity_penalty + sum(group_lasso_peanlty)))

        def solve_training(self, lam):
            self.lam.value = lam
            self.training_problem.solve(cp.ECOS)
            return self.x.value

        def p_value(self):
            return np.array([float(self.penalty[i].value) for i in range(len(self.penalty))])


    class Training_model1:
        def __init__(self, data, settings) -> None:  
            n, p, M = settings.num_train, settings.num_features, settings.num_experiment_groups
            cal_group_sizes = [p//M] * M
            group_ind = np.concatenate( [[0], np.cumsum(cal_group_sizes)] )

            self.x = cp.Variable(p)
            self.r = cp.Parameter(M+1, nonneg=True)
            LS_lower = .5/n * cp.sum_squares( data.y_train - data.X_train @ self.x )
            group_lasso_constraint = [cp.pnorm(self.x[group_ind[i]:group_ind[i+1]], 2) <= self.r[i] for i in range(M)] 
            sparsity_constraint = cp.pnorm(self.x, 1) <= self.r[M]
            self.constraint = group_lasso_constraint + [sparsity_constraint]
            self.training_problem = cp.Problem(cp.Minimize(LS_lower), self.constraint)

        def solve_training(self, r):
            self.r.value = r
            self.training_problem.solve(solver=cp.ECOS)
            return self.x.value

        def solve_training2(self, r):
            r1 = r
            r1[:-1] = np.sqrt(r1[:-1])
            self.r.value = r1
            self.training_problem.solve(solver=cp.ECOS)
            return self.x.value


    group_size = cal_group_sizes
    group_ind = np.concatenate([[0], np.cumsum(group_size)])

    F = lambda w: .5 * ls_err(data.X_validate, data.y_validate, w) 
    dwF = lambda w: data.X_validate.T @ (data.X_validate @ w - data.y_validate) / len(data.y_validate)

    f = lambda w: .5 * ls_err(data.X_train, data.y_train, w)
    dwf = lambda w: data.X_train.T @ (data.X_train @ w - data.y_train) / len(data.y_train)

    g = lambda r, w: np.concatenate([
    [np.sum(np.square(w[group_ind[i]:group_ind[i+1]])) - r[i] for i in range(M)], 
    [(np.linalg.norm(w, 1) - r[M])]
    ])
    dwlamg = lambda r, w, lam: np.concatenate([2 * w[group_ind[i]:group_ind[i+1]] * lam[i] for i in range(M)]) + np.sign(w) * lam[M]
    drlamg = lambda w, r, lam: - lam

    lower_solver = Training_model1(data,settings)
    
    lower0  = Training_model0(data, settings)
    w0 = lower0.solve_training(.1*np.ones(M+1))
    r0 = lower0.p_value()
    r0[:-1] = np.square(r0[:-1])
    w1 = np.copy(w0)
    lam0 = 0.1 * np.ones(M + 1) 
    lam1 = 0.1 * np.ones(M + 1) 

    monitor = Monitor() 

    T = 0 
    time0 = time.time() 

    r1 = r0
    r1[:-1] = [r**2 for r in r1[:-1]]
    w_hat = lower_solver.solve_training2(r0)
    T += time.time() - time0
    monitor.append({"k": 0, "time": T, 
                    "validation_error": F(w0), "train_error": f(w0), 
                    "test_error": .5 * ls_err(data.X_test, data.y_test, w0), 
                    "dy": np.linalg.norm(w0 - w_hat),
                    "g": np.linalg.norm(np.maximum(g(r0, w0), 0)), 
                    })
    

    for k in range(nIter):
        time0 = time.time()
        # ck = np.min([ck0 * (k + 1) ** p, 50])
        ck = ck0 * (k + 1) ** p

        for _ in range(1):
            dw1 = dwf(w0) + dwlamg(r0, w0, lam0) + 1 / gam1 * (w1 - w0)
            w1p = w1 - etak * dw1

            lam0 = lam1 + gam2 * g(r0, w0)
            lam0 = np.maximum(0, lam0)
        
        dr = drlamg(r0, w0, lam0) - drlamg(r0, w1p, lam1)
        r0 = r0 - alphak * dr

        dw0 = 1 / ck * dwF(w0) + dwf(w0) + dwlamg(r0, w0, lam0) - (w0 - w1p) / gam1
        w0 = w0 - alphak * dw0

        dlam1 = - (lam1 - lam0) / gam2 - g(r0, w1p)
        lam1 = lam1 - betak * dlam1
        lam1 = np.minimum(np.maximum(0, lam1), lamU)

        w1 = w1p

        T += time.time() - time0         
        
        monitor.append({"k": k+1, "time": T, 
                    "validation_error": F(w0), "train_error": f(w0), 
                    "test_error": .5 * ls_err(data.X_test, data.y_test, w0), 
                    "dy": np.linalg.norm(w0 - w_hat),
                    "g": np.linalg.norm(np.maximum(g(r0, w0), 0)), 
                    })
        

    return monitor.to_df()