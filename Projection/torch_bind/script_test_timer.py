# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 10:21:37 2023

Michel Barlaud, Guillaume Perez, and Jean-Paul Marmorat.

Linear time bi-level l1,infini projection ; application to feature selection and
sparsification of auto-encoders neural networks.

http://arxiv.org/abs/2407.16293, 2024


l1inftyB_cpp.proj_l1infty, is the C++ code provided by Chu et al ICML 2020

"""

#%%
#import os


import timeit
import numpy as np
import torch
import projections as l1inftyB_cpp 




#%%



def proj_l2ball(w0, eta, device="cpu"):
    w = torch.as_tensor(w0, dtype=torch.get_default_dtype(), device=device)

    n = torch.linalg.norm(w, ord=2)
    #n = torch.norm(w,p=2)
    if n <= eta:
        return w
    #else:
    return torch.mul(eta / n, w)


def proj_l21ball(w2, eta, device="cpu"):

    w = torch.as_tensor(w2, dtype=torch.get_default_dtype(), device=device)

    if w.dim() == 1:
        Q = proj_l1ball(w, eta, device=device)
    else:

        init_shape = w.shape
        Res = torch.empty(init_shape)
        nrow, ncol = init_shape[0:2]

        #W = torch.tensor(
        #    [torch.sum(torch.abs(w[:, i])).data.item() for i in range(ncol)]
        #)
        
        W = torch.tensor(
            [torch.sum(torch.square(w[:, i])).data.item() for i in range(ncol)]
        )

        PW = proj_l1ball(W, eta, device=device)

        for i in range(ncol):
            Res[:, i] = proj_l2ball(w[:, i], PW[i].data.item(), device=device)

        Q = Res.clone().detach().requires_grad_(True)

    if not torch.is_tensor(w2):
        Q = Q.data.numpy()
    return Q



def proj_l11ball(w2, eta, device="cpu"):

    w = torch.as_tensor(w2, dtype=torch.get_default_dtype(), device=device)

    if w.dim() == 1:
        Q = proj_l1ball(w, eta, device=device)
    else:

        init_shape = w.shape
        Res = torch.empty(init_shape)
        nrow, ncol = init_shape[0:2]

        W = torch.tensor(
            [torch.sum(torch.abs(w[:, i])).data.item() for i in range(ncol)]
        )

        PW = proj_l1ball(W, eta, device=device)

        for i in range(ncol):
            Res[:, i] = proj_l1ball(w[:, i], PW[i].data.item(), device=device)

        Q = Res.clone().detach().requires_grad_(True)

    if not torch.is_tensor(w2):
        Q = Q.data.numpy()
    return Q


def proj_l1ball(w0, eta, device="cpu"):
    # To help you understand, this function will perform as follow:
    #    a1 = torch.cumsum(torch.sort(torch.abs(y),dim = 0,descending=True)[0],dim=0)
    #    a2 = (a1 - eta)/(torch.arange(start=1,end=y.shape[0]+1))
    #    a3 = torch.abs(y)- torch.max(torch.cat((a2,torch.tensor([0.0]))))
    #    a4 = torch.max(a3,torch.zeros_like(y))
    #    a5 = a4*torch.sign(y)
    #    return a5

    w = torch.as_tensor(w0, dtype=torch.get_default_dtype(), device=device)

    init_shape = w.size()

    if w.dim() > 1:
        init_shape = w.size()
        w = w.reshape(-1)

    Res = torch.sign(w) * torch.max(
        torch.abs(w)
        - torch.max(
            torch.cat(
                (
                    (
                        torch.cumsum(
                            torch.sort(torch.abs(w), dim=0, descending=True)[0],
                            dim=0,
                            dtype=torch.get_default_dtype(),
                        )
                        - eta
                    )
                    / torch.arange(
                        start=1,
                        end=w.numel() + 1,
                        device=device,
                        dtype=torch.get_default_dtype(),
                    ),
                    torch.tensor([0.0], dtype=torch.get_default_dtype(), device=device),
                )
            )
        ),
        torch.zeros_like(w),
    )

    Q = Res.reshape(init_shape).clone().detach()

    if not torch.is_tensor(w0):
        Q = Q.data.numpy()
    return Q
   
def bilevel_proj_l1Inftyball(w2, RHO, device="cpu"):

    w = torch.as_tensor(w2, dtype=torch.get_default_dtype(), device=device)
    
    if w.dim() == 1:
        Q = proj_l1ball(w, RHO, device=device)
    else:
    
        init_shape = w.shape
        Res = torch.empty(init_shape)
        nrow, ncol = init_shape[0:2]
    
        W = torch.tensor(
            [torch.max(torch.abs(w[:, i])).data.item() for i in range(ncol)]
        )
    
        PW = proj_l1ball(W, RHO, device=device)
    
        for i in range(ncol):
            Res[:, i] = torch.clamp(torch.abs(w[:, i]), max=PW[i].data.item())
            Res[:, i] = Res[:, i] * torch.sign(w[:, i])
    
        Q = Res.clone().detach().requires_grad_(True)
    
    if not torch.is_tensor(w2):
        Q = Q.data.numpy()
    return Q


def bilevel_proj_l1Inftyball_unbounded(w2, RHO, device="cpu"):

    w = torch.as_tensor(w2, dtype=torch.get_default_dtype(), device=device)
    
    if w.dim() == 1:
        Q = proj_l1ball(w, RHO, device=device)
    else:
    
        init_shape = w.shape
        Res = torch.empty(init_shape)
        nrow, ncol = init_shape[0:2]
    
        W = torch.tensor(
            [torch.max(torch.abs(w[:, i])).data.item() for i in range(ncol)]
        )
    
        PW = proj_l1ball(W, RHO, device=device)
    
        for i in range(ncol):
            Res[:, i] =max(PW[i].data.item())
            Res[:, i] = Res[:, i] * torch.sign(w[:, i])
    
        Q = Res.clone().detach().requires_grad_(True)
    
    if not torch.is_tensor(w2):
        Q = Q.data.numpy()
    return Q





def f1(w):
    return torch.max(torch.abs(w)).data.item()
def f2(w, PW):
    return torch.clamp(torch.abs(w), max=PW.data.item()) * torch.sign(w)


def proj_l1Inftyball_line(w2, C, device="cpu"):

    tps3 = time.perf_counter() 
    w = torch.as_tensor(w2, dtype=torch.get_default_dtype(), device=device)
    # w = torch.as_tensor(w2, device=device)

    if w.dim() == 1:
        Q = proj_l1ball(w, C, device=device)
    else:

        init_shape = w.shape
        Res = torch.empty(init_shape)
        nrow, ncol = init_shape[0:2]

        X = torch.abs(w)
        X = torch.sort(X, 0, True).values
        S = torch.cumsum(X, 0)

        k = [0 for _ in range(ncol)]
        a = [j for j in range(ncol)]
        theta_num = sum([X[0, i] for i in range(ncol)])
        theta_den = ncol
        theta = (theta_num - C) / theta_den
        changed = True
        while changed:
            for j in a:
                i = k[j]
                while i < nrow - 1 and (S[i, j] - theta) / (i + 1) < X[i + 1, j]:
                    i += 1
                theta_num -= S[k[j], j] / (k[j] + 1)
                theta_den -= 1.0 / (k[j] + 1)
                k[j] = i
                if i == nrow - 1 and S[i, j] < theta:
                    a.remove(j)
                    continue
                theta_num += S[k[j], j] / (k[j] + 1)
                theta_den += 1.0 / (k[j] + 1)
            theta_prime = (theta_num - C) / theta_den
            changed = theta_prime != theta
            theta = theta_prime

        Q = w.clone()
        for j in range(ncol):
            if S[-1, j] < theta:
                Q[:, j] = 0
            else:
                mu = (S[k[j], j] - theta) / (k[j] + 1)
                Q[:, j] = torch.min(mu, abs(Q[:, j]))
        Q = Q * torch.sign(w)
        Q = Q.clone().detach().requires_grad_(True)

        print("Theta = " + str(theta))
    if not torch.is_tensor(w2):
        Q = Q.data.numpy()
    tps4 = time.perf_counter()
    print("Executtion time projection  : ",tps4 - tps3)
    return Q



 

def proj_l1inf_numpy(Y, c, tol=1e-5, direction="row"):
    """
    {X : sum_n max_m |X(n,m)| <= c}
    for some given c>0

        Author: Laurent Condat
        Version: 1.0, Sept. 1, 2017
    
    This algorithm is new, to the author's knowledge. It is based
    on the same ideas as for projection onto the l1 ball, see
    L. Condat, "Fast projection onto the simplex and the l1 ball",
    Mathematical Programming, vol. 158, no. 1, pp. 575-585, 2016. 
    
    The algorithm is exact and terminates in finite time*. Its
    average complexity, for Y of size N x M, is O(NM.log(M)). 
    Its worst case complexity, never found in practice, is
    O(NM.log(M) + N^2.M).

    Note : This is a numpy transcription of the original MATLAB code
    *Due to floating point errors, the actual implementation of the algorithm
    uses a tolerance parameter to guarantee halting of the program
    """
    added_dimension = False

    if direction == "col":
        Y = np.transpose(Y)

    if Y.ndim == 1:
        # for vectors
        Y = np.expand_dims(Y, axis=0)
        added_dimension = True

    X = np.flip(np.sort(np.abs(Y), axis=1), axis=1)
    v = np.sum(X[:, 0])
    if v <= c:
        # inside the ball
        X = Y
    else:
        N, M = Y.shape
        S = np.cumsum(X, axis=1)
        idx = np.ones((N, 1), dtype=int)
        theta = (v - c) / N
        mu = np.zeros((N, 1))
        active = np.ones((N, 1))
        theta_old = 0
        while np.abs(theta_old - theta) > tol:
            for n in range(N):
                if active[n]:
                    j = idx[n]
                    while (j < M) and ((S[n, j - 1] - theta) / j) < X[n, j]:
                        j += 1
                    idx[n] = j
                    mu[n] = S[n, j - 1] / j
                    if j == M and (mu[n] - (theta / j)) <= 0:
                        active[n] = 0
                        mu[n] = 0
            theta_old = theta
            theta = (np.sum(mu) - c) / (np.sum(active / idx))
        X = np.minimum(np.abs(Y), (mu - theta / idx) * active)
        X = X * np.sign(Y)

    if added_dimension:
        X = np.squeeze(X)

    if direction == "col":
        X = np.transpose(X)
    return X


def proj_l1infball(w0, eta, AXIS=1, device="cpu", tol=1e-5):
    """See the documentation of proj_l1inf_numpy for details
    Note: Due to 
    1. numpy's C implementation and 
    2. the non-parallelizable nature of the algorithm,
    it is faster to do this projection on the cpu with numpy arrays 
    than on the gpu with torch tensors
    """
    w = w0.detach().cpu().numpy()
    res = proj_l1inf_numpy(w, eta, direction="col" if AXIS else "row", tol=tol)
    Q = torch.as_tensor(res, dtype=torch.get_default_dtype(), device=device)
    return Q
   
if __name__ == "__main__":
    

    ######## Parameters ########
    ETA = 1. # Controls feature selection (projection) (L1, L11, L21)
    RHO =1. # Controls feature selection for l1inf
    TOL = 1e-3  # error margin for the L1inf algorithm and gradient masking
    nb=400
    
    # Set device (GPU or CPU)
    # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(DEVICE)
    
    #Test with Pytorch without C++
    start_time1 = timeit.default_timer() 
    X1=torch.rand(1000,8000).double()    
    X0=bilevel_proj_l1Inftyball(X1, RHO,device="cpu")
    end_time1 = timeit.default_timer()

    # Calculate and print the execution time
    execution_time1 = end_time1 - start_time1
    

    print(f"Execution time without C++ (Matrix 1000 x 8000):={execution_time1} seconds")
    
    
    for method in [l1inftyB_cpp.proj_l1infty_bilevel,
                   l1inftyB_cpp.proj_l1infty,
                   l1inftyB_cpp.proj_l11_bilevel,
                   l1inftyB_cpp.proj_l12_bilevel,
                   ]:
        print()
        print(method.__name__)
        print()
        times = []
        for y in [1000, 2000, 4000, 6000, 8000]:
            m=y
            n=1000
            
            X=torch.rand(n,m).double()
            
            time=0
            for i in range(nb):
                # Record the start time
                start_time = timeit.default_timer() 
              
                
                X0 = method(X, ETA) 
                
                # Record the end time
                end_time = timeit.default_timer()
            
                # Calculate and print the execution time
                execution_time = end_time - start_time
                time+=execution_time
            times.append(time/nb)
            print(f"Execution time: {time/nb} seconds")
            
            
            