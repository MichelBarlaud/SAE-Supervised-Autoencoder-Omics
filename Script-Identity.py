# -*- coding: utf-8 -*-
"""
Created on Fri May 24 08:38:34 2024

@author: USER
"""

import torch
from numpy import linalg as LA
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import scale as scale
# import math
import pathlib
from adjustText import adjust_text

pathlib.Path('Figures').mkdir(exist_ok=True) 

# %%


def ReadData(
    file_name, doScale=True, doLog=True,
):
    try:
        data_pd = pd.read_csv(
            "data/" + str(file_name),
            delimiter=";",
            decimal=",",
            header=0,
            encoding="ISO-8859-1",
            low_memory=False,
        )
    except:
        data_pd = pd.read_csv(
            "datas/" + str(file_name),
            delimiter=";",
            decimal=",",
            header=0,
            encoding="ISO-8859-1",
            low_memory=False,
        )
    X = (data_pd.iloc[1:, 1:].values.astype(float)).T
    Y = data_pd.iloc[0, 1:].values.astype(float).astype(np.int64)
    col = data_pd.columns.to_list()
    if col[0] != "Name":
        col[0] = "Name"
    data_pd.columns = col
    feature_name = data_pd["Name"].values.astype(str)[1:]
    label_name = np.unique(Y)
    patient_name = data_pd.columns[1:]

    # Do standardization
    if doLog:
        X = np.log(abs(X + 1))  # Transformation

    X = X - np.mean(X, axis=0)

    if doScale:
        X = scale(X, axis=0)  # Standardization along rows
    for index, label in enumerate(
        label_name
    ):  # convert string labels to number (0,1,2....)
        Y = np.where(Y == label, index, Y)
    Y = Y.astype(np.int64)
    # if not Y.all():
    #     Y += 1  # 0,1,2,3.... labels -> 1,2,3,4... labels
    return X, Y, feature_name, label_name, patient_name


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
                            torch.sort(torch.abs(w), dim=0,descending=True)[0],
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


def bilevel_proj_l1Inftyball(w2, eta, device="cpu"):

    w = torch.as_tensor(w2, dtype=torch.get_default_dtype(), device=device)

    if w.dim() == 1:
        Q = proj_l1ball(w, eta, device=device)
    else:

        init_shape = w.shape
        Res = torch.empty(init_shape)
        nrow, ncol = init_shape[0:2]

        W = torch.tensor(
            [torch.max(torch.abs(w[:, i])).data.item() for i in range(ncol)]
        )

        PW = proj_l1ball(W, eta, device=device)

        for i in range(ncol):
            Res[:, i] = torch.clamp(torch.abs(w[:, i]), max=PW[i].data.item())
            Res[:, i] = Res[:, i] * torch.sign(w[:, i])

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

def proj_l2ball(w0, eta, device="cpu"):
    w = torch.as_tensor(w0, dtype=torch.get_default_dtype(), device=device)

    n = torch.linalg.norm(w, ord=2)
    if n <= eta:
        return w
    return torch.mul(eta / n, w)

def proj_l12ball(w2, eta, device="cpu"):

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
        
        W = torch.sqrt(torch.tensor(
            [torch.sum(torch.square(w[:, i])).data.item() for i in range(ncol)]
        ))

        PW = proj_l1ball(W, eta, device=device)

        for i in range(ncol):
            Res[:, i] = proj_l2ball(w[:, i], PW[i].data.item(), device=device)

        Q = Res.clone().detach().requires_grad_(True)

    if not torch.is_tensor(w2):
        Q = Q.data.numpy()
    return Q






def make_pchip_graph(x, y, color, npoints=400, typeCurve='normal'):
    pchip = interpolate.PchipInterpolator(x, y)
    xnew = np.linspace(min(x), max(x), num=npoints)
    yfit = pchip(xnew)
    if typeCurve == 'normal':
        plt.plot(xnew, yfit, color=color)
    if typeCurve == 'semilogx':
        plt.semilogx(xnew, yfit, color=color)
    if typeCurve == 'semilogy':
        plt.semilogy(xnew, yfit, color=color)
    if typeCurve == 'loglog':
        plt.loglog(xnew, yfit, color=color)


def pltCurveWithPoints(x, y, projection, ETA, colors, title, xlabel, ylabel, 
                       typeCurve, zoom=False, fontsize=10, save=None):
    projName = []
    fig, ax = plt.subplots()
    # flattened_y = [element for sublist in y for element in sublist]
    # print(y, flattened_y)
    # Trouver le minimum dans le tableau fusionné
    # ymin = min(flattened_y)+0.001
    for i in range(len(projection)):
        projName.append(projection[i].__name__)
        make_pchip_graph(x[i], y[i], color=colors[i], typeCurve=typeCurve)
        if typeCurve == 'normal':
            ax.plot(x[i], y[i], marker='o',
                    linestyle='None', color=colors[i], label='_nolegend_')
        if typeCurve == 'semilogy':
            ax.semilogx(x[i], y[i], marker='o',
                        linestyle='None', color=colors[i], label='_nolegend_')
        if typeCurve == 'semilogx':
            ax.semilogy(x[i], y[i], marker='o',
                        linestyle='None', color=colors[i], label='_nolegend_')
        if typeCurve == 'loglog':
            ax.loglog(x[i], y[i], marker='o',
                      linestyle='None', color=colors[i], label='_nolegend_')

    ts = []
    projName = getNameProj(projName)
    for i in range(len(projection)):
        for x1, y1, eta in zip(x[i], y[i], ETA):
            ts.append(plt.text(x1, y1, f'$\eta: ${eta}'))
            #ax.annotate(f'$\eta: ${eta}', xy=(x1, y1), xytext=(10, 10),
            #            textcoords='offset points', ha='center', horizontalalignment = 'left')
        adjust_text(ts, x=x[i], y=y[i], force_points=0.1)
    # for x1, y1, eta in zip(x[1], y[1], ETA):
    #     ts.append(plt.text(x1, y1, f'$\eta: ${eta}'))
    #     #ax.annotate(f'$\eta: ${eta}', xy=(x1, y1), xytext=(10, -10),
    #     #            textcoords='offset points', ha='center', horizontalalignment = 'left')
    # adjust_text(ts, x=x[1], y=y[1], force_points=0.1)

    ax.legend(projName)  
    ax.grid(True, which="both", ls="--")
    # ax.set_ylim(bottom = ymin)
    ax.set_title(title, y=1.0, pad=30)  # titre général
    ax.set_xlabel(xlabel, fontsize=fontsize)                         # abcisses

    ax.set_ylabel(ylabel, rotation=90, fontsize=fontsize,
                  labelpad=10)                      # ordonnées

    if typeCurve == 'loglog' and zoom:
        x1, x2, y1, y2 = 2.35e-03, 20e-02, 0.9, 0.998  # subregion of the original image

        axins = ax.inset_axes(
            [1.25, 0.25, 0.5, 0.5],
            xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
        for i in range(len(projection)):
            if typeCurve == 'normal':
                axins.plot(x[i], y[i], marker='o')
            if typeCurve == 'semilogy':
                axins.semilogy(x[i], y[i], marker='o')
            if typeCurve == 'semilogx':
                axins.semilogx(x[i], y[i], marker='o')
            if typeCurve == 'loglog':
                axins.loglog(x[i], y[i], marker='o')
            ts = []
            for x1, y1, eta in zip(x[i], y[i], ETA):
                ts.append(plt.text(x1, y1, f'$\eta: ${eta}'))
                # axins.annotate(f'$\eta: ${eta}', xy=(x1, y1), xytext=(
                #     15, 5), textcoords='offset points', ha='center')
                
            adjust_text(ts, x=x[0], y=y[0], force_points=0.1)

        ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.tight_layout()
    if save: 
        figsavename = pathlib.Path('Figures') / save
        plt.savefig(figsavename) 
    else:
        plt.show()

def getNameProj(nameProj):
    for i in range(len(nameProj)):
        if nameProj[i]== 'proj_l12ball':
            nameProj[i] ="Bilevel $\ell_{1,2}$"
        if nameProj[i]== 'proj_l11ball':
            nameProj[i] ="Bilevel $\ell_{1,1}$"
        if nameProj[i]=='bilevel_proj_l1Inftyball':
            nameProj[i] ="Bilevel $\ell_{1,\infty}$"
        if nameProj[i]== 'proj_l1infball':
            nameProj[i] ="Proj $\ell_{1, \infty}$"
            
    return nameProj

def normlinf1(x):
    # ||Y||_inf,1 = max_j (somme des i à N (abs(Y_i,j)))
    X = x.copy()
    abs_tensor = np.abs(X)
    column_sums = np.sum(abs_tensor, axis=0)
    max_column_sum = np.max(column_sums)

    return  np.float64(max_column_sum.item())


def norml1inf(x):
    # ||Y||1,inf =  somme des j à N (max_i(abs(Y_i,j)))
    X = x.copy()
    abs_tensor = np.abs(X)
    max_in_columns = np.max(abs_tensor, axis=0)
    sum_of_max = np.sum(max_in_columns)

    return  np.float64(sum_of_max.item())

def norml12(x):
    # ||Y||1,2 =  somme des j à N (max_i(abs(Y_i,j)))
    X = x.copy()
    L2_in_columns =LA.norm(X,2, axis=0) 
    sum_of_max = np.sum(L2_in_columns)

    return  np.float64(sum_of_max.item())

def norml11(x):
    # ||Y||1,2 =  somme des j à N (max_i(abs(Y_i,j)))
    X = x.copy()
    L1_in_columns =LA.norm(X,1, axis=0) 
    sum_of_max = np.sum(L1_in_columns)

    return  np.float64(sum_of_max.item())


def computeValueNorm2(norm, value):
    if isnorm("L1inf"):
        projection = [bilevel_proj_l1Inftyball,proj_l1infball ]
        return norml1inf(value), projection
    if isnorm("Linf1"):
        projection = [bilevel_proj_l1Inftyball,proj_l1infball ]
        return normlinf1(value), projection
    if isnorm("L12"):
        projection = [proj_l12ball]
        return norml12(value), projection
    if isnorm("L11"):
        projection = [proj_l11ball]
        return norml11(value), projection
    
def computeValueNorm(norm, value):
    if isnorm("L1inf"):
        return norml1inf(value)
    if isnorm("Linf1"):
        return normlinf1(value)
    if isnorm("L12"):
        return norml12(value)
    if isnorm("L11"):
        return norml11(value)


if __name__ == "__main__":
    import sys
      
    FILES = {
        'lung':'LUNG.csv',
        's1000':'Synth_1000f_64inf_1000s.csv',
        's1000-16':'Synth_1000f_16inf_1000s.csv',
    }
    
    try:
        arg = sys.argv[1]
    except:
        arg = 's1000-16'
    try:
        norm = sys.argv[2]
    except:
        norm = "L1inf"
        #norm = "L12"
        #norm = "Linf1"
        #norm = "L11"
    
    assert arg in FILES.keys(), f'Unexpected arg {arg}'
    assert norm.lower() in ["l1inf", "l12", "linf1","l11"],  f'Unexpected norm {norm}'
    
    isnorm = lambda N: norm.lower() == N.lower()
    
    print(f'Data {arg} - Norm {norm}\n')
    
    # read file name
    # file_name = "Synth_1000f_64inf_1000s.csv"
    file_name = FILES[arg]

    # Set device (GPU or CPU)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # enregistrer la matrice A
    if arg=="lung":
        A, Y, feature_names, label_name, patient_name = ReadData(
            file_name, doScale=True, doLog=True
        )
    else:
      A, Y, feature_names, label_name, patient_name = ReadData(
          file_name, doScale=False, doLog=False
      )  
    nbF = len(feature_names)

    # definir les eta choisis et les projections
    ######## Parameters ########
    
    normA , projection= computeValueNorm2(norm, A)
    #normA2 = computeValueNorm(norm, A)
    #ETA = [1000000, 200000,100000, 50000, 10000, 1000 ]
    #ETA = [1000, 300, 200,100 ,10,1]
    NP = 10
    #ETA = np.flip(np.linspace(0.01, 0.9, NP)*normAL1inf)
    etamax = 0.4
    ETANormalized =[f"{x:.2f}" for x in np.flip(np.linspace(0.01, etamax, NP)) ] 
    ETA =  np.flip(np.linspace(0.01, etamax, NP)*normA)

    projectionsA = []
    sparsityA = []
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    # run la projection pour plusieurs valeurs de eta et enregistrer les projections
    for proj in projection:
        tmp = []
        tmp2 = []
        for eta, eta_n in zip(ETA,  ETANormalized):
            print(proj.__name__, eta, eta_n)
            if proj == proj_l1infball:
                A = torch.Tensor(A)
            Q = proj(A, eta)
            # Faire la sparsité on doit voir quels sont les colonnes qui n'ont que des valeurs nulles
            if proj == proj_l1infball:
                # convert Q from tensor to np.array
                Q = Q.cpu().detach().numpy()

            tmp.append(Q)
            zero_columns_count = np.sum(np.all(Q == 0, axis=0)) / nbF
            tmp2.append(zero_columns_count)

        projectionsA.append(tmp)
        sparsityA.append(tmp2)

    # Calculer A-P(A) norme l2 pour toutes les projections
    projResultsAPA = []
    projResultsPA = []
    APA = []
    if (proj_l1infball in projection):
        A = A.cpu().detach().numpy()
    for i in range(len(projection)):
        tmp = []
        tmp2 = []
        tmp3 = []
        projectionA = projectionsA[i]
        for y in range(len(projectionA)):
            tmp3.append(A-projectionA[y])
            tmp.append(computeValueNorm(norm, A-projectionA[y]))
            tmp2.append(computeValueNorm(norm, projectionA[y]))

        projResultsAPA.append(tmp)
        projResultsPA.append(tmp2)
        APA.append(tmp3)



    projectionsResultsAPA = projResultsAPA/normA
    projectionsResultsPA = projResultsPA/normA
    
    mkname = lambda lab: f'{arg}-{norm}-{lab}.png'
    
    #print('PLOT', norm, np.array(sparsityA).sum())
    print('PLOT', norm, np.array(sparsityA).sum(axis=1))
    # Afficher les courbes
    if isnorm("L1inf"):
        
        # Courbe A-PA PA
        pltCurveWithPoints(projectionsResultsAPA, projectionsResultsPA, 
                           projection, ETANormalized, colors, 
                           r"Identity norm for projections ",
                           r"$\frac{||Y-P(Y)||_{1,infty}}{||Y||_{1,infty}}$", 
                           r"$\frac{||P(Y)||_{1,infty}}{||Y||_{1,infty}}$", 
                           "normal",  fontsize=20, save=mkname('APA-PA'))
        
        
        # # Courbe A - PA et sparsity
        # formula = r"$\frac{||Y-P(Y)||_{1,infty}}{||Y||_{1,infty}}$"
        # title = f'    Sparsity vs relative error ({formula})'
        # pltCurveWithPoints(projectionsResultsAPA, sparsityA, projection, 
        #                    ETANormalized, colors, 
        #                    title, r"relative error", "sparsity", "semilogy", 
        #                    fontsize=20, save=mkname('APA-sparsity')) 
        
        
        # Courbe PA et sparsity
        formula =  r"$\eta~=~\frac{||P(Y)||_{1,infty}}{||Y||_{1,infty}}$"
        title = f' Sparsity vs normalized ball radius {formula}'
        pltCurveWithPoints(np.flip(projectionsResultsPA), np.flip(sparsityA, axis=1), 
                           projection, np.flip(ETANormalized), colors, 
                           title, r"normalized ball radius", "sparsity", 
                           "normal", fontsize=15, save=mkname('PA-sparsity'))
        
        
    if isnorm("Linf1"):
        # Courbe A-PA PA
        pltCurveWithPoints(projectionsResultsAPA, projectionsResultsPA, 
                           projection, ETANormalized, colors, 
                           "L_curve for projections ",
                           r"$\frac{||A-P(A)||_{infty,1}}{||A||_{infty,1}}$",
                           r"$\frac{||P(A)||_{infty,1}}{||A||_{infty,1}}$", 
                           "normal", fontsize=10, save=mkname('APA-PA'))
 
        # Courbe A - PA et sparsity
        pltCurveWithPoints(projectionsResultsAPA, sparsityA, projection, 
                           ETANormalized, colors, 
                           r"Sparsity vs  $\frac{||A-P(A)||_{infty,1}}{||A||_{infty,1}}$",
                           r"$\frac{||A-P(A)||_{infty,1}}{||A||_{infty,1}}$", 
                           "sparsity", "normal", fontsize=10, save=mkname('APA-sparsity'))
        
 
        # Courbe PA et sparsity
        pltCurveWithPoints(np.flip(projectionsResultsPA), np.flip(sparsityA,axis=1), 
                           projection, np.flip(ETANormalized), colors, 
                           r"Sparsity vs $\frac{||P(A)||_{infty,1}}{||A||_{infty,1}}$",
                           r"$\frac{||P(A)||_{infty,1}}{||A||_{infty,1}}$", "sparsity", 
                           "normal", fontsize=10, save=mkname('PA-sparsity'))
        
         
    if isnorm("L12"):
        # Courbe A-PA PA
        pltCurveWithPoints(projectionsResultsAPA, projectionsResultsPA, projection, 
                            ETANormalized, colors, "Identity  L12 ",
                            r"$\frac{||Y-P(Y)||_{1,2}}{||Y||_{1,2}}$", 
                            r"$\frac{||P(Y)||_{1,2}}{||Y||_{1,2}}$", 
                            "normal", fontsize=15, save=mkname('APA-PA'))          
        
    #     # Courbe A - PA et sparsity
    #     pltCurveWithPoints(projectionsResultsAPA, sparsityA, projection, 
    #                        ETANormalized, colors, 
    #                        r"Sparsity vs  $\frac{||A-P(A)||_{1,2}}{||A||_{1,2}}$",
    #                        r"$\frac{||A-P(A)||_{1,2}}{||A||_{1,2}}$", "sparsity", 
    #                        "normal", fontsize=10, save=mkname('APA-sparsity')) 
         
        
        # Courbe PA et sparsity
        pltCurveWithPoints(np.flip(projectionsResultsPA), np.flip(sparsityA, axis=1), 
                            projection, np.flip(ETANormalized), colors, 
                            r"Sparsity as function of the $\frac{||P(A)||_{1,2}}{||A||_{1,2}}$",
                            r"$\frac{||P(A)||_{1,2}}{||A||_{1,2}}$", "sparsity", 
                            "normal", fontsize=10, save=mkname('PA-sparsity')) 
        
    if isnorm("L11"):
            # Courbe A-PA PA
            pltCurveWithPoints(projectionsResultsAPA, projectionsResultsPA, projection, 
                               ETANormalized, colors, "Identity  L1 ",
                               r"$\frac{||Y-P(Y)||_{1,1}}{||Y||_{1,1}}$", 
                               r"$\frac{||P(Y)||_{1,1}}{||Y||_{1,1}}$", 
                               "normal", fontsize=15, save=mkname('APA-PA')) 
            
    
            # # Courbe A - PA et sparsity
            # pltCurveWithPoints(projectionsResultsAPA, sparsityA, projection, 
            #                    ETANormalized, colors, 
            #                    r"Sparsity vs  $\frac{||Y-P(Y)||_{1,1}}{||A||_{1,1}}$",
            #                    r"$\frac{||Y-P(Y)||_{1,1}}{||A||_{1,1}}$", "sparsity", 
            #                    "normal", fontsize=10, save=mkname('APA-sparsity')) 
             
            
            # Courbe PA et sparsity
            pltCurveWithPoints(np.flip(projectionsResultsPA), np.flip(sparsityA, axis=1), 
                                projection, np.flip(ETANormalized), colors, 
                                r"Sparsity as function of the $\frac{||P(Y)||_{1,1}}{||Y||_{1,1}}$",
                                r"$\frac{||P(Y)||_{1,1}}{||Y||_{1,1}}$", "sparsity", 
                                "normal", fontsize=15, save=mkname('PA-sparsity')) 
        
    plt.show()