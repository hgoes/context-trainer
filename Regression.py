import numpy as np
from math import *

def matlab_2d_array(arr):
    res = "["
    for i in range(arr.shape[0]):
        if i != 0:
            res+= " ; "
        for j in range(arr.shape[1]):
            if j != 0:
                res+= ", "
            res+= str(arr[i,j])
    res+= "]"
    return res

def debug_regression(params,training_data,fn):
    count_clusters = 0
    means_shape = None
    for means,covars in params:
        count_clusters += means.shape[0]
        means_shape = means.shape[1]
    vmeans = np.empty((count_clusters,means_shape))
    i = 0
    for means,covars in params:
        for mean in means:
            vmeans[i] = mean
            i+=1
    tr_dat_l = 0
    for cl,dat in training_data:
        tr_dat_l += dat.shape[0]
    tr_dat_w = training_data[0][1].shape[1]
    ntraining_data = np.empty((tr_dat_l,tr_dat_w+1))
    i = 0
    for cl,dat in training_data:
        for ln in dat:
            ntraining_data[i,0:tr_dat_w] = ln
            ntraining_data[i,tr_dat_w] = cl
            i += 1
    with open(fn,'w') as f:
        print >> f, "covars = [];"
        i = 1
        for means,covars in params:
            for covar in covars:
                print >> f, "covars(:,:,",i,") =",matlab_2d_array(covar),";"
                i+=1
        print >> f, "means =",matlab_2d_array(vmeans),";"
        
        print >> f, "training_data =",matlab_2d_array(ntraining_data),";"
        print >> f, "disp(regression(means,covars,training_data));"

def linear_regression(params,training_data):
    #debug_regression(params,training_data,"test_regression.m")
    tr_len = 0
    tr_width = None
    for cl,dat in training_data:
        l,tr_width = dat.shape
        tr_len += l
    count_clusters = 0
    for means,covars in params:
        count_clusters += means.shape[0]
    
    w = np.empty((count_clusters,tr_len))

    i = 0
    for means,covars in params:
        for means_i,covars_i in zip(means,covars):
            sig_inv = np.linalg.inv(covars_i)
            j = 0
            for cl,dat in training_data:
                dists = dat - means_i
                # this is faster, but not numerically equivalent
                #alphas = np.sum(np.dot(dists,sig_inv)*dists,1)
                alphas = np.diag(np.dot(np.dot(dists,sig_inv),dists.T))
                #alphas = np.diag(np.tensordot(np.dot(dists,sig_inv),dists.T,1))
                w[i,j:j+dat.shape[0]] = np.exp(-0.5*alphas)
                j += dat.shape[0]
            i += 1

    w /= np.sum(w,0)

    B = np.empty((tr_len,count_clusters*(tr_width+1)))

    for i in range(count_clusters):
        j = 0
        for cl,dat in training_data:
            for dat_j in dat:
                B[j,(i+1)*(tr_width+1)-1] = w[i,j]
                B[j,i*(tr_width+1):(i+1)*(tr_width+1)-1] = dat_j * w[i,j]
                j += 1

    u = np.empty(tr_len)
    i = 0
    for cl,dat in training_data:
        for j in range(dat.shape[0]):
            u[i] = cl
            i += 1
    u = np.array([u])

    a = np.dot(np.dot(np.linalg.inv(np.dot(B.T,B)),B.T),u.T).flatten()

    ras = []
    i = 0
    for means,covars in params:
        ra = np.empty((means.shape[0],tr_width+1))
        for j in range(means.shape[0]):
            ra[j] = a[i*(tr_width+1) : (i+1)*(tr_width+1)]
            i += 1
        ras.append(ra)
        
    return ras
