import numpy as np
from math import *

def linear_regression(means,covars,training_data):
    tr_len,tr_width = training_data.shape
    count_clusters = means.shape[0]
    
    w = np.empty((count_clusters,tr_len))
    
    for i in range(count_clusters):
        sig_inv = np.linalg.inv(covars[i])
        for j in range(tr_len):
            dist = np.array([training_data[j] - means[i]])
            w[i,j] = exp(-0.5*np.dot(np.dot(dist,sig_inv),dist.T))
            
    sum_w = np.sum(w,0)

    for i in range(count_clusters):
        for j in range(tr_len):
            w[i,j] /= sum_w[j]
    
    B = np.empty((tr_len,count_clusters*tr_width))
    Btmp = np.ones((tr_len,count_clusters*tr_width))
    for i in range(count_clusters):
        for j in range(tr_len):
            B[j,(i+1)*tr_width-1] = w[i,j]
            Btmp[j,(i+1)*tr_width-1] -= 1.0
            for k in range(tr_width-1):
                B[j,i*tr_width+k] = w[i,j]*training_data[j,k]
                Btmp[j,i*tr_width+k] -= 1.0

    print np.nonzero(Btmp)
    print "count_clusters: ",count_clusters
    print "tr_len: ",tr_len
    print "tr_width: ",tr_width
    #print "B'*B:",np.dot(B.T,B)

    u = training_data[:,tr_width-1]

    a = np.dot(np.dot(np.linalg.inv(np.dot(B.T,B)),B.T),u)

    ra = np.empty((count_clusters,tr_width))
    for i in range(count_clusters):
        ra[i] = a[i*tr_width : (i+1)*tr_width]
        
    return ra

