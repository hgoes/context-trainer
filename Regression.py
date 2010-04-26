import numpy as np
from math import *

def linear_regression(means,covars,training_data,c):
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

    B = np.empty((tr_len,count_clusters*(tr_width+1)))
    #Btmp = np.ones((tr_len,count_clusters*(tr_width+1)))

    for i in range(count_clusters):
        for j in range(tr_len):
            B[j,(i+1)*(tr_width+1)-1] = w[i,j]
            #Btmp[j,(i+1)*(tr_width+1)-1] -= 1.0
            for k in range(tr_width):
                B[j,i*(tr_width+1)+k] = w[i,j]*training_data[j,k]
                #Btmp[j,i*(tr_width+1)+k] -= 1.0

    #print B[0:5,0:5]
    #print np.nonzero(Btmp)
    #print B
    #print "count_clusters: ",count_clusters
    #print "tr_len: ",tr_len
    #print "tr_width: ",tr_width
    #print "B'*B:",np.dot(B.T,B)

    u = c * np.ones((1,training_data.shape[0]))

    a = np.dot(np.dot(np.linalg.inv(np.dot(B.T,B)),B.T),u.T).flatten()

    ra = np.empty((count_clusters,tr_width+1))
    for i in range(count_clusters):
        ra[i] = a[i*(tr_width+1) : (i+1)*(tr_width+1)]
        
    return ra

#print linear_regression(np.array([[1,2,3],[2,3,4]]),np.array([[[1,2,3],[4,5,6],[7,8,9]],[[1.5,1.5,2],[6,1,9],[5,1,3]]]),np.array([[1,2,3],[4,5,6],[3,3,3]]),1)
