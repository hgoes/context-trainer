import numpy as np
from math import exp

def normalize(arr):
    mins = np.min(arr,0)
    maxs = np.max(arr,0)
    ranges = maxs-mins
    return arr / ranges

def potential(p,ps,ra=4.0):
    alpha = -4.0 / ra / ra
    diff = ps - p
    dots = np.sum(diff*diff,1)
    return np.sum(np.exp(alpha*dots))

def revise(c,P,data,rb=5.0):
    beta = -4.0 / rb / rb
    pc = P[c]
    for i in range(len(P)):
        v = data[i] - data[c]
        P[i] = P[i] - pc*exp(beta*np.vdot(v,v))
    
def subclust(data,ra=0.4,rb=0.5):
    nP,L = data.shape
    potentials = np.empty(nP)
    for i in range(nP):
        potentials[i] = potential(data[i],data,ra)
        #print i," of ",nP,":",potentials[i]
    clusters = []

    firstPot = None
    while True:
        center = max(range(nP),key=lambda i: potentials[i])
        if firstPot is None:
            firstPot = potentials[center]
        elif potentials[center] < 0.15*firstPot:
            break
        clusters.append(center)
        #print "Found cluster ",center
        revise(center,potentials,data,rb)
    return clusters

#arr = np.array([[0.6,3.9],
#                [0.9,3.4],
#                [1.2,4.1],
#                [1.4,3.5],
#                [3.5,0.5],
#                [4.0,1.3],
#                [4.1,0.4],
#                [4.9,0.5]])

#print subclust(normalize(arr*100000))
#print subclust(arr)
