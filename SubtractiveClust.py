import numpy as np
from math import exp

def normalize(arr):
    mins = np.min(arr,0)
    maxs = np.max(arr,0)
    for i in range(len(mins)):
        if mins[i] == maxs[i]:
            mins[i] -= 0.0001*(1.0 + abs(mins[i]))
            maxs[i] += 0.0001*(1.0 + abs(maxs[i]))
    ranges = maxs-mins
    return (arr - mins) / (maxs - mins)

def potential(p,ps,accumMultp):
    diff = (ps - p) * accumMultp
    dots = np.sum(diff*diff,1)
    return np.sum(np.exp(-4.0*dots))

def revise(c,P,data,sqshMultp):
    vs = (data[c] - data) * sqshMultp
    P -= P[c] * np.exp(-4.0 * np.sum(vs ** 2.0,1))
    for i in range(len(P)):
        if P[i] < 0.0:
            P[i] = 0.0
    
def subclust(data,radii=0.5,sqshFactor=1.25,acceptRatio=0.5,rejectRatio=0.15):
    nP,L = data.shape

    if type(radii) is not np.ndarray:
        radii = radii * np.ones(L)
    accumMultp = 1.0 / radii
    sqshMultp = 1.0 / (sqshFactor * radii)
    
    potentials = np.empty(nP)

    for i in range(nP):
        potentials[i] = potential(data[i],data,accumMultp)

    clusters = []

    firstPot = None
    while True:
        center = np.argmax(potentials)
        
        if firstPot is None:
            firstPot = potentials[center]
            accept = True
        elif potentials[center] > acceptRatio*firstPot:
            accept = True
        elif potentials[center] > rejectRatio*firstPot:
            minDistSq = None
            for cl in clusters:
                dx = (data[center] - data[cl]) * accumMultp
                dxSq = np.dot(dx,dx)
                if minDistSq is None or dxSq < minDistSq:
                    minDistSq = dxSq
            if (potentials[center] / firstPot) + np.sqrt(minDistSq) >= 1:
                accept = True
            else:
                accept = False
        else:
            break

        if accept:
            clusters.append(center)
            revise(center,potentials,data,sqshMultp)
        else:
            potentials[center] = 0.0
    return clusters
