"""
Subtractive clustering
======================
"""
import numpy as np
from math import exp

def normalize(arr):
    """
    Normalizes a dataset such that every value lies within 0.0 and 1.0.
    If all values in a dimension are the same then a tiny range is calculated.
    """
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
    """
    Finds a set of points in a data set that define optimal cluster centers.

    :param data: A dataset represented by a matrix. Columns represent dimensions, rows represent different data points.
    :param radii: Either a vector representing cluster radii for each dimension or a number giving the same radius for every dimension.
    :param sqshFactor: A factor that is multiplied with the radius to discourage the selection of cluster centers within that range around another center.
    :param acceptRatio: If the ratio between the first center and the current is greater than this, the cluster center is automatically accepted.
    :param rejectRatio: If the ratio between the first center and the current is less than this, the cluster center is automatically rejected.
    :returns: A list of indices of cluster centers.
    :rtype: [:class:`int`]
    """
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
            #The point is the first cluster center, accept it by default.
            firstPot = potentials[center]
            accept = True
        elif potentials[center] > acceptRatio*firstPot:
            #The point lies within the accept ratio, accept it.
            accept = True
        elif potentials[center] > rejectRatio*firstPot:
            #The point is neither within the accept nor the reject ratio.
            #Accept it only if it has a good ratio between a high potential and being far away from other centers
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
            #The point lies within the reject ratio, all following points will also, so terminate.
            break

        if accept:
            clusters.append(center)
            revise(center,potentials,data,sqshMultp)
        else:
            potentials[center] = 0.0
    return clusters
