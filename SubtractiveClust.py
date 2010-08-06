import numpy as np
from math import exp

def normalize(arr):
    mins = np.min(arr,0)
    maxs = np.max(arr,0)
    ranges = maxs-mins
    return arr / ranges

def potential(p,ps,ra=0.5):
    alpha = -4.0 / (ra * ra)
    diff = ps - p
    dots = np.sum(diff*diff,1)
    return np.sum(np.exp(alpha*dots))

def revise(c,P,data,rb=0.75):
    beta = -4.0 / (rb * rb)
    #pc = P[c]
    vs = data - data[c]
    P -= P[c] * np.exp(beta * np.sum(vs * vs,1))
    #for i in range(len(P)):
        #v = data[i] - data[c]
        #v = vs[i]
        #P[i] = P[i] - pc*exp(beta*np.vdot(v,v))
    
def subclust(data,ra=0.5,rb=0.75,ar=0.5,rr=0.15):
    nP,L = data.shape
    potentials = np.empty(nP)
    #print range(nP)
    for i in range(nP):
        potentials[i] = potential(data[i],data,ra)
        #print i," of ",nP,":",potentials[i]
    clusters = []

    firstPot = None
    while True:
        #center = max(range(nP),key=lambda i: potentials[i])
        center = np.argmax(potentials)
        if firstPot is None:
            firstPot = potentials[center]
	    clusters.append(center)
	elif potentials[center] > ar*firstPot:
	    clusters.append(center)
	    #print "new cluster found " + center
        elif potentials[center] < rr*firstPot:
            break
	else:
	    cl=clusters.__len__()
	    if cl > 1:
	      d = np.empty(cl)
	    for i in range(cl):
		if cl > 1:
		  #d[i] = np.sqrt(np.sum((data[center]-data[center[i]])**2))
		  d = np.sqrt(np.sum((data[center]-data[center])**2))
		  #print "Distance of " + data[center] + " to cluster center " + data[center[i]] + " is " + d[i]
		else:
		  d = np.sqrt(np.sum((data[center]-data[center])**2))
	    d_min = np.min(d)
	    p =((d_min/ra) + (potentials[center]/firstPot))
	    #print "value p=" + p
	    if p >= 1:
		clusters.append(center)
		#print "new cluster found " + center
	    else:
		potentials[center] = 0
        
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
