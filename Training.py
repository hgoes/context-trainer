import numpy as np
from SubtractiveClust import subclust,normalize
from GathGeva import GathGeva
from Regression import linear_regression
from evolve import evolve_fis
import rule

def build_classifier(means,covars,rvecs):
    rules = []
    for i in range(means.shape[0]):
        r = rule.ComplexRule(rvecs[i,:-1],rvecs[i,-1],means[i],np.linalg.inv(covars[i]))
        rules.append(r)
    return rule.RuleSet(rules)


def adjustTrainingData(dat,dat2,classifier):
    dat2[:,-1] = classifier.evaluates(dat)
    #for i in range(dat.shape[0]):
    #    dat2[i,-1] = classifier.evaluate(dat[i])

def buildFIS(dat,c):
    print "Initial clustering..."
    clust = subclust(normalize(dat),0.1,0.2)
    print len(clust),"initial cluster centers"
    arr = np.array([dat[c] for c in clust])
    ggres = GathGeva(dat,arr)
    means = ggres[2]
    sigmas = ggres[3]
    ra = linear_regression(means,sigmas,dat,c)
    cl = build_classifier(means,sigmas,ra)
    add_arr = np.empty((dat.shape[0],1))
    dat2 = np.concatenate((dat,add_arr),1)
    adjustTrainingData(dat,dat2,cl)
    fis = None
    for it in range(5):
        print "Clustering..."
        clust2 = subclust(normalize(dat2),0.1,0.2)
        print len(clust2),"cluster centers"
        def gen_fis(vec):
            ggres2 = GathGeva(dat2,vec)
            ra2 = linear_regression(ggres2[2],ggres2[3],dat2,c)
            return build_classifier(ggres2[2],ggres2[3],ra2)
        def eval_fis(fis):
            res = 0.0
            for i in range(dat2.shape[0]):
                delt = c - fis.evaluate(dat2[i])
                res += delt*delt
            return dat2.shape[0]/res
        fis = evolve_fis(np.array([dat2[i] for i in clust2]),gen_fis,eval_fis)
        adjustTrainingData(dat2,dat2,fis)
    return fis
