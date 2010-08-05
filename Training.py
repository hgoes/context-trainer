"""
The training algorithm
======================
"""
import numpy as np
import numpy.ma as ma
from SubtractiveClust import subclust,normalize
from GathGeva import GathGeva
from Regression import linear_regression
from evolve import evolve_fis
import rule

class TrainingState:
    """
    The current state of the algorithm
    """
    def __init__(self):
        self.classifier_states = []
    def add_classifier_state(self,st):
        """
        Add a new state to the training state

        :param st: The classifier state
        :type st: :class:`ClassifierState`
        """
        self.classifier_states.append(st)
    def buildFIS(self,iterations=2,cb=None):
        """
        Create a fuzzy inference system using a genetic algorithm        

        :param iterations: The algorithm can use the results from the last iteration to stabilize the resulting FIS. Specifies the amount of iterations to perform.
        :param cb: A callback function that is called with the progress encoded as a float from 0.0 to 1.0
        """
        classifiers = []
        rng = self.max_range() - self.min_range()
        for i,cl_state in enumerate(self.classifier_states):
            best_fis = None
            best_quality = 0
            for it in range(iterations):
                if cb:
                    cb((float(i)+float(it)/iterations) / len(self.classifier_states))
                fis = cl_state.best_fis(rng)
                if fis:
                    quality,count = cl_state.quality_fis(fis,it!=0)
                    print "Quality:",100.0*quality / count
                    if quality > best_quality:
                        best_fis = fis
                    cl_state.adjust_data(fis,it==0)
            if not best_fis:
                return None
            classifiers.append(rule.Classifier(best_fis,cl_state.membership(),cl_state.name))
        if cb:
            cb(1.0)
        return rule.ClassifierSet(classifiers)
    def min_range(self):
        return np.min([ cl.min_range() for cl in self.classifier_states ],0)
    def max_range(self):
        return np.max([ cl.max_range() for cl in self.classifier_states ],0)

class ClassifierState:
    """
    The state of a classifier while training.

    :param name: The name of the classifier
    :type name: :class:`str`
    """
    def __init__(self,name):
        self.name = name
        self.classes = []
    def add_class_state(self,st):
        self.classes.append(st)
    def clusters(self,rng):
        return [ cls.clusters(rng) for cls in self.classes ]
    def gen_fis(self,vecs):
        try:
            means = []
            covars = []
            for vec,cl_state in zip(vecs,self.classes):
                ggres = cl_state.gath_geva(vec)
                means.append(ggres[2])
                covars.append(ggres[3])
            ras = linear_regression(zip(means,covars),
                                    [(cl.id,cl.training_data) for cl in self.classes])
            print "Success"
            return build_classifier(means,covars,ras)
        except Exception as err:
            print err
            return None
    def eval_fis(self,fis):
        """
        Evaluate a FIS using the training data in the class states.

        :returns: A number representing the quality of the FIS. Higher is better.
        :return-type: :class:`float`
        """
        res = 0.0
        for cl_state in self.classes:
            res += cl_state.eval_fis(fis)
        return 1.0/res
    def best_fis(self,rng):
        """
        Calculate the best Fuzzy Inference System for this classifier using a genetic algorithm.

        :param rng: The normalization factor for the training data
        """
        return evolve_fis(self.clusters(rng),self.gen_fis,self.eval_fis)
    def quality_fis(self,fis,attach_dim=False):
        correct = 0
        count = 0
        for cl_state in self.classes:
            r,c = cl_state.quality_fis(fis,attach_dim)
            correct += r
            count += c
        return (correct,count)
    def adjust_data(self,fis,attach_dim=False):
        for cl_state in self.classes:
            cl_state.adjust_data(fis,attach_dim)
    def membership(self):
        return [(cl.name,cl.id) for cl in self.classes if cl.name!=None]
    def min_range(self):
        return np.min([cl.min_range() for cl in self.classes],0)
    def max_range(self):
        return np.min([cl.max_range() for cl in self.classes],0)

class ClassState:
    """
    The state of a context class
    """
    def __init__(self,name,id,tr_dat,ch_dat=None):
        self.name = name
        self.id = id
        self.training_data = tr_dat
        self.check_data = ch_dat
    def clusters(self,rng):
        """
        Use the subclustering algorithm to calculate initial clusters for this class from the training data
        
        :param rng: Normalization factor for the training data.
        :type rng: :class:`float`
        """
        #clusts = subclust(normalize(self.training_data),0.4,0.5)
        clusts = subclust(self.training_data / rng,0.4,0.5,7)
        print len(clusts),"initial clusters for class",self.name
        return np.array([self.training_data[i] for i in clusts])
    def gath_geva(self,vec):
        """
        Perform the Gath-Geva clustering algorithm on the training data using an initial state

        :param vec: The initial state for the algorithm
        """
        return GathGeva(self.training_data,vec)
    def eval_fis(self,fis):
        delt = self.id - fis.evaluates(self.training_data)
        return np.sum(delt*delt) / self.training_data.shape[0]
    def quality_fis(self,fis,attach_dim=False):
        """
        Count the correct classifications of a given FIS on the check data.

        :param fis: The Fuzzy Inference System to be tested
        :param attach_dim: Does the 
        """
        if attach_dim:
            dat = np.hstack((self.check_data,np.zeros((self.check_data.shape[0],1))))
        else:
            dat = self.check_data
        rvec = fis.evaluates(dat) - self.id
        rvec = ma.masked_inside(rvec,-0.5,0.5)
        return (ma.count_masked(rvec),self.check_data.shape[0])
    def adjust_data(self,fis,attach_dim=False):
        res = fis.evaluates(self.training_data)
        if attach_dim:
            self.training_data = np.concatenate((self.training_data,np.array([res]).T),1)
        else:
            self.training_data[:,-1] = res
    def min_range(self):
        return np.min(self.training_data,0)
    def max_range(self):
        return np.max(self.training_data,0)

def build_classifier(means,covars,ras):
    rules = []
    for mean,covar,ra in zip(means,covars,ras):
        for i in range(mean.shape[0]):
            r = rule.ComplexRule(ra[i,:-1],ra[i,-1],mean[i],np.linalg.inv(covar[i]))
            rules.append(r)
    return rule.RuleSet(rules)

def adjustTrainingData(dat,dat2,classifier):
    add = classifier.evaluates(dat)
    #print add
    dat2[:,-1] = add
    #for i in range(dat.shape[0]):
    #    dat2[i,-1] = classifier.evaluate(dat[i])

def buildFIS(training_data,iterations=2,cb=None,check_data=None):
    """
    Create a fuzzy inference system using a genetic algorithm

    :param training_data: The training data used to generate the FIS
    :param iterations: The algorithm can use the results from the last iteration to stabilize the resulting FIS. Specifies the amount of iterations to perform.
    :param cb: A callback function that is called with the progress encoded as a float from 0.0 to 1.0
    :param check_data: A data set that is used to calculate the quality of the FIS. The resulting FIS will be the one producing best results on the check data.
    """
    if check_data:
        best_fis = (None,0.0)
    fis = None
    for it in range(iterations):
        if cb:
            cb(float(it)/float(iterations))
        clusts = []
        for cls,dat in training_data:
            print "Clustering for class",cls,"..."
            clust = subclust(normalize(dat),0.4,0.5)
            print len(clust),"cluster centers"
            clusts.append(np.array([dat[i] for i in clust]))
        
        def gen_fis(vecs):
            means = []
            covars = []
            try:
                for vec,(cl,dat) in zip(vecs,training_data):
                    ggres = GathGeva(dat,vec)
                    means.append(ggres[2])
                    covars.append(ggres[3])
                ras = linear_regression(zip(means,covars),training_data)
                return build_classifier(means,covars,ras)
            except:
                return None
        def eval_fis(fis):
            res = 0.0
            for cls,dat in training_data:
                delt = cls - fis.evaluates(dat)
                res += np.sum(delt**2)/dat.shape[0]
            return 1.0/res
        fis = evolve_fis(clusts,gen_fis,eval_fis)
        if check_data:
            quality = 0
            count_all = 0
            for cls,dat in check_data:
                if it==0:
                    rvec = fis.evaluates(dat)
                else:
                    rvec = fis.evaluates(np.hstack((dat,np.zeros((dat.shape[0],1)))))
                rvec -= cls
                rvec = ma.masked_inside(rvec,-0.5,0.5)
                count_all += dat.shape[0]
                quality += ma.count_masked(rvec)
            
            print "Quality:",100.0*quality/count_all
            if quality > best_fis[1]:
                best_fis = (fis,quality)
        if it == 0:
            training_data2 = []
            for cls,dat in training_data:
                add_arr = np.empty((dat.shape[0],1))
                dat2 = np.concatenate((dat,add_arr),1)
                adjustTrainingData(dat,dat2,fis)
                training_data2.append((cls,dat2))
            training_data = training_data2
        else:
            for cls,dat in training_data:
                adjustTrainingData(dat,dat,fis)
    if cb:
        cb(1.0)
    if check_data:
        return best_fis[0]
    else:
        return fis
