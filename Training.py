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
import math

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
        :type iterations: :class:`int`
        :param cb: A callback function that is called with the progress encoded as a float from 0.0 to 1.0
        """
        classifiers = []
        #rng = self.max_range() - self.min_range()
        for i,cl_state in enumerate(self.classifier_states):
            best_fis = None
            best_quality = 0
            for it in range(iterations):
                rng = self.max_range() - self.min_range()
                for i in range(rng.shape[0]):
                    if rng[i] == 0.0:
                        rng[i] = 1.0
                if cb:
                    cb((float(i)+float(it)/iterations) / len(self.classifier_states))
                fis = cl_state.best_fis(rng)
                if fis:
                    quality,count = cl_state.quality_fis(fis)
                    print "Quality:",100.0*quality / count
                    if quality > best_quality:
                        best_fis = fis
                    if it==0:
                        cl_state.attach_dimension()
                    cl_state.adjust_data(fis)
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
        """
        Generate a new Fuzzy Inference System by using start values for the clustering algorithm

        :param vecs: A vector of vectors of initial cluster centers for the gath geva algorithm
        """
        try:
            means = []
            covars = []
            for vec,cl_state in zip(vecs,self.classes):
                ggres = cl_state.gath_geva(vec)
                means.append(ggres[2])
                covars.append(ggres[3])
            ras = linear_regression(zip(means,covars),
                                    [(cl.id,cl.get_training_data()) for cl in self.classes])
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
        #res = 0.0
        #for cl_state in self.classes:
        #    res += cl_state.eval_fis(fis)
        #print "=>",res
        #return 1.0/res
        correct,count = self.quality_fis(fis)
        return correct
    def best_fis(self,rng):
        """
        Calculate the best Fuzzy Inference System for this classifier using a genetic algorithm.

        :param rng: The normalization factor for the training data
        """
        return evolve_fis(self.clusters(rng),self.gen_fis,self.eval_fis)
    def quality_fis(self,fis):
        """
        Get a quality estimate for a FIS by counting the correct classifications on the check data.

        :returns: A tuple with the correct classifications and the total number of check-datasets
        :return-type: (:class:`int`, :class:`int`)
        """
        correct = 0
        count = 0
        for cl_state in self.classes:
            r,c = cl_state.quality_fis(fis)
            print "For",cl_state.name,r,"/",c
            correct += r
            count += c
        return (correct,count)
    def adjust_data(self,fis):
        for cl_state in self.classes:
            cl_state.adjust_data(fis)
    def membership(self):
        return [(cl.name,cl.id) for cl in self.classes if cl.name!=None]
    def min_range(self):
        return np.min([cl.min_range() for cl in self.classes],0)
    def max_range(self):
        return np.min([cl.max_range() for cl in self.classes],0)
    def attach_dimension(self):
        for cl_state in self.classes:
            cl_state.attach_dimension()

class ClassState:
    """
    The state of a context class
    """
    def __init__(self,name,id,tr_dat,ch_dat=None):
        self.name = name
        self.id = id
        #self.training_data = tr_dat
        self.training_data = np.hstack((tr_dat,np.zeros((tr_dat.shape[0],1))))
        self.check_data = ch_dat
        self.extended = False
    def clusters(self,rng):
        """
        Use the subclustering algorithm to calculate initial clusters for this class from the training data
        
        :param rng: Normalization factor for the training data.
        :type rng: :class:`float`
        """
        #clusts = subclust(normalize(self.training_data),0.4,0.5)
        if self.extended:
            dat = self.training_data / rng
        else:
            dat = self.training_data[:,0:-1] / rng[0:-1]

        clusts = subclust(normalize(dat),0.2,0.3,0.2,0.1) #0.4 0.5

        print len(clusts),"initial clusters for class",self.name
        if self.extended:
            return np.array([self.training_data[i] for i in clusts])
        else:
            return np.array([self.training_data[i,0:-1] for i in clusts])
    def gath_geva(self,vec):
        """
        Perform the Gath-Geva clustering algorithm on the training data using an initial state

        :param vec: The initial state for the algorithm
        """
        if self.extended:
            return GathGeva(self.training_data,vec)
        else:
            return GathGeva(self.training_data[:,0:-1],vec)
    def eval_fis(self,fis):
        if fis.dimension() == self.training_data.shape[1]:
            delt = self.id - fis.evaluates(self.training_data)
        else:
            delt = self.id - fis.evaluates(self.training_data[:,0:-1])
        print ma.count_masked(ma.masked_inside(delt,-0.5,0.5)),"/",self.training_data.shape[0]
        return np.sum(delt*delt) / self.training_data.shape[0]
    def quality_fis(self,fis):
        """
        Count the correct classifications of a given FIS on the check data.

        :param fis: The Fuzzy Inference System to be tested
        :type fis: :class:`rule.ClassifierSet`
        :returns: A tuple containing the number of correct classifications and the total number of classifications
        :rtype: (:class:`int`,:class:`int`)
        """
        if fis.dimension() == self.training_data.shape[1]:
            last_res = 0.0
            count = 0
            for i in range(self.check_data.shape[0]):
                last_res = fis.evaluate(np.hstack((self.check_data[i],last_res)))
                if abs(last_res - self.id) < 0.5:
                    count = count + 1
            return (count,self.check_data.shape[0])
        else:
            rvec = fis.evaluates(self.check_data) - self.id
            rvec = ma.masked_inside(rvec,-0.5,0.5)
            return (ma.count_masked(rvec),self.check_data.shape[0])
        
        if fis.dimension() == self.training_data.shape[1]:
            dat = np.hstack((self.check_data,self.id*np.ones((self.check_data.shape[0],1))))
        else:
            dat = self.check_data
        #if self.check_data.shape[1] == self.training_data.shape[1]:
        #    dat = self.check_data
        #else:
        #    dat = np.hstack((self.check_data,np.zeros((self.check_data.shape[0],1))))
        rvec = fis.evaluates(dat) - self.id
        rvec = ma.masked_inside(rvec,-0.5,0.5)
        return (ma.count_masked(rvec),self.check_data.shape[0])
    def adjust_data(self,fis):
        """
        Use a FIS to adjust the last dimension of the training data with the results of the evaluation.

        :param fis: The Fuzzy Inference System to be used.
        """
        if fis.dimension() == self.training_data.shape[1]:
            self.training_data[:,-1] = fis.evaluates(self.training_data[:,0:-1])
        else:
            last_res = 0.0
            for i in range(self.training_data.shape[0]):
                if math.isnan(last_res):
                    last_res = 0.0
                self.training_data[i,-1] = last_res
                last_res = fis.evaluate(self.training_data[i,:-1])
    def min_range(self):
        return np.min(self.training_data,0)
    def max_range(self):
        return np.max(self.training_data,0)
    def attach_dimension(self):
        print "Attaching new dimension"
        self.extended = True
        #self.training_data = np.hstack((self.training_data,np.zeros((self.training_data.shape[0],1))))
    def get_training_data(self):
        if self.extended:
            return self.training_data
        else:
            return self.training_data[:,0:-1]

def build_classifier(means,covars,ras):
    rules = []
    for mean,covar,ra in zip(means,covars,ras):
        for i in range(mean.shape[0]):
            r = rule.ComplexRule(ra[i,:-1],ra[i,-1],mean[i],np.linalg.inv(covar[i]))
            rules.append(r)
    return rule.RuleSet(rules)
