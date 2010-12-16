from pyevolve import G1DBinaryString
from pyevolve import GSimpleGA
import numpy as np

def evolve_fis(vecs,gen_fis,eval_fis,generations=10,pop_size=20):
    cache = {}
    size = 0
    for vec in vecs:
        size += vec.shape[0]
    genome = G1DBinaryString.G1DBinaryString(size)
    
    def loc_fitness(bitvec):
        if bitvec.getDecimal() in cache:
            return cache[bitvec.getDecimal()]
        offset = 0
        rvecs = []
        for vec in vecs:
            sel_vec = np.array([vec[i] for i in range(vec.shape[0]) if bitvec[offset+i]==1])
            if sel_vec.shape[0] == 0:
                cache[bitvec.getDecimal()] = 0.0
                return 0.0
            rvecs.append(sel_vec)
            offset += vec.shape[0]
        fis = gen_fis(rvecs)
        if fis is None:
            res = 0.0
        else:
            res = eval_fis(fis)
            
        cache[bitvec.getDecimal()] = res
        return res

    genome.evaluator.set(loc_fitness)

    ga = GSimpleGA.GSimpleGA(genome,interactiveMode=True)
    ga.setGenerations(generations)
    ga.setPopulationSize(pop_size)
    #ga.terminationCriteria.set(GSimpleGA.ConvergenceCriteria)
    ga.evolve(freq_stats=1)

    bitvec = ga.bestIndividual()

    offset = 0
    rvecs = []
    for vec in vecs:
        sel_vec = np.array([vec[i] for i in range(vec.shape[0]) if bitvec[offset+i]==1])
        rvecs.append(sel_vec)
        offset += vec.shape[0]

    return gen_fis(rvecs)

def bit(c):
    if c == '0':
        return 0
    else:
        return 1

def evolve_bitvec(fis,eval_fis,generations,pop_size):
    cache = {}
    genome = G1DBinaryString.G1DBinaryString(fis.dimension()*len(fis.rules))

    def set_bitvec(bitvec):
        rvec = np.array(map(bit,bitvec.getBinary()))
        for (i,rule) in enumerate(fis.rules):
            rule.set_bitvec(rvec[i*fis.dimension():(i+1)*fis.dimension()].nonzero())

    def loc_fitness(bitvec):
        if bitvec.getDecimal() in cache:
            return cache[bitvec.getDecimal()]
        set_bitvec(bitvec)
        res = eval_fis(fis)
        cache[bitvec.getDecimal()] = res
        return res

    genome.evaluator.set(loc_fitness)

    ga = GSimpleGA.GSimpleGA(genome,interactiveMode=True)
    ga.setGenerations(generations)
    ga.setPopulationSize(pop_size)
    #ga.terminationCriteria.set(GSimpleGA.ConvergenceCriteria)
    ga.evolve(freq_stats=1)

    bitvec = ga.bestIndividual()
    print "Evolved bitvec: ",bitvec.getBinary()
    set_bitvec(bitvec)

