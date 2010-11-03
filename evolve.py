from pyevolve import G1DBinaryString
from pyevolve import GSimpleGA
import numpy as np

def evolve_fis(vecs,gen_fis,eval_fis):
    size = 0
    for vec in vecs:
        size += vec.shape[0]
    genome = G1DBinaryString.G1DBinaryString(size)
    
    def loc_fitness(bitvec):
        offset = 0
        rvecs = []
        for vec in vecs:
            sel_vec = np.array([vec[i] for i in range(vec.shape[0]) if bitvec[offset+i]==1])
            if sel_vec.shape[0] == 0:
                return 0.0
            rvecs.append(sel_vec)
        fis = gen_fis(rvecs)
        if fis is None:
            return 0.0
        else:
            return eval_fis(fis)

    genome.evaluator.set(loc_fitness)

    ga = GSimpleGA.GSimpleGA(genome,interactiveMode=True)
    ga.setGenerations(10)
    ga.setPopulationSize(10)
    ga.terminationCriteria.set(GSimpleGA.ConvergenceCriteria)
    ga.evolve(freq_stats=1)

    bitvec = ga.bestIndividual()

    offset = 0
    rvecs = []
    for vec in vecs:
        sel_vec = np.array([vec[i] for i in range(vec.shape[0]) if bitvec[offset+i]==1])
        rvecs.append(sel_vec)
        offset += vec.shape[0]

    return gen_fis(rvecs)
