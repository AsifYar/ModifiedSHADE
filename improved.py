"""
Implements the MTS-LS1 indicated in MTS 
http://sci2s.ugr.es/EAMHCO/pdfs/contributionsCEC08/tseng08mts.pdf 
Lin-Yu Tseng; Chun Chen, "Multiple trajectory search for Large Scale Global Optimization," Evolutionary Computation, 2008. CEC 2008. (IEEE World Congress on Computational Intelligence). IEEE Congress on , vol., no., pp.3052,3059, 1-6 June 2008
doi: 10.1109/CEC.2008.4631210
and used by MOS
"""
from numpy import clip, zeros, flatnonzero, copy

from numpy.random import permutation

from ea.DE import EAresult
import numpy as np

from functools import partial

def _mtsls_improve_dim(function, sol, best_fitness, i, check , lower , upper , population  , totalevalsp):
    totalevals1 = 0
    score = best_fitness
    bestArr = copy(sol)   # bestArr === newsol 
    FitBest = best_fitness
    wmax = 0.2
    wmin = 0
    r2  =  np.random.rand() 
    

    mu = copy(bestArr)
    k0 = np.random.randint(0, len(population))
    n = np.random.randint(0 , len(sol) - 1 )
    while (n == i):
        n = np.random.randint(0 , len(sol) - 1)
    r2 = wmin + (((totalevalsp +  totalevals1 + 0.0) / 3000000) * (wmax - wmin))
    if ( np.random.rand() <= r2 ):
        mu[i] = bestArr[n] + (2 * np.random.rand() - 1) * (bestArr[n] -  population[k0][n] )
    else:
        mu[i] = bestArr[i] + (2 * np.random.rand() - 1) * (bestArr[n] - population[k0][n] )
            # making sure a gen isn't out of boundary
    if mu[i] > upper:
        mu[i] = (upper + bestArr[i]) / 2
    if (mu[i] < lower): 
        mu[i] = (lower + bestArr[i]) / 2
    mu = check(mu)
    score = function(mu) 
    totalevals1 += 1
    if (score <= FitBest):
        bestArr = mu
        FitBest = score
    return EAresult(solution=bestArr, fitness=FitBest, evaluations=totalevals1)

    

def mtsls(function, sol, fitness, lower, upper, maxevals, SR , population , totalevalsp):
    
    dim = len(sol)

    improved_dim = zeros(dim, dtype=bool)
    check = partial(clip, a_min=lower, a_max=upper)
    current_best = EAresult(solution=sol, fitness=fitness, evaluations=0)
    totalevals = 0

    improvement = zeros(dim)

    if totalevals < maxevals:
        dim_sorted = permutation(dim)

        for i in dim_sorted:
            result = _mtsls_improve_dim(function, current_best.solution, current_best.fitness, i, check , lower , upper , population  , totalevalsp)
            totalevals += result.evaluations
            improve = max(current_best.fitness - result.fitness, 0)
            improvement[i] = improve

            if improve:
                improved_dim[i] = True
                current_best = result
  

        dim_sorted = improvement.argsort()[::-1]
        d = 0


    return current_best, SR
