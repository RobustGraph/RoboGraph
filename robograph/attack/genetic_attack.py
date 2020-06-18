import numpy as np
from robograph.attack.utils import calculate_Fc, check_budgets_and_symmetry
from copy import deepcopy
import random


np.random.seed(0)

population_size = 100 #Population Size for Each Generation
cross_rate = 0.2 #Generally from 0.1 to 0.5
mutate_rate = 0.4 #Generally from 0.1 to 0.5
# rounds = 10 #Number of Rounds of evolution

def calculate_budget(A_org, A_perturbed):
    """
    Returns the total pertubations count
    """
    return int(np.sum(abs(A_org - A_perturbed)))

def check_local(A_org_row, A_perturbed_row):
    """
    Returns the total pertubations in a row 
    """
    return int(np.sum(abs(A_org_row - A_perturbed_row)))

class Genetic_Attack(object):
    """
    Genetic algorithm (upper bound) for solving min_{A_G^{1+2+3}} F_c(A)

    param: 
        A_org:          original adjacency matrix
        XW:             XW
        U:              (u_y-u_c)/nG
        delta_l:        row budgets (vector)
        delta_g:        global budget (scalar)
        activation:     'linear' or 'relu'
    """

    def __init__(self, A_org, XW, U, delta_l, delta_g, activation):
        self.A_org = A_org
        self.XW, self.U = XW, U
        self.XWU = XW @ U
        self.delta_l = delta_l
        self.delta_g = delta_g
        self.activation = activation
        self.nG = A_org.shape[0]
        self.fc = calculate_Fc(A_org, self.XW, self.U,self.activation)
        self.population = []
        self.solution = None
        self.sol = None
        for k in range(population_size):
            L = np.zeros(self.nG,dtype=int) #Array Containing Local Budgets
            F = np.zeros((self.nG,self.nG),dtype=int)
            for i in range(self.nG):
                while True:
                    l_temp = np.random.randint(0,int(self.delta_l[0])+1,size=1)
                    if sum(L) + l_temp <= self.delta_g:
                        break
                L[i] = l_temp
                indices_flip = np.random.choice(self.nG,l_temp, replace = False)
                F[i,indices_flip] = 1
            A_new = np.multiply(1-2*A_org, F) + A_org
            np.fill_diagonal(A_new, 0)
            A_new = (A_new + A_new.T)//2
            self.population.append(A_new)

    
    def fitness(self):
        fitness_scores = np.zeros(len(self.population))
        if self.solution is None:
            for i in range(len(self.population)):
                fitness_scores[i] = calculate_Fc(self.population[i], self.XW, self.U, self.activation)
                if fitness_scores[i] < self.fc:
                    self.fc = fitness_scores[i]
                    self.sol = {
                    'opt_A': self.population[i],
                    'opt_f': fitness_scores[i]
                    }
                if fitness_scores[i] < 0:
                    self.solution = {
                    'opt_A': self.population[i],
                    'opt_f': fitness_scores[i]
                    }
                    break
        return fitness_scores

    
    def select(self, fitness):
        scores = np.exp(fitness)
        min_args = np.argsort(scores)

        result = []
        for i in range(population_size - population_size // 2):
            result.append(deepcopy(self.population[min_args[i]]))
        
        idx = np.random.choice(np.arange(population_size), 
                                size=population_size // 2,
                                replace=True, 
                                p=scores/scores.sum())
        
        for i in idx:
            result.append(deepcopy(self.population[i]))                                

        return result

    
    def crossover(self, parent, pop):
        if np.random.rand() < cross_rate:
            another = pop[np.random.randint(len(pop))]
            lll = np.random.rand()
            if lll <= 0.25:
                return np.copy(another)
            elif lll>0.25 and lll<=0.5:
                return np.copy(parent)
            else:
                tem = None
                count = 0 
                for k in range(3):
                    tem = np.zeros((self.nG, self.nG),dtype=int)
                    for i in range(self.nG):
                        if np.random.rand() < 0.5:
                            tem[i,i:] = parent[i,i:]
                            tem[i:,i] = parent[i:,i]
                        else:
                            tem[i,i:] = another[i,i:]
                            tem[i:,i] = another[i:,i]
                        if check_local(tem[i],self.A_org) > int(self.delta_l[0]):
                            tem = None
                            break
                    if tem!=None and calculate_budget(tem, self.A_org) <= self.delta_g:
                        break
                if tem!=None:
                    return tem
                else:
                    return np.copy(another)
        else:
            return np.copy(parent)


    def mutate(self, child):
        if calculate_budget(self.A_org,child) == self.delta_g:
            return child
        mutated = []
        for i in range(self.nG):
            if np.random.rand() < mutate_rate:
                for k in range(int(self.delta_l[0])//2):
                    indices = np.random.choice(self.nG, 2, replace = False)
                    for j in indices:
                        if i in mutated or j in mutated:
                            continue
                        child[i,j], child[j,i] = 1 - child[i,j], 1 - child[j,i]
                        mutated.append(i)
                        mutated.append(j)
                        if check_local(self.A_org[i],child[i]) <= int(self.delta_l[0]) and check_local(self.A_org[j],child[j]) <= int(self.delta_l[0]) and \
                        calculate_budget(self.A_org,child) <= self.delta_g:
                            break
                        else:
                            child[i,j], child[j,i] = 1 - child[i,j], 1 - child[j,i]
        np.fill_diagonal(child, 0)
        return child

    
    def evolve(self):
        fitness = self.fitness()
        if self.solution is not None:
            return
        pop = self.select(fitness)

        new_pop_list = []
        for parent in pop:
            child = self.crossover(parent, pop)
            child = self.mutate(child)
            new_pop_list.append(child)

        self.population = new_pop_list
    
    def attack(self, rounds):
        for i in range(rounds):
            self.evolve()
        if self.solution!=None:
            return self.solution
        else:
            return self.sol








