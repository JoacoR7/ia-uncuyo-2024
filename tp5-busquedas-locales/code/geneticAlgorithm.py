import random
import environment

def fitness_function(solution):
    return 1 / (1 + environment.objective_function(solution))

def generate_population(population_size, environment_size):
    population = []
    for i in range(population_size):
        solution = []
        for j in range(environment_size):
            solution.append(random.randint(0, environment_size-1))
        population.append(solution)
        solution = []
    return population

import random

def reproduce(solution_x, solution_y):
    crossover = random.randint(1, len(solution_x)-1) 
    child_solution = solution_x[:crossover] + solution_y[crossover:]
    return child_solution

    
def mute(solution):
    index = random.randint(0, len(solution) - 1)
    initial_row = solution[index]
    while solution[index] == initial_row:
        solution[index] = random.randint(0, len(solution)-1)
    return solution

def random_selection(population):
    solution = []
    while solution == []:
        pre_solution = population[random.randint(0, len(population)-1)]
        if(fitness_function(pre_solution) >= random.random()):
            solution = pre_solution
    return solution
            
    
def geneticAlgorithm(solution):
    actual_population = generate_population(100, len(solution))
    actual_fit = fitness_function(solution)
    actual_solution = solution
    time = 0
    while actual_fit != 1 and time < 100:
        new_population = []
        for i in range(len(actual_population)):
            solution_x = random_selection(actual_population)
            solution_y = random_selection(actual_population)
            child = reproduce(solution_x, solution_y)
            if(0.05 >= random.random()):
                mute(child)
            new_population.append(child)
            child_fit = fitness_function(child)
            if(actual_fit < child_fit):
                actual_solution = child
                actual_fit = child_fit
        time += 1
        actual_population = new_population
    return actual_solution, environment.objective_function(actual_solution)
                