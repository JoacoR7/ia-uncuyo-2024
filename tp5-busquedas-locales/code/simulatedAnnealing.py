import random
import math
import environment
import time
                
def generate_neighbor(solution):
    neighbor = solution.copy()
    N = len(solution)
    column = random.randint(0, N - 1)
    row = solution[column]
    while row == solution[column]: 
        row = random.randint(0, N - 1)
    neighbor[column] = row
    return neighbor

def geometric_cooling(temperature, alpha):
    return temperature*alpha

def probability_of_acceptance(new_solution, actual_solution, temperature):
    delta_e = environment.objective_function(new_solution) - environment.objective_function(actual_solution)
    return math.exp(-delta_e / temperature)

def simmulated_annealing(solution, alpha, minimum_temperature):
    start = time.time()
    actual_solution = solution
    actual_solution_cost = environment.objective_function(actual_solution)
    fitness_over_time = [actual_solution_cost]
    temperature = 100
    iterations = 1
    while temperature > minimum_temperature or actual_solution_cost != 0 and iterations < 1000:
        new_solution = generate_neighbor(actual_solution)
        actual_solution_cost = environment.objective_function(actual_solution)
        new_solution_cost = environment.objective_function(new_solution)
        if(new_solution_cost <= actual_solution_cost):
            actual_solution = new_solution
            actual_solution_cost = new_solution_cost
        else:
            random_number = random.random()
            probability = probability_of_acceptance(new_solution, actual_solution, temperature)
            if(probability >= random_number):
                actual_solution = new_solution
                actual_solution_cost = new_solution_cost
        fitness_over_time.append(actual_solution_cost)
        temperature = geometric_cooling(temperature, alpha)
        iterations += 1
    return actual_solution, actual_solution_cost, iterations, time.time() - start, fitness_over_time
    