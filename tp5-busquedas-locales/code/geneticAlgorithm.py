import random
import environment
import time as t
import numpy as np
from typing import List, Tuple

def fitness_function(solution: List[int]) -> float:
    """Calcula el fitness de una solución."""
    return 1 / (1 + environment.objective_function(solution))

def generate_initial_population(population_size: int, solution_length: int) -> List[List[int]]:
    """Genera una población inicial más diversa usando numpy."""
    return [np.random.randint(0, solution_length, size=solution_length).tolist() 
            for _ in range(population_size)]

def reproduce(solution_x: List[int], solution_y: List[int]) -> List[int]:
    """Implementa crossover de dos puntos para mejor diversidad genética."""
    N = len(solution_x)
    point1, point2 = sorted(random.sample(range(N), 2))
    child = solution_x[:point1] + solution_y[point1:point2] + solution_x[point2:]
    return child

def mutate(solution: List[int], mutation_rate: float = 0.05) -> List[int]:
    """Implementa mutación con probabilidad por gen."""
    N = len(solution)
    mutated = solution.copy()
    for i in range(N):
        if random.random() < mutation_rate:
            mutated[i] = random.randint(0, N-1)
    return mutated

def tournament_selection(population: List[List[int]], k: int) -> List[int]:
    """Selección por torneo mejorada."""
    contestants = random.sample(population, k)
    return max(contestants, key=fitness_function)

def genetic_algorithm(initial_solution: List[int], 
                     population_size: int = 100,
                     tournament_size: int = 3,
                     mutation_rate: float = 0.05,
                     max_generations: int = 1000,
                     convergence_generations: int = 15) -> Tuple[List[int], float, int, float]:
    """
    Algoritmo genético optimizado con:
    - Elitismo
    - Criterio de convergencia
    """
    start = t.time()
    solution_length = len(initial_solution)
    
    # Genera población inicial
    population = generate_initial_population(population_size, solution_length)
    
    # Seguimiento del mejor individuo
    best_solution = initial_solution
    best_fitness = fitness_function(best_solution)
    
    # Variables para convergencia
    generations_without_improvement = 0
    generation = 0
    
    while (best_fitness != 1 and 
           generation < max_generations and 
           generations_without_improvement < convergence_generations):
        
        # Evaluar población actual y mantener élite
        population_with_fitness = [(ind, fitness_function(ind)) for ind in population]
        population_with_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Mantener el 10% mejor
        elite_size = population_size // 10
        elite = [ind for ind, _ in population_with_fitness[:elite_size]]
        
        # Verificar si hay mejora
        current_best_fitness = population_with_fitness[0][1]
        if current_best_fitness > best_fitness:
            best_solution = population_with_fitness[0][0]
            best_fitness = current_best_fitness
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1
        
        # Generar nueva población
        new_population = elite.copy()
        
        # Generar el resto de la población
        while len(new_population) < population_size:
            # Selección de padres por torneo
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)
            
            # Crossover y mutación
            child = reproduce(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        
        population = new_population
        generation += 1
        
        if generations_without_improvement > 20:
            mutation_rate = min(0.25, mutation_rate * 1.05) 
        else:
            mutation_rate = max(0.01, mutation_rate * 0.95)  
    
    return (best_solution, 
            environment.objective_function(best_solution), 
            generation, 
            t.time() - start)