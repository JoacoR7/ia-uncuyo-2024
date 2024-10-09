import environment
import hillClimbing
import simulatedAnnealing
import geneticAlgorithm
import time

solucion_inicial = environment.generate_initial_solution(15)

print("Hill climbing")
print(hillClimbing.hill_climbing(solucion_inicial))
print()
print("Simulated annealing")
print(simulatedAnnealing.simmulated_annealing(solucion_inicial, 0.99, 0.01))
print()
print("Genetic algorithm")
print(geneticAlgorithm.geneticAlgorithm(solucion_inicial))