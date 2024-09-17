import environment
import hillClimbing
import simulatedAnnealing
import geneticAlgorithm

solucion_inicial = [5, 6, 3, 4, 1, 5, 2, 6]

print(hillClimbing.hill_climbing(solucion_inicial))
print(simulatedAnnealing.simmulated_annealing(solucion_inicial, 0.99, 0.01))
print(geneticAlgorithm.geneticAlgorithm(solucion_inicial))