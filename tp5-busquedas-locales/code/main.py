import environment
import hillClimbing
import simulatedAnnealing
import geneticAlgorithm
import data
import analysis


env_sizes = [4, 8, 10, 12, 15]
algorithm_names = ["hill_climbing", "simulated_annealing", "genetic_algorithm"]
"""data.initialize_csv()

hill_climbing_results = []
simulated_annealing_results = []
genetic_algorithm_results = []

for i in range(len(env_sizes)):
    for _ in range(30):
        solucion_inicial = environment.generate_initial_solution(env_sizes[i])
        hill_climbing_results.append(hillClimbing.hill_climbing(solucion_inicial))
        #data.save_results("hill_climbing", results[1], results[2], results[3], results[0])
        simulated_annealing_results.append(simulatedAnnealing.simmulated_annealing(solucion_inicial, 0.99, 0.01))
        #data.save_results("simulated_annealing", results[1], results[2], results[3], results[0])
        genetic_algorithm_results.append(geneticAlgorithm.genetic_algorithm(solucion_inicial))
        #data.save_results("genetic_algorithm", results[1], results[2], results[3], results[0])

results = [hill_climbing_results, simulated_annealing_results, genetic_algorithm_results]

for i in range(3):
    for j in range(150):
        algorithm = results[i]
        result = algorithm[j]
        data.save_results(algorithm_names[i], len(result[0]), result[1], result[2], result[3], result[0])
     """   

solucion_inicial = environment.generate_initial_solution(10)
objective_function_results = []
objective_function_results.append(hillClimbing.hill_climbing(solucion_inicial)[4])
objective_function_results.append(simulatedAnnealing.simmulated_annealing(solucion_inicial, 0.99, 0.01)[4])
objective_function_results.append(geneticAlgorithm.genetic_algorithm(solucion_inicial)[4])
analysis.plotear_iteraciones(objective_function_results)
